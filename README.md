# LLM Semantic Cache

A Python library that adds **semantic caching** in front of any OpenAI-compatible LLM API.

When your application sends a prompt, instead of always forwarding it to the model, the library first checks whether a semantically similar prompt has been asked before. If yes, the cached response is returned immediately. If no, the request is forwarded, the response is stored, and future similar prompts benefit.

---

## The Problem

Exact-match caching (hashing the prompt string) is nearly useless for LLM workloads — prompts are dynamic by nature. Semantic caching, using embeddings to find near-matches, is the approach that actually works. LLM inference costs are typically the first thing teams optimize after shipping. A semantic cache directly cuts API bills with a concrete, measurable result.

---

## Architecture

```
Your Application
      │
      ▼
┌─────────────────────┐
│   SemanticCache      │  wrap() intercepts the call
│                     │
│  1. Embed prompt    │  local sentence-transformers model, runs in-process
│  2. Search index    │  cosine similarity against stored embeddings
│  3. Threshold check │  configurable per profile or raw float
│     hit → return    │  cached response returned immediately, no LLM call
│     miss → forward  │  original API call proceeds normally
│  4. Store result    │  embedding + response stored for future hits
└────────┬────────────┘
         │ on miss
         ▼
  Your LLM Provider
  (OpenAI / Anthropic / vLLM / Ollama / anything OpenAI-compatible)
```

---

## Design Decisions

### Local embeddings — no external dependency

The library uses `all-MiniLM-L6-v2` via `sentence-transformers`, running in-process. No embedding API calls. Small (80MB), fast (sub-10ms on CPU), good enough for semantic similarity on short to medium length text.

### Named threshold profiles

| Profile | Threshold | Intended for |
|---------|-----------|--------------|
| `strict` | 0.97 | Code generation, factual Q&A |
| `balanced` | 0.92 | General assistants, summarization |
| `loose` | 0.85 | Customer support, FAQ bots |

Override with a raw float for full control.

### Namespace-based cache invalidation

TTL alone is not sufficient. Cache entries are tagged at write time; entire namespaces can be invalidated in one operation without scanning or expiring unrelated entries.

### Pluggable storage

- **In-memory** — zero dependencies, good for development, lost on restart
- **Redis** — persistent, shared across workers and replicas

---

## What This Is Not

- Not a proxy, load balancer, or rate limiter
- Not a replacement for LiteLLM
- Not vLLM-specific

---

## Status

Early development. See [PROJECT_DIRECTION2.md](https://github.com/munimx/LLM-Inference-Optimization-Engine/blob/main/PROJECT_DIRECTION2.md) in the previous repo for the full project rationale and scope.

---

## Plan Addendum — Review Resolutions

The following addresses five issues raised during plan review. Each resolution is a binding design decision for implementation.

---

### Issue 1: Redis Scalability — ACCEPTED WITH MITIGATION

**Problem:** The Redis backend fetches all vectors for a namespace to Python for client-side cosine similarity. At scale (10k+ entries), this transfers ~15MB per cache miss and is slower than the LLM call itself.

**Resolution:** Two changes.

**1a. Document the scaling boundary.** The Redis backend README and docstring will state:

> Redis storage is designed for small-to-medium namespaces (under ~5,000 entries). For larger workloads, a vector store backend (pgvector, Chroma) is on the roadmap. If your namespace exceeds this range, Redis will degrade — use namespace partitioning or wait for a vector-native backend.

**1b. Add candidate pre-filtering via quantized bucketing.** Before the full cosine similarity scan, reduce the candidate set using a coarse locality filter. Each embedding is assigned a bucket key at write time derived from quantized top-k dimensions. At query time, only entries sharing the same bucket (plus adjacent buckets) are fetched.

This lives in `storage/redis.py` as an internal optimization — the `StorageBackend` ABC does not change.

```python
# storage/redis.py — internal pre-filtering

def _compute_bucket_key(self, embedding: list[float], num_dims: int = 4, num_bins: int = 8) -> str:
    """Quantize the top-k embedding dimensions into a coarse bucket key."""
    top_dims = sorted(range(len(embedding)), key=lambda i: abs(embedding[i]), reverse=True)[:num_dims]
    bins = [str(int(embedding[d] * num_bins) % num_bins) for d in top_dims]
    return ":".join(bins)

async def search(self, namespace: str, embedding: list[float], ...) -> list[CacheEntry]:
    bucket = self._compute_bucket_key(embedding)
    # Fetch candidates from this bucket + adjacent buckets only
    candidate_keys = await self._get_bucket_candidates(namespace, bucket)
    # Fall back to full scan if bucket is empty (cold start)
    if not candidate_keys:
        candidate_keys = await self._get_all_keys(namespace)
    # Cosine similarity on reduced candidate set
    ...
```

**New test cases:**
- `test_redis_bucket_reduces_candidate_set` — verify fewer entries fetched than total namespace size.
- `test_redis_bucket_cold_start_falls_back` — verify full scan when bucket is empty.
- `test_redis_search_correctness_with_bucketing` — verify top result is identical with and without pre-filtering for a known dataset.

---

### Issue 2: Streaming (`stream=True`) — ACCEPTED, BYPASS IN V1

**Problem:** `wrap()` has no strategy for `stream=True` requests. A naive implementation will crash or silently drop the stream.

**Resolution:** In v1, **streaming requests bypass the cache entirely**. This is the simplest correct behavior and avoids the complexity of stream teeing, which introduces backpressure, error-handling, and memory concerns that are disproportionate for an initial release.

Behavior in `cache.py`:

```python
def _is_streaming_request(self, kwargs: dict) -> bool:
    return kwargs.get("stream", False) is True

# Inside wrap():
if self._is_streaming_request(kwargs):
    log.info("cache.stream_bypass", reason="stream=True not cached in v1")
    STREAM_BYPASS_COUNTER.inc()
    return await original_fn(*args, **kwargs)
```

**Guarantees:**
- `stream=True` calls are forwarded to the provider unchanged — no interception, no accumulation, no modification.
- A structlog event is emitted on every bypass so users can measure how much traffic skips the cache.
- A Prometheus counter `semantic_cache_stream_bypass_total` tracks bypass volume.

**Documentation will state:**
> Streaming responses (`stream=True`) bypass the cache in the current version. The request is forwarded directly to the provider. This is a known limitation — stream caching is planned for a future release.

**New test cases:**
- `test_wrap_stream_true_bypasses_cache` — verify no cache lookup or store occurs.
- `test_wrap_stream_true_returns_original_response` — verify the provider's streaming response is returned unmodified.
- `test_wrap_stream_bypass_emits_log_and_metric` — verify structlog event and counter increment.

---

### Issue 3: Fail-Open Resilience — ACCEPTED

**Problem:** If cache infrastructure (Redis connection, embedding model, serialization) throws an exception, `wrap()` propagates it to the caller — crashing the application even though the LLM call itself would have succeeded.

**Resolution:** All cache operations inside `wrap()` are wrapped in a fail-open guard. If any cache operation fails, the original function is called as if the cache does not exist.

```python
# cache.py — fail-open guard

async def _cached_call(self, original_fn, *args, **kwargs):
    # Attempt cache lookup
    try:
        hit = await self._lookup(kwargs)
        if hit is not None:
            return hit
    except Exception as exc:
        log.error("cache.lookup_failed", error=str(exc), exc_info=True)
        CACHE_ERROR_COUNTER.labels(operation="lookup").inc()
        # Fall through — call the provider

    # Call the original function (always happens on miss or error)
    response = await original_fn(*args, **kwargs)

    # Attempt cache store — failure must not affect the response
    try:
        await self._store(kwargs, response)
    except Exception as exc:
        log.error("cache.store_failed", error=str(exc), exc_info=True)
        CACHE_ERROR_COUNTER.labels(operation="store").inc()

    return response
```

**Invariant:** `wrap()` never raises an exception that the unwrapped function would not have raised. The cache is purely additive — its failure is invisible to the caller except via logs and metrics.

**New Prometheus metric:** `semantic_cache_errors_total` with label `operation` (`lookup` | `store` | `embed`).

**New test cases:**
- `test_wrap_returns_response_when_lookup_raises` — Redis connection error during lookup, provider is still called.
- `test_wrap_returns_response_when_store_raises` — provider succeeds, store fails, response is still returned.
- `test_wrap_returns_response_when_embedding_raises` — model load failure, falls through to provider.
- `test_fail_open_logs_error_with_structlog` — verify structured log fields on failure.

---

### Issue 4: Context Serialization Robustness — ACCEPTED

**Problem:** `json.dumps` in `context.py` will fail with `TypeError` on common Python types: `set`, `datetime`, `bytes`, `UUID`, Pydantic models. Users will pass these as context values and get cryptic errors.

**Resolution:** `context.py` will include a canonical serializer that handles all common non-JSON-serializable types deterministically. The serializer is used exclusively for context fingerprinting — it produces a stable string, not a round-trippable format.

```python
# context.py — canonical serializer

import datetime
import uuid
from collections.abc import Set

def _canonical_default(obj: object) -> object:
    """json.dumps default hook for deterministic context hashing."""
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    if isinstance(obj, datetime.date):
        return obj.isoformat()
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, bytes):
        return obj.hex()
    if isinstance(obj, Set):
        return sorted(str(item) for item in obj)  # deterministic order
    if isinstance(obj, frozenset):
        return sorted(str(item) for item in obj)
    if hasattr(obj, "model_dump"):  # Pydantic v2
        return obj.model_dump(mode="json")
    if hasattr(obj, "dict"):  # Pydantic v1
        return obj.dict()
    raise TypeError(f"Context value of type {type(obj).__name__} is not serializable. "
                    f"Convert it to a JSON-compatible type before passing it as context.")

def compute_context_hash(context: dict) -> str:
    """Deterministic hash of context dict for cache key construction."""
    serialized = json.dumps(context, sort_keys=True, default=_canonical_default, ensure_ascii=True)
    return hashlib.sha256(serialized.encode()).hexdigest()
```

**Design notes:**
- `sort_keys=True` ensures dict ordering does not affect the hash.
- Sets are sorted-then-stringified for deterministic ordering.
- The `TypeError` at the end of `_canonical_default` is intentional — unknown types should fail loudly with a clear message, not silently produce wrong hashes.
- `ensure_ascii=True` prevents encoding-dependent hash differences.

**New test cases:**
- `test_context_hash_with_datetime` — same datetime produces same hash.
- `test_context_hash_with_set` — `{1, 2, 3}` and `{3, 1, 2}` produce the same hash.
- `test_context_hash_with_uuid` — UUID context value serializes correctly.
- `test_context_hash_with_pydantic_model` — Pydantic v2 model in context values.
- `test_context_hash_with_bytes` — bytes value produces deterministic hash.
- `test_context_hash_unknown_type_raises_clear_error` — custom class raises `TypeError` with actionable message.
- `test_context_hash_deterministic_across_calls` — same input always produces same output.

---

### Issue 5: Sync/Async Wrapper Separation — ACCEPTED

**Problem:** `inspect.iscoroutinefunction()` is unreliable on decorated functions (e.g., functions wrapped by `@functools.wraps`, `@retry`, or framework decorators strip the coroutine flag). A single wrapper body that tries to handle both sync and async in one code path will produce subtle runtime failures.

**Resolution:** `wrap()` becomes a factory that inspects the target at decoration time and returns one of two distinct wrapper classes. The user can also force the mode explicitly to bypass detection entirely.

```python
# cache.py — wrapper factory

from typing import Literal

def wrap(
    self,
    fn: Callable,
    *,
    mode: Literal["auto", "sync", "async"] = "auto",
    **cache_opts,
) -> Callable:
    """Wrap a callable with semantic caching.

    Args:
        fn: The function to wrap.
        mode: "auto" detects sync/async (default). "sync" or "async"
              forces the wrapper type — use when auto-detection fails
              on heavily decorated functions.
    """
    if mode == "async" or (mode == "auto" and self._is_async(fn)):
        return self._make_async_wrapper(fn, **cache_opts)
    return self._make_sync_wrapper(fn, **cache_opts)

def _is_async(self, fn: Callable) -> bool:
    """Best-effort async detection, checking through common decorator layers."""
    unwrapped = inspect.unwrap(fn)  # follows __wrapped__ chain
    return inspect.iscoroutinefunction(unwrapped)

def _make_async_wrapper(self, fn: Callable, **cache_opts) -> Callable:
    @functools.wraps(fn)
    async def async_wrapper(*args, **kwargs):
        return await self._cached_call(fn, *args, **kwargs)
    async_wrapper.__wrapped__ = fn
    return async_wrapper

def _make_sync_wrapper(self, fn: Callable, **cache_opts) -> Callable:
    @functools.wraps(fn)
    def sync_wrapper(*args, **kwargs):
        return self._cached_call_sync(fn, *args, **kwargs)
    sync_wrapper.__wrapped__ = fn
    return sync_wrapper
```

**Key decisions:**
- `inspect.unwrap()` follows the `__wrapped__` chain before checking, which handles `@functools.wraps`-based decorators correctly.
- The `mode` parameter provides an explicit escape hatch — if auto-detection fails, the user sets `mode="async"` and it works. No guessing.
- Sync wrapper calls `_cached_call_sync`, which uses synchronous Redis/storage operations. Async wrapper calls `_cached_call`, which uses `await`. These are **separate code paths** — no `asyncio.run()` or `loop.run_until_complete()` bridging.
- Both wrappers set `__wrapped__` so downstream decorators can continue unwrapping.

**New test cases:**
- `test_wrap_auto_detects_async_function` — plain `async def` is wrapped as async.
- `test_wrap_auto_detects_sync_function` — plain `def` is wrapped as sync.
- `test_wrap_auto_detects_decorated_async` — `@functools.wraps`-decorated async function is correctly detected.
- `test_wrap_mode_override_forces_async` — `mode="async"` on a sync-looking function produces async wrapper.
- `test_wrap_mode_override_forces_sync` — `mode="sync"` on an async function produces sync wrapper.
- `test_sync_wrapper_does_not_use_event_loop` — sync wrapper never calls `asyncio.run` or touches an event loop.
- `test_async_wrapper_is_awaitable` — return value of async wrapper is a coroutine.
