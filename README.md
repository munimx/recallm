# LLM Semantic Cache

A Python library that adds **semantic caching** in front of any OpenAI-compatible LLM API.

When your application sends a prompt, the library checks whether a semantically similar prompt has been answered before. If yes, the cached response is returned immediately. If no, the request goes to the provider, the response is stored, and future similar prompts benefit.

---

## The Problem

Exact-match caching (hashing the prompt string) is nearly useless for LLM workloads — prompts are dynamic by nature. Semantic caching, using embeddings to find near-matches, directly cuts API costs with a measurable result.

---

## Installation

```bash
pip install llm-semantic-cache
```

For persistent caching with Redis:
```bash
pip install "llm-semantic-cache[redis]"
```

For sentence-transformers (torch-based embeddings, ~700MB):
```bash
pip install "llm-semantic-cache[torch]"
```

The default embedder uses `fastembed` (ONNX, ~20MB) — no PyTorch required.

---

## Quickstart

```python
from openai import OpenAI
from llm_semantic_cache import SemanticCache, CacheConfig, InMemoryStorage

client = OpenAI()
cache = SemanticCache(
    storage=InMemoryStorage(),
    config=CacheConfig(threshold="balanced"),
)
cached_create = cache.wrap(client.chat.completions.create)

# First call — goes to OpenAI
response = cached_create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is Python?"}],
    cache_context={"user_id": "u123"},
)

# Second call with a similar prompt — returned from cache (no API call)
response2 = cached_create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Tell me about the Python language"}],
    cache_context={"user_id": "u123"},
)
```

---

## API Reference

### `SemanticCache`

```python
SemanticCache(
    storage: StorageBackend,
    config: CacheConfig | None = None,  # defaults to CacheConfig()
    embedder: Embedder | None = None,   # defaults to FastEmbedEmbedder
)
```

**`.wrap(fn, *, mode="auto")`** — Returns a wrapped callable. The wrapper accepts the same arguments as `fn` plus two optional kwargs:
- `cache_context: dict` **(required)** — Context that scopes the cache lookup. Pass `{}` if there is no context.
- `cache_namespace: str` — Namespace for this call (default: `config.default_namespace`).

**`.invalidate_namespace(namespace)`** — Delete all cached entries in a namespace.

### `CacheConfig`

| Field | Default | Description |
|-------|---------|-------------|
| `threshold` | `"balanced"` | `"strict"` (0.97), `"balanced"` (0.92), `"loose"` (0.85), or raw float |
| `default_namespace` | `"default"` | Default namespace for all cache entries |
| `default_ttl` | `None` | Entry TTL in seconds (`None` = no expiration) |
| `embedding_model` | `"all-MiniLM-L6-v2"` | Embedding model name |
| `cache_timeout_seconds` | `0.05` | Hard timeout for cache operations (fail-open on timeout) |

### Threshold Profiles

| Profile | Threshold | Use Case |
|---------|-----------|----------|
| `strict` | 0.97 | Code generation, factual Q&A |
| `balanced` | 0.92 | General assistants, summarization |
| `loose` | 0.85 | Customer support, FAQ bots |

---

## Storage Backends

### In-Memory (default)

```python
from llm_semantic_cache import InMemoryStorage
storage = InMemoryStorage()
```

Zero dependencies. Data is lost on process restart.

### Redis

```python
import redis.asyncio as aioredis
from llm_semantic_cache import RedisStorage  # requires [redis] extra

client = aioredis.Redis.from_url("redis://localhost:6379")
storage = RedisStorage(client)
```

Persistent across restarts. Shared across workers.

**Scaling note:** The Redis backend performs client-side cosine similarity. Performance degrades when a namespace exceeds ~5,000 entries. Use namespace partitioning for larger workloads.

---

## Architecture

```
Your Application
      │
      ▼
┌─────────────────────┐
│   SemanticCache      │  wrap() intercepts the call
│                     │
│  1. Embed prompt    │  local fastembed model, runs in-process
│  2. Filter index    │  by namespace + model_id + context_hash
│  3. Cosine search   │  threshold check
│     hit → return    │  cached response, no LLM call
│     miss → forward  │  original call proceeds
│  4. Store result    │  embedding + response stored
└────────┬────────────┘
         │ on miss
         ▼
  Your LLM Provider
```

---

## Known Limitations

**Streaming:** `stream=True` requests bypass the cache entirely in the current version.

**Redis scale:** The Redis backend is not suitable for namespaces with >5,000 entries without partitioning.

**Context sensitivity:** Always pass `cache_context` with any session, document, or user state that affects the expected answer. The library enforces this — `cache_context` is a required argument.

---

## Expected Hit Rates

| Use Case | Expected Hit Rate | Why |
|----------|-------------------|-----|
| FAQ / support bot | 40–70% | High repetition, forgiving threshold |
| Document summarization | 20–50% | Same docs re-processed |
| General chat | 5–15% | High prompt diversity |
| Code generation | 3–10% | Exact problem statements vary |

If semantic caching does not help your workload, the library will tell you via its metrics. It does not oversell itself.

---

## Status

Active development. See `AGENTS.md` for the full project specification.
