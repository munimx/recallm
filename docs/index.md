# Recallm

Semantic caching for LLMs. Ask once, recall forever.

## Why semantic caching?

Exact-match caching is useless for LLMs — two users asking the same question in slightly different words both pay the full API cost. Recallm uses sentence embeddings to find near-matches and return cached responses instantly. The result: lower API costs, reduced latency, and no changes to your existing LLM client code.

## Install

```bash
pip install recallm
pip install "recallm[redis]"   # persistent cache, shared across workers
pip install "recallm[torch]"   # sentence-transformers embedder (700MB, PyTorch)
```

## Quickstart

```python
from openai import OpenAI
from llm_semantic_cache import SemanticCache, CacheConfig, InMemoryStorage

client = OpenAI()
cache = SemanticCache(
    storage=InMemoryStorage(),
    config=CacheConfig(threshold=0.92),
)
create = cache.wrap(client.chat.completions.create)

# First call: cache miss — LLM is called
response = create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    cache_context={"user_id": "u123"},
)

# Second call: semantically equivalent — cache hit, no LLM call
response = create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Capital of France?"}],
    cache_context={"user_id": "u123"},
)
```

!!! note "Async frameworks"
    `SemanticCache(...)` loads the embedding model synchronously. In FastAPI or other
    async frameworks, use `await cache.async_warmup()` during startup instead.

## Expected hit rates

| Use case | Expected hit rate | Why |
|---|---|---|
| FAQ / support bot | 40–70% | High repetition, forgiving similarity |
| Document summarization | 20–50% | Same docs re-processed, template prompts |
| General chat assistant | 5–15% | High diversity, dynamic context |
| Code generation | 3–10% | Exact problem statements vary, strict threshold |

## Known limitations

- `stream=True` bypasses the cache entirely
- Redis backend is not suitable for namespaces > 5,000 entries without partitioning
- Sync callers using `RedisStorage` have no timeout protection (v0.1.0)

## Next steps

- [Getting started](getting-started.md) — full walkthrough with `cache_context` explanation
- [Configuration](configuration.md) — `CacheConfig` reference and threshold tuning
- [Storage backends](storage-backends.md) — Redis setup and scaling notes
