"""llm-semantic-cache — semantic caching for OpenAI-compatible LLM APIs."""
from llm_semantic_cache.cache import SemanticCache
from llm_semantic_cache.config import CacheConfig
from llm_semantic_cache.storage.base import CacheEntry, SearchResult, StorageBackend
from llm_semantic_cache.storage.memory import InMemoryStorage

__all__ = [
    "SemanticCache",
    "CacheConfig",
    "CacheEntry",
    "SearchResult",
    "StorageBackend",
    "InMemoryStorage",
]

try:
    from llm_semantic_cache.storage.redis import RedisStorage
    __all__.append("RedisStorage")
except ImportError:
    pass
