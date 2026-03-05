"""Storage backends for SemanticCache."""
from llm_semantic_cache.storage.base import CacheEntry, StorageBackend
from llm_semantic_cache.storage.memory import InMemoryStorage

__all__ = ["CacheEntry", "StorageBackend", "InMemoryStorage"]

try:
    from llm_semantic_cache.storage.redis import RedisStorage

    __all__ = [*__all__, "RedisStorage"]
except ImportError:
    pass
