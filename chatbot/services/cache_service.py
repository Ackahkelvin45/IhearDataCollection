import hashlib
from typing import Optional, Dict, Any
from django.core.cache import cache
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class CacheService:
    """Service for managing query caching"""

    def __init__(self):
        self.default_timeout = 3600  # 1 hour
        self.cache_prefix = "chatbot"

    def get_query_cache_key(
        self, query: str, context: Optional[str] = None
    ) -> str:
        """
        Generate cache key for a query
        
        Args:
            query: User's query
            context: Additional context (e.g., chat history)
            
        Returns:
            Cache key
        """
        combined = f"{query}:{context or ''}"
        hash_key = hashlib.md5(combined.encode()).hexdigest()
        return f"{self.cache_prefix}:query:{hash_key}"

    def get_cached_response(
        self, query: str, context: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached response for a query
        
        Args:
            query: User's query
            context: Additional context
            
        Returns:
            Cached response or None
        """
        cache_key = self.get_query_cache_key(query, context)
        cached_data = cache.get(cache_key)
        
        if cached_data:
            logger.info(f"Cache hit for query: {query[:50]}")
            # Increment hit count if tracking
            cached_data["cache_hit"] = True
            
        return cached_data

    def set_cached_response(
        self,
        query: str,
        response: Dict[str, Any],
        context: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        """
        Cache a query response
        
        Args:
            query: User's query
            response: Response to cache
            context: Additional context
            timeout: Cache timeout in seconds
        """
        cache_key = self.get_query_cache_key(query, context)
        timeout = timeout or self.default_timeout
        
        cache_data = {
            **response,
            "cached_at": datetime.now().isoformat(),
            "cache_hit": False,
        }
        
        cache.set(cache_key, cache_data, timeout=timeout)
        logger.info(f"Cached response for query: {query[:50]}")

    def invalidate_document_cache(self, doc_id: str):
        """
        Invalidate all cached queries related to a document
        
        Note: This is a simple implementation. For production,
        consider using cache tags or a more sophisticated system.
        """
        # Since we can't easily list all keys, we'll just clear the entire cache
        # In production, use Redis SCAN or maintain a separate index
        logger.warning(
            f"Document {doc_id} updated - consider clearing related cache"
        )

    def clear_user_cache(self, user_id: str):
        """Clear all cached queries for a user"""
        # Placeholder - in production, implement proper user-specific cache clearing
        logger.info(f"Cache clear requested for user {user_id}")

    def get_session_cache_key(self, session_id: str) -> str:
        """Get cache key for session data"""
        return f"{self.cache_prefix}:session:{session_id}"

    def cache_session_context(
        self, session_id: str, context: Dict[str, Any], timeout: int = 1800
    ):
        """
        Cache session context (30 min default)
        
        Args:
            session_id: Chat session ID
            context: Session context to cache
            timeout: Cache timeout in seconds
        """
        cache_key = self.get_session_cache_key(session_id)
        cache.set(cache_key, context, timeout=timeout)

    def get_session_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get cached session context"""
        cache_key = self.get_session_cache_key(session_id)
        return cache.get(cache_key)

    def invalidate_session_cache(self, session_id: str):
        """Invalidate cached session context"""
        cache_key = self.get_session_cache_key(session_id)
        cache.delete(cache_key)
        logger.info(f"Invalidated cache for session {session_id}")

