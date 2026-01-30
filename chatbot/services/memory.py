from django.core.cache import cache

SESSION_TTL = 60 * 60 * 6  # 6 hours
CACHE_PREFIX = "chatbot:memory:"


def get_conversation_memory(session_id: str) -> dict:
    """Get conversation memory for a session (namespaced to avoid collisions)."""
    key = f"{CACHE_PREFIX}{session_id}"
    return cache.get(key, _default_memory())


def save_conversation_memory(session_id: str, memory: dict) -> None:
    """Persist conversation memory for a session."""
    key = f"{CACHE_PREFIX}{session_id}"
    cache.set(key, memory, SESSION_TTL)


def _default_memory() -> dict:
    return {
        "message_count": 0,
        "topics": [],
        "last_intent": None,
        "last_question": None,
        "last_answer": None,
        "preferences": {},
        "context_flags": [],
    }
