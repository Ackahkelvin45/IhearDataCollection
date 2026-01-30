import hashlib
import json
import re
import logging
from typing import Dict, Any

from django.conf import settings
from django.core.cache import cache

logger = logging.getLogger(__name__)

# Canonical intents; validate LLM output against this
ALLOWED_INTENTS = {"NUMERIC", "EXPLANATORY", "MIXED"}
DEFAULT_INTENT = "EXPLANATORY"

# Cache: avoid re-calling LLM for same question + context in a short window
INTENT_CACHE_PREFIX = "chatbot:intent:"
INTENT_CACHE_TTL = 60 * 10  # 10 minutes

# Fallback keywords (word-boundary match to avoid "show" matching "how")
NUMERIC_KEYWORDS = [
    "how many", "count", "total", "number of", "average", "sum", "maximum", "minimum",
    "how much", "statistics", "dataset", "datasets", "recordings", "documents",
]
EXPLANATORY_KEYWORDS = ["what is", "explain", "tell me about", "who", "why", "when", "where", "how does"]


class IntentClassifier:
    """Classify user questions into NUMERIC, EXPLANATORY, or MIXED for routing."""

    def __init__(self):
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                model_name=settings.OPENAI_MODEL,
                openai_api_key=settings.OPENAI_API_KEY,
                temperature=0,
                max_tokens=150,
            )
        return self._llm

    def classify(self, question: str, context: Dict = None) -> Dict[str, Any]:
        """
        Classify intent using LLM with optional context (conversation_length, recent_topics, last_intent).
        Results are cached by (question_hash + last_intent) for INTENT_CACHE_TTL.
        Falls back to keyword-based classification if LLM fails.
        """
        context = context or {}
        question_lower = (question or "").strip().lower()
        if not question_lower:
            return {"intent": DEFAULT_INTENT, "confidence": 0.5, "reasoning": "Empty question"}

        # Cache key: same question + same last_intent → reuse result
        last_intent = context.get("last_intent") or ""
        cache_key = f"{INTENT_CACHE_PREFIX}{hashlib.md5((question_lower + last_intent).encode()).hexdigest()}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        # Try LLM classification first
        try:
            intents_list = ", ".join(sorted(ALLOWED_INTENTS))
            prompt = f"""Classify the user question into exactly one intent.

Question: {question[:500]}

Context (optional): conversation_length={context.get('conversation_length', 0)}, last_intent={context.get('last_intent')}

Allowed intents only: {intents_list}
- NUMERIC: counts, totals, averages, statistics, "how many", database-style queries
- EXPLANATORY: explanations, "what is", "tell me about", document-based or conversational
- MIXED: clearly needs both a number and an explanation

Return JSON only, no markdown:
{{"intent": "NUMERIC" or "EXPLANATORY" or "MIXED", "confidence": 0.0-1.0, "reasoning": "one short phrase"}}"""

            llm = self._get_llm()
            response = llm.invoke(prompt)
            text = (response.content if hasattr(response, "content") else str(response)).strip()
            # Strip markdown code block if present
            if text.startswith("```"):
                text = re.sub(r"^```\w*\n?", "", text).strip()
                text = re.sub(r"\n?```\s*$", "", text).strip()
            data = json.loads(text)
            intent = (data.get("intent") or DEFAULT_INTENT).upper().strip()
            if intent not in ALLOWED_INTENTS:
                intent = DEFAULT_INTENT
            confidence = _safe_confidence(data.get("confidence", 0.8))
            result = {
                "intent": intent,
                "confidence": confidence,
                "reasoning": str(data.get("reasoning", ""))[:200],
            }
            cache.set(cache_key, result, INTENT_CACHE_TTL)
            return result
        except Exception as e:
            logger.debug("Intent LLM fallback: %s", e)

        # Keyword-based fallback (not cached; cheap)
        return self._classify_by_keywords(question_lower)

    def _classify_by_keywords(self, question_lower: str) -> Dict[str, Any]:
        # Word-boundary match so "show" doesn't match "how", etc.
        has_numeric = any(re.search(rf"\b{re.escape(k)}\b", question_lower) for k in NUMERIC_KEYWORDS)
        has_explanatory = any(re.search(rf"\b{re.escape(k)}\b", question_lower) for k in EXPLANATORY_KEYWORDS)
        if has_numeric and has_explanatory:
            return {"intent": "MIXED", "confidence": 0.7, "reasoning": "keywords: both numeric and explanatory"}
        if has_numeric:
            return {"intent": "NUMERIC", "confidence": 0.8, "reasoning": "keywords: numeric"}
        return {"intent": "EXPLANATORY", "confidence": 0.8, "reasoning": "keywords: explanatory or default"}

    def get_routing_info(self, question: str, context: Dict = None) -> Dict[str, Any]:
        """API used by views and DatasetService: same shape as classify()."""
        return self.classify(question, context or {})

    def classify_intent(self, question: str) -> str:
        """Return only the intent string (for dataset_service / helpers)."""
        return self.get_routing_info(question).get("intent", DEFAULT_INTENT)


def get_question_routing(question: str, context: Dict = None) -> Dict[str, Any]:
    """Standalone helper: classify and return routing info."""
    classifier = IntentClassifier()
    return classifier.classify(question, context or {})


def _safe_confidence(value: Any) -> float:
    """Parse confidence from LLM (may be str like 'high') and clamp to [0, 1]."""
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 0.8
    return max(0.0, min(confidence, 1.0))


def classify_question_intent(question: str) -> str:
    """Return only the intent string (used by __init__ and callers)."""
    return IntentClassifier().classify_intent(question)
