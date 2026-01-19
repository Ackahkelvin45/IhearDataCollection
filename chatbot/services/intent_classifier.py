import logging
from typing import Dict, Any, Optional
from django.conf import settings

logger = logging.getLogger(__name__)


class IntentClassifier:
    """Classify user questions into different intent types for routing"""
    
    # Keywords that indicate numeric/exact queries
    NUMERIC_KEYWORDS = [
        # Counting
        "how many", "count", "total", "sum", "number of",
        # Aggregation
        "average", "mean", "median", "max", "min", "maximum", "minimum",
        "highest", "lowest", "top", "bottom", "most", "least",
        # Filtering/Grouping
        "group by", "filter", "where", "sort by", "order by",
        # Time-based
        "in month", "in year", "per month", "per year", "daily", "weekly", "monthly",
        # Comparisons
        "compare", "vs", "versus", "difference between",
        # Statistics
        "percentage", "percent", "ratio", "rate", "frequency",
        # Specific data operations
        "list all", "show me", "find", "search for"
    ]
    
    # Keywords that indicate explanatory queries
    EXPLANATORY_KEYWORDS = [
        # Analysis/Understanding
        "why", "explain", "analyze", "understand", "tell me about",
        "what happened", "how did", "what caused", "reason for",
        # Patterns/Trends
        "trend", "pattern", "insight", "observation", "finding",
        "correlation", "relationship", "connection",
        # Recommendations/Advice
        "recommend", "suggest", "should", "advice", "best practice",
        # Qualitative
        "describe", "summary", "overview", "breakdown",
        # Complex analysis
        "impact of", "effect of", "influence of", "role of"
    ]
    
    # Keywords that indicate mixed queries (both numeric and explanatory)
    MIXED_KEYWORDS = [
        "and why", "but why", "explain why", "tell me why",
        "what's the reason", "why is it", "why are they"
    ]
    
    def classify_intent(self, question: str) -> str:
        """
        Classify the intent of a question
        
        Returns:
            'NUMERIC' - for counting, filtering, aggregation queries
            'EXPLANATORY' - for analysis, patterns, explanations
            'MIXED' - for questions needing both numbers and explanation
        """
        question_lower = question.lower().strip()
        
        # Check for mixed intent first (contains both numeric and explanatory elements)
        has_numeric = any(keyword in question_lower for keyword in self.NUMERIC_KEYWORDS)
        has_explanatory = any(keyword in question_lower for keyword in self.EXPLANATORY_KEYWORDS)
        has_mixed = any(keyword in question_lower for keyword in self.MIXED_KEYWORDS)
        
        # Explicit mixed indicators
        if has_mixed:
            return 'MIXED'
        
        # Questions that have both numeric and explanatory elements
        if has_numeric and has_explanatory:
            return 'MIXED'
        
        # Pure numeric queries
        if has_numeric:
            return 'NUMERIC'
        
        # Pure explanatory queries
        if has_explanatory:
            return 'EXPLANATORY'
        
        # Default to explanatory for ambiguous cases
        return 'EXPLANATORY'
    
    def get_routing_info(self, question: str) -> Dict[str, Any]:
        """
        Get complete routing information for a question
        
        Returns:
            {
                'intent': 'NUMERIC'|'EXPLANATORY'|'MIXED',
                'confidence': float,
                'reasoning': str,
                'suggested_tools': list
            }
        """
        intent = self.classify_intent(question)
        
        routing_info = {
            'intent': intent,
            'confidence': 0.8,  # Simplified confidence score
            'reasoning': self._get_reasoning(intent, question),
            'suggested_tools': self._get_suggested_tools(intent)
        }
        
        return routing_info
    
    def _get_reasoning(self, intent: str, question: str) -> str:
        """Generate reasoning for the classification"""
        if intent == 'NUMERIC':
            return f"Question contains numeric operations like counting, averaging, or filtering: '{question}'"
        elif intent == 'EXPLANATORY':
            return f"Question seeks explanation, analysis, or patterns: '{question}'"
        elif intent == 'MIXED':
            return f"Question requires both specific data and explanatory analysis: '{question}'"
        return f"Defaulted to explanatory analysis for: '{question}'"
    
    def _get_suggested_tools(self, intent: str) -> list:
        """Get suggested tools for the intent"""
        if intent == 'NUMERIC':
            return ['SQL', 'Pandas', 'Database Query']
        elif intent == 'EXPLANATORY':
            return ['RAG', 'Document Analysis', 'Pattern Recognition']
        elif intent == 'MIXED':
            return ['SQL + RAG', 'Database + Analysis', 'Hybrid Query']
        return ['RAG', 'General Analysis']


# Convenience function for easy use
def classify_question_intent(question: str) -> str:
    """Quick function to classify question intent"""
    classifier = IntentClassifier()
    return classifier.classify_intent(question)


def get_question_routing(question: str) -> Dict[str, Any]:
    """Quick function to get routing information"""
    classifier = IntentClassifier()
    return classifier.get_routing_info(question)
