import re
import difflib
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dateutil import parser as dateutil_parser

from django.conf import settings
from langchain_core.messages import HumanMessage
from loguru import logger

from core.models import Region, Community
from data.models import NoiseDataset


def _is_clarification_v2_enabled() -> bool:
    """Check whether Phase 2 clarification improvements are active."""
    try:
        return settings.AI_INSIGHT.get("AGENT", {}).get(
            "CLARIFICATION_V2_ENABLED", False
        )
    except Exception:
        return False


# ── Analysis type labels ────────────────────────────────────────────

ANALYSIS_TYPE_LABELS = {
    "energy": "Energy analysis (RMS, dB levels)",
    "spectral": "Spectral analysis (centroid, bandwidth, rolloff)",
    "frequency": "Frequency analysis (dominant frequencies, ZCR)",
    "correlation": "Correlation analysis (relationships between features)",
    "statistical": "Statistical distribution (aggregates, percentiles)",
    "temporal": "Temporal trends (changes over time)",
    "overview": "Overview (general summary)",
}

# ── Ranking keywords ────────────────────────────────────────────────

RANKING_WORDS = {
    "best",
    "worst",
    "top",
    "bottom",
    "highest",
    "lowest",
    "loudest",
    "quietest",
    "most",
    "least",
    "maximum",
    "minimum",
    "max",
    "min",
    "strongest",
    "weakest",
}

EXPLICIT_METRIC_WORDS = {
    "decibel",
    "db",
    "rms",
    "energy",
    "centroid",
    "spectral",
    "frequency",
    "zcr",
    "zero crossing",
    "duration",
    "count",
    "recordings",
    "complaint",
    "peak",
    "amplitude",
    "loudness",
}

TEMPORAL_VAGUE_WORDS = {"recent", "latest", "new", "current"}

TIME_REFERENCE_PATTERNS = [
    r"\b(last|past|previous)\s+\d+\s+(day|week|month|year|hour)s?\b",
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    r"\b\d{4}-\d{2}-\d{2}\b",
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b",
    r"\b(today|yesterday|this week|this month|this year|all time|ytd)\b",
]

ENTITY_HINT_WORDS = {
    "region",
    "community",
    "category",
    "area",
    "district",
    "zone",
}

# Priority order: highest first
SIGNAL_PRIORITY_ORDER = [
    "missing_entity",
    "vague_ranking_metric",
    "multi_analysis_type",
    "missing_time_range",
    "ml_missing_label",
    "ml_split_strategy",
    "ml_ambiguous_features",
    "ambiguous_temporal_word",
]


# ── Heuristic helpers ───────────────────────────────────────────────


def _match_analysis_types(query: str) -> List[str]:
    """Return which analysis_type buckets the query maps to."""
    q = query.lower()
    matched = []
    if any(w in q for w in ("energy", "rms", "db", "decibel", "loud", "quiet")):
        matched.append("energy")
    if any(
        w in q
        for w in (
            "spectral",
            "centroid",
            "bandwidth",
            "rolloff",
            "flatness",
            "mfcc",
            "chroma",
        )
    ):
        matched.append("spectral")
    if any(w in q for w in ("frequency", "dominant", "zcr", "zero crossing", "pitch")):
        matched.append("frequency")
    if any(
        w in q for w in ("correlation", "relationship", "compare", "versus", " vs ")
    ):
        matched.append("correlation")
    if any(
        w in q
        for w in (
            "distribution",
            "statistical",
            "percentile",
            "std",
            "variance",
            "spread",
        )
    ):
        matched.append("statistical")
    if any(
        w in q
        for w in (
            "trend",
            "over time",
            "temporal",
            "daily",
            "weekly",
            "monthly",
            "change",
            "growth",
        )
    ):
        matched.append("temporal")
    if any(w in q for w in ("overview", "summary", "general", "tell me about")):
        matched.append("overview")
    return matched


def _needs_time_range(query: str) -> bool:
    q = query.lower()
    return any(
        w in q
        for w in (
            "trend",
            "over time",
            "daily",
            "weekly",
            "monthly",
            "history",
            "change",
            "growth",
            "how has",
            "pattern",
        )
    )


def _has_time_reference(query: str) -> bool:
    q = query.lower()
    for pat in TIME_REFERENCE_PATTERNS:
        if re.search(pat, q, re.I):
            return True
    return False


def _has_ranking_intent(query: str) -> bool:
    q = query.lower()
    return any(w in q for w in RANKING_WORDS)


def _has_explicit_metric(query: str) -> bool:
    q = query.lower()
    return any(w in q for w in EXPLICIT_METRIC_WORDS)


ML_TASK_WORDS = {
    "classify",
    "predict",
    "train a model",
    "which features separate",
    "which features best separate",
    "balance",
    "balanced",
    "imbalanced",
    "class distribution",
    "label distribution",
    "stratify",
    "stratified",
    "split",
    "splitting",
    "feature importance",
}

LABEL_COLUMN_WORDS = {
    "category",
    "class",
    "subclass",
    "region",
    "by category",
    "by region",
    "by class",
    "by subclass",
}


def _has_ml_task_intent(query: str) -> bool:
    q = query.lower()
    return any(w in q for w in ML_TASK_WORDS)


def _has_label_column(query: str) -> bool:
    q = query.lower()
    return any(w in q for w in LABEL_COLUMN_WORDS)


def _needs_entity(query: str, context: Optional[Dict[str, Any]] = None) -> bool:
    q = query.lower()
    # Structural hint words
    if any(w in q for w in ENTITY_HINT_WORDS):
        return True
    # Entity names from DB context
    if context:
        all_entities = context.get("available_regions", []) + context.get(
            "available_communities", []
        )
        for name in all_entities:
            # Word-boundary match against known entity names so short names
            # like "Ho" don't false-positive on "how", "who", "shows", etc.
            if re.search(r"\b" + re.escape(name.lower()) + r"\b", q):
                return True
    return False


def _has_entity_match(query: str, context: Dict[str, Any]) -> bool:
    available = context.get("available_regions", []) + context.get(
        "available_communities", []
    )
    q = query.lower()
    for name in available:
        # Word-boundary matching so "Central" doesn't match "centralized"
        # and short names like "Ho" don't match "how", "who", "hours".
        if re.search(r"\b" + re.escape(name.lower()) + r"\b", q):
            return True
    return False


def _build_entity_context() -> Dict[str, Any]:
    try:
        regions = list(Region.objects.values_list("name", flat=True)[:20])
        communities = list(Community.objects.values_list("name", flat=True)[:20])
        return {
            "available_regions": regions,
            "available_communities": communities,
        }
    except Exception:
        return {"available_regions": [], "available_communities": []}


# ── Ambiguity signals ───────────────────────────────────────────────


def _options_multi_analysis(matched_types: List[str]) -> List[str]:
    if not matched_types:
        # Zero-match — show all available analysis types
        return [ANALYSIS_TYPE_LABELS[t] for t in ANALYSIS_TYPE_LABELS] + [
            "All of the above"
        ]
    return [ANALYSIS_TYPE_LABELS[t] for t in matched_types] + ["All of the above"]


AMBIGUITY_SIGNALS: Dict[str, Dict[str, Any]] = {
    "multi_analysis_type": {
        "description": "Query maps to zero or 2+ analysis_type buckets (not exactly 1)",
        "detection": lambda query, context: (
            len(_match_analysis_types(query)) != 1
            if _is_clarification_v2_enabled()
            else len(_match_analysis_types(query)) > 1
        ),
        "dimension": "analysis_type",
        "question": "What kind of analysis do you want?",
        "options_fn": lambda matched: _options_multi_analysis(matched),
    },
    "missing_time_range": {
        "description": "Query implies time-based analysis but no time reference found",
        "detection": lambda query, context: _needs_time_range(query)
        and not _has_time_reference(query),
        "dimension": "time_range",
        "question": "What time range should I look at?",
        "options": [
            "Last 7 days",
            "Last 30 days",
            "Last 3 months",
            "Last year",
            "All time",
        ],
    },
    "vague_ranking_metric": {
        "description": "User asks for ranking without specifying by what metric",
        "detection": lambda query, context: _has_ranking_intent(query)
        and not _has_explicit_metric(query),
        "dimension": "metric",
        "question": "Rank by which metric?",
        "options": [
            "Decibel level (dB)",
            "RMS energy",
            "Recording count",
            "Spectral centroid",
            "Dominant frequency",
        ],
    },
    "missing_entity": {
        "description": "Query implies filtering by region/category/community but none found",
        "detection": lambda query, context: _needs_entity(query, context)
        and not _has_entity_match(query, context),
        "dimension": "entity",
        "question": "Which area or category should I focus on?",
        "options_fn": lambda context: (
            (context.get("available_regions", [])[:4] + ["All areas"])
            if context.get("available_regions")
            else ["All areas", "Greater Accra", "Ashanti", "Northern Region"]
        ),
    },
    "ambiguous_temporal_word": {
        "description": "Query contains 'recent', 'latest', 'new' with no baseline",
        "detection": lambda query, context: bool(
            re.search(r"\b(recent|latest|new|current)\b", query, re.I)
        ),
        "dimension": "time_range",
        "question": "How far back should 'recent' go?",
        "options": ["Last 7 days", "Last 30 days", "Last 3 months"],
    },
    "ml_missing_label": {
        "description": "User asks about ML tasks (classify, predict, balance, features for) but no label column specified",
        "detection": lambda query, context: _has_ml_task_intent(query)
        and not _has_label_column(query),
        "dimension": "label",
        "question": "Which label/class should I analyze?",
        "options": ["Category", "Class", "Subclass", "Region"],
    },
    "ml_split_strategy": {
        "description": "User asks for a data split but doesn't specify stratified, random, or time-based",
        "detection": lambda query, context: (
            any(
                w in query.lower()
                for w in ("split", "splitting", "train/val", "train test", "train-test")
            )
            and not any(
                w in query.lower()
                for w in ("stratified", "random", "time-based", "time based", "by date")
            )
        ),
        "dimension": "split_strategy",
        "question": "What split strategy should I use?",
        "options": [
            "Stratified (preserves class proportions)",
            "Random (uniform shuffle)",
            "Time-based (split by recording date)",
        ],
    },
    "ml_ambiguous_features": {
        "description": "User asks about correlations or feature importance without specifying feature subset",
        "detection": lambda query, context: (
            any(
                w in query.lower()
                for w in (
                    "correlation",
                    "correlated",
                    "correlate",
                    "feature importance",
                    "which features matter",
                )
            )
            and not any(
                w in query.lower()
                for w in (
                    "rms",
                    "centroid",
                    "bandwidth",
                    "zcr",
                    "energy",
                    "db",
                    "decibel",
                    "duration",
                )
            )
        ),
        "dimension": "feature_subset",
        "question": "Which features should I focus on?",
        "options": [
            "All audio features",
            "Energy-related (RMS, dB)",
            "Spectral (centroid, bandwidth, rolloff)",
            "Temporal (duration, ZCR)",
        ],
    },
}


# ── Gate node ───────────────────────────────────────────────────────


def clarification_gate_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate the latest user message for ambiguity signals.

    If ambiguous: sets clarification_pending=True and emits clarification payload.
    If clear: passes through unchanged.
    If clarification_pending + clarification_answer set: passes through to resolver.

    IMPORTANT: only return changed state fields — never spread **state because
    the `messages` key uses the `add` reducer and would duplicate messages.
    """
    messages = state.get("messages", [])
    if not messages:
        return {"clarification_pending": False}

    if state.get("clarification_pending"):
        # Second pass with answer — let should_clarify route to resolver
        if state.get("clarification_answer"):
            return {}
        # Already waiting but no answer yet. If the latest message is NEW
        # (different from the one that triggered the original clarification),
        # cancel the pending clarification and re-evaluate. Otherwise
        # (same message, edge trigger re-entry) keep waiting.
        original_query = state.get("original_query", "")
        last_message = messages[-1]
        new_query = (
            last_message.content
            if hasattr(last_message, "content")
            else str(last_message)
        )
        if new_query and new_query != original_query:
            logger.info(
                f"New message received during pending clarification "
                f"— cancelling pending and re-evaluating. "
                f"original={original_query[:80]} new={new_query[:80]}"
            )
            return {
                "clarification_pending": False,
                "clarification_answer": None,
                "clarification_question": None,
                "clarification_options": None,
                "clarification_dimension": None,
                "original_query": None,
            }
        return {}

    last_message = messages[-1]
    query = (
        last_message.content if hasattr(last_message, "content") else str(last_message)
    )
    context = _build_entity_context()

    fired_signal = None
    matched_types = None
    for signal_name in SIGNAL_PRIORITY_ORDER:
        signal = AMBIGUITY_SIGNALS[signal_name]
        try:
            if signal["detection"](query, context):
                fired_signal = (signal_name, signal)
                if signal_name == "multi_analysis_type":
                    matched_types = _match_analysis_types(query)
                break
        except Exception as exc:
            logger.warning(f"Signal detection error for {signal_name}: {exc}")
            continue

    if fired_signal is None:
        logger.info(f"Clarification gate: no signal fired query={query[:100]}")
        return {"clarification_pending": False}

    signal_name, signal = fired_signal
    logger.info(
        f"Clarification gate: signal fired signal={signal_name} "
        f"dimension={signal['dimension']} query={query[:100]}"
    )

    if "options_fn" in signal:
        if signal_name == "multi_analysis_type":
            options = signal["options_fn"](matched_types if matched_types else [])
        else:
            options = signal["options_fn"](context)
    else:
        options = signal["options"]

    return {
        "clarification_pending": True,
        "clarification_question": signal["question"],
        "clarification_options": options,
        "clarification_dimension": signal["dimension"],
        "original_query": query,
        "confirmation_needed": False,
    }


def should_clarify(state: Dict[str, Any]) -> str:
    """LangGraph conditional edge function."""
    if state.get("clarification_pending") and not state.get("clarification_answer"):
        return "emit_clarification"
    if state.get("clarification_pending") and state.get("clarification_answer"):
        return "clarification_resolver"
    return "agent"


# ── Normalizers ─────────────────────────────────────────────────────


def _parse_natural_date_range(answer: str) -> Optional[str]:
    """Parse natural language date range into ISO string."""
    answer = answer.strip().lower()

    # Handle relative ranges
    relative_patterns = [
        (
            r"last\s+(\d+)\s+days?",
            lambda m: (
                (datetime.now() - timedelta(days=int(m.group(1)))).strftime("%Y-%m-%d"),
                datetime.now().strftime("%Y-%m-%d"),
            ),
        ),
        (
            r"last\s+(\d+)\s+weeks?",
            lambda m: (
                (datetime.now() - timedelta(weeks=int(m.group(1)))).strftime(
                    "%Y-%m-%d"
                ),
                datetime.now().strftime("%Y-%m-%d"),
            ),
        ),
        (
            r"last\s+(\d+)\s+months?",
            lambda m: (
                (datetime.now() - timedelta(days=int(m.group(1)) * 30)).strftime(
                    "%Y-%m-%d"
                ),
                datetime.now().strftime("%Y-%m-%d"),
            ),
        ),
        (
            r"last\s+year",
            lambda m: (
                (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
                datetime.now().strftime("%Y-%m-%d"),
            ),
        ),
    ]

    for pattern, fn in relative_patterns:
        m = re.search(pattern, answer)
        if m:
            start, end = fn(m)
            return f"{start} to {end}"

    # Try dateutil for explicit date ranges
    parts = re.split(r"\s+(?:to|until|through|and|–|-)\s+", answer)
    if len(parts) == 2:
        try:
            start = dateutil_parser.parse(parts[0], fuzzy=True).strftime("%Y-%m-%d")
            end = dateutil_parser.parse(parts[1], fuzzy=True).strftime("%Y-%m-%d")
            return f"{start} to {end}"
        except (ValueError, OverflowError):
            pass

    # Single date
    try:
        d = dateutil_parser.parse(answer, fuzzy=True)
        return d.strftime("%Y-%m-%d")
    except (ValueError, OverflowError):
        pass

    return None


def _fuzzy_match_metric(answer: str) -> Optional[str]:
    known = {
        "decibel": "mean_db",
        "db": "mean_db",
        "loudness": "mean_db",
        "decibel level": "mean_db",
        "rms": "rms_energy",
        "rms energy": "rms_energy",
        "energy": "rms_energy",
        "count": "recording_count",
        "recordings": "recording_count",
        "recording count": "recording_count",
        "spectral centroid": "spectral_centroid",
        "centroid": "spectral_centroid",
        "frequency": "dominant_frequency",
        "dominant frequency": "dominant_frequency",
    }

    answer_lower = answer.strip().lower()
    if answer_lower in known:
        return known[answer_lower]

    matches = difflib.get_close_matches(
        answer_lower, list(known.keys()), n=1, cutoff=0.6
    )
    if matches:
        return known[matches[0]]

    return answer


def _fuzzy_match_entity(answer: str) -> Optional[str]:
    try:
        regions = list(Region.objects.values_list("name", flat=True))
        communities = list(Community.objects.values_list("name", flat=True))
        all_names = list(regions) + list(communities)

        answer_lower = answer.strip().lower()
        for name in all_names:
            if name.lower() == answer_lower:
                return name

        matches = difflib.get_close_matches(
            answer_lower, [n.lower() for n in all_names], n=1, cutoff=0.6
        )
        if matches:
            idx = [n.lower() for n in all_names].index(matches[0])
            return all_names[idx]
    except Exception:
        pass

    return answer


def _classify_analysis_type(answer: str) -> Optional[str]:
    answer_lower = answer.strip().lower()
    for key, label in ANALYSIS_TYPE_LABELS.items():
        if key in answer_lower or any(
            w in answer_lower for w in label.lower().split()[:2]
        ):
            return key
    return answer


def _map_structured_answer(answer: str, dimension: str) -> str:
    """Map a structured pick to a canonical parameter value."""
    if dimension == "time_range":
        mapping = {
            "last 7 days": "last_7_days",
            "last 30 days": "last_30_days",
            "last 3 months": "last_3_months",
            "last year": "last_year",
            "all time": "all_time",
        }
        return mapping.get(answer.strip().lower(), answer)

    if dimension == "metric":
        mapping = {
            "decibel level (db)": "mean_db",
            "rms energy": "rms_energy",
            "recording count": "recording_count",
            "spectral centroid": "spectral_centroid",
            "dominant frequency": "dominant_frequency",
        }
        return mapping.get(
            answer.strip().lower(), _fuzzy_match_metric(answer) or answer
        )

    if dimension == "entity":
        if answer.strip().lower() == "all areas":
            return "all"
        return _fuzzy_match_entity(answer) or answer

    if dimension == "analysis_type":
        return _classify_analysis_type(answer) or answer

    if dimension == "label":
        mapping = {
            "category": "category",
            "class": "class",
            "subclass": "subclass",
            "region": "region",
        }
        return mapping.get(answer.strip().lower(), answer)

    if dimension == "split_strategy":
        mapping = {
            "stratified (preserves class proportions)": "stratified",
            "random (uniform shuffle)": "random",
            "time-based (split by recording date)": "time_based",
        }
        return mapping.get(answer.strip().lower(), answer)

    if dimension == "feature_subset":
        mapping = {
            "all audio features": "all",
            "energy-related (rms, db)": "energy",
            "spectral (centroid, bandwidth, rolloff)": "spectral",
            "temporal (duration, zcr)": "temporal",
        }
        return mapping.get(answer.strip().lower(), answer)

    return answer


def _normalize_custom_answer(answer: str, dimension: str) -> str:
    """Normalise a free-text answer for the given dimension."""
    if dimension == "time_range":
        parsed = _parse_natural_date_range(answer)
        return parsed if parsed else answer
    elif dimension == "metric":
        matched = _fuzzy_match_metric(answer)
        return matched if matched else answer
    elif dimension == "entity":
        matched = _fuzzy_match_entity(answer)
        return matched if matched else answer
    elif dimension == "analysis_type":
        classified = _classify_analysis_type(answer)
        return classified if classified else answer
    elif dimension == "label":
        answer_lower = answer.strip().lower()
        valid_labels = {"category", "class", "subclass", "region"}
        return answer_lower if answer_lower in valid_labels else answer
    elif dimension == "split_strategy":
        answer_lower = answer.strip().lower()
        strategies = {
            "stratified": "stratified",
            "random": "random",
            "time-based": "time_based",
            "time based": "time_based",
        }
        return strategies.get(answer_lower, answer)
    elif dimension == "feature_subset":
        answer_lower = answer.strip().lower()
        subsets = {
            "all": "all",
            "all audio features": "all",
            "energy": "energy",
            "energy-related": "energy",
            "spectral": "spectral",
            "temporal": "temporal",
        }
        return subsets.get(answer_lower, answer)
    else:
        return answer


# ── Built-in fallback options ───────────────────────────────────────


def _get_fallback_options(dimension: str) -> List[str]:
    return {
        "time_range": [
            "Last 7 days",
            "Last 30 days",
            "Last 3 months",
            "Last year",
            "All time",
        ],
        "metric": [
            "Decibel level (dB)",
            "RMS energy",
            "Recording count",
            "Spectral centroid",
            "Complaint rate",
        ],
        "entity": ["All areas"],
        "analysis_type": [
            "Overview",
            "Energy analysis",
            "Spectral analysis",
            "Temporal trends",
        ],
    }.get(dimension, ["Yes, proceed"])


# ── Resolver node ───────────────────────────────────────────────────


def clarification_resolver_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Receive the clarification answer and inject resolved context into messages."""
    answer = state.get("clarification_answer", "")
    is_custom = state.get("clarification_is_custom", False)
    dimension = state.get("clarification_dimension", "") or ""
    original_query = state.get("original_query", "")

    if is_custom:
        resolved = _normalize_custom_answer(answer, dimension)
        confirmation_needed = False  # Confirmation step deferred to future iteration
    else:
        resolved = _map_structured_answer(answer, dimension)
        confirmation_needed = False

    logger.info(
        f"Clarification resolved: dimension={dimension} "
        f"answer={str(answer)[:80]} resolved_value={resolved} "
        f"is_custom={is_custom}"
    )

    context_injection = HumanMessage(
        content=(
            f"[Clarification resolved] dimension={dimension} "
            f"resolved_value={resolved} original_query={original_query}"
        )
    )

    # Only return the NEW message (add reducer appends it) and state updates.
    # Never spread **state — the messages key uses the `add` reducer.
    return {
        "messages": [context_injection],
        "clarification_pending": False,
        "clarification_answer": resolved,
        "confirmation_needed": confirmation_needed,
        "confirmation_resolved_value": resolved if confirmation_needed else None,
    }


# ── Emit clarification node ─────────────────────────────────────────


def emit_clarification_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node that yields the clarification payload to the frontend.

    Stores the clarification payload in the LangGraph state so it persists
    through the PostgresSaver checkpointer and is accessible from any worker.
    views.py reads pending_clarification_payload from the checkpointer state
    after the stream completes (same pattern as pending_chart/pending_artifact).
    """
    payload = {
        "question": state.get("clarification_question"),
        "options": state.get("clarification_options"),
        "dimension": state.get("clarification_dimension"),
    }
    thread_id = str(state.get("session_id", ""))
    signal_dim = payload.get("dimension", "unknown")
    question_str = (payload.get("question") or "")[:80]

    logger.info(
        f"Clarification emitted: dimension={signal_dim} "
        f"question={question_str} thread={thread_id}"
    )
    return {"pending_clarification_payload": payload}
