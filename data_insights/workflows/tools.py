from pydantic import BaseModel, Field
from typing import Dict, List, Any, Literal, Optional
from typing import Optional, Dict, Any
from pydantic import BaseModel
from django.conf import settings
from django.db.models import Q
from django.db import models
import uuid
import logging
import os
import re
from contextvars import ContextVar
from langchain_core.tools import BaseTool
from django.utils import timezone
from datetime import timedelta, datetime
from data_insights.models import QueryCacheModel
from langchain_openai import ChatOpenAI
from django.conf import settings
from langchain_core.tools.base import ArgsSchema
from langchain_core.messages import HumanMessage
from .sql_agent import TextToSQLAgent


class ToolAmbiguityError(Exception):
    """Raised by a tool when it cannot determine the right parameters for a query.

    The agent layer catches this and routes back to the clarification gate
    instead of completing the turn. Fields:
        dimension:  which gate dimension is ambiguous (analysis_type, entity, etc.)
        query:      the user query that triggered the ambiguity
    """

    def __init__(self, dimension: str, query: str):
        self.dimension = dimension
        self.query = query
        super().__init__(f"Tool ambiguity on dimension={dimension} query={query[:100]}")


logger = logging.getLogger(__name__)

# Set by the view before each agent invocation so tools can attribute cache entries correctly.
_current_user_id: ContextVar[Optional[int]] = ContextVar(
    "current_user_id", default=None
)

# Performance optimization settings
QUERY_TIMEOUT = 30  # seconds
MAX_RESULTS_LIMIT = 1000
DEFAULT_RESULTS_LIMIT = 100

# P6-4: cap on how many raw per-row decibel values the distribution branch
# materializes into Python per group. The box-plot five-number summary is now
# computed server-side (exact, via percentile_cont), so this bounded sample is
# only a representative point cloud for any chart that overlays raw points —
# it does NOT affect the rendered box statistics. Bounds memory/transfer
# regardless of dataset size.
DISTRIBUTION_SAMPLE_LIMIT = 500

# Valid field mappings to prevent field reference errors
VALID_AUDIO_FEATURE_FIELDS = {
    "rms_energy",
    "zero_crossing_rate",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "spectral_flatness",
    "duration",
    "sample_rate",
    "num_samples",
    "mfccs",
    "chroma_stft",
    "mel_spectrogram",
    "waveform_data",
    "harmonic_ratio",
    "percussive_ratio",
}

VALID_NOISE_ANALYSIS_FIELDS = {
    "mean_db",
    "max_db",
    "min_db",
    "std_db",
    "peak_count",
    "peak_interval_mean",
    "dominant_frequency",
    "frequency_range",
    "event_count",
    "event_durations",
}

VALID_NOISE_DATASET_FIELDS = {
    "name",
    "collector",
    "description",
    "region",
    "category",
    "time_of_day",
    "community",
    "class_name",
    "subclass",
    "microphone_type",
    "audio",
    "recording_date",
    "recording_device",
    "updated_at",
    "noise_id",
    "created_at",
    "dataset_type",
}

# Related field mappings
VALID_RELATED_FIELDS = {
    "region__name": True,
    "category__name": True,
    "community__name": True,
    "microphone_type__name": True,
    "time_of_day__name": True,
    "class_name__name": True,
    "subclass__name": True,
    # PII minimization: only the collector username (a display handle) is a
    # valid reference. Direct PII (first_name/last_name/email/phone) must never
    # be surfaced to the chat/LLM, so they are intentionally excluded here.
    "collector__username": True,
}


def validate_field_reference(model_prefix: str, field_name: str) -> bool:
    """Validate that a field reference is correct for the given model"""
    if model_prefix == "audio_features":
        return field_name in VALID_AUDIO_FEATURE_FIELDS
    elif model_prefix == "noise_analysis":
        return field_name in VALID_NOISE_ANALYSIS_FIELDS
    elif model_prefix == "" or model_prefix == "noise_dataset":
        return field_name in VALID_NOISE_DATASET_FIELDS
    elif f"{model_prefix}__{field_name}" in VALID_RELATED_FIELDS:
        return True
    return False


def get_user_friendly_error(error_msg: str, context: str = "") -> str:
    """Convert technical errors to user-friendly messages"""
    error_lower = str(error_msg).lower()

    if "cannot resolve keyword" in error_lower and "into field" in error_lower:
        return "Invalid field reference in audio data query. The system is using an outdated field name. Please contact support to resolve this issue."
    elif "timeout" in error_lower:
        return "The query is taking too long to process. Please try a more specific search or contact support."
    elif "connection" in error_lower or "database" in error_lower:
        return "Unable to connect to the audio database. Please check your connection and try again."
    elif "permission" in error_lower or "access" in error_lower:
        return "You don't have permission to access this audio data. Please contact your administrator."
    elif "not found" in error_lower or "does not exist" in error_lower:
        return f"The requested audio data was not found. Please check your search criteria."
    elif "invalid" in error_lower or "malformed" in error_lower:
        return (
            "Invalid search parameters provided. Please check your input and try again."
        )
    elif "memory" in error_lower or "limit" in error_lower:
        return "The request is too large to process. Please try filtering your search to fewer results."
    else:
        return f"An unexpected error occurred while processing your audio data request. {context}"


def safe_field_reference(model_prefix: str, field_name: str) -> str:
    """Get a safe field reference, with fallback for known field mappings"""

    # Handle common field name mappings/corrections
    field_corrections = {
        "rms": "rms_energy",
        "zcr": "zero_crossing_rate",
        "db": "mean_db",
        "decibel": "mean_db",
        "frequency": "dominant_frequency",
    }

    # Apply corrections if needed
    corrected_field = field_corrections.get(field_name, field_name)

    # Build the full field reference
    if model_prefix:
        full_reference = f"{model_prefix}__{corrected_field}"
    else:
        full_reference = corrected_field

    # Validate the reference
    if validate_field_reference(model_prefix, corrected_field):
        return full_reference
    else:
        logger.warning(f"Potentially invalid field reference: {full_reference}")
        return full_reference  # Return anyway, let Django handle the error


def validate_query_fields(queryset_operations: list) -> list:
    """Validate field references in query operations before execution"""
    validated_operations = []

    for operation in queryset_operations:
        # This is a placeholder for more sophisticated validation
        # For now, just log the operations
        logger.debug(f"Query operation: {operation}")
        validated_operations.append(operation)

    return validated_operations


# ---------------------------------------------------------------------------
# Pure-math helpers (no DB / ORM) so analytics correctness can be unit tested
# without Postgres. Each is statistically self-contained and documented with
# the correct formula it implements.
# ---------------------------------------------------------------------------


def running_cumulative(values: List[float]) -> List[float]:
    """Return the running (prefix-sum) cumulative total over ordered values.

    Correct "cumulative" semantics: out[i] = sum(values[0..i]). The final
    element equals the grand total. ``None`` per-period values are treated as
    0 so the running total never resets or breaks on a gap.

    Example: [3, 1, 4] -> [3, 4, 8].
    """
    total = 0.0
    out: List[float] = []
    for v in values:
        total += float(v) if v is not None else 0.0
        out.append(total)
    return out


def complete_case_matrix(rows, feature_cols):
    """Complete-case alignment for a correlation matrix.

    Keeps only rows where EVERY selected feature is non-null, so that row i of
    feature A and row i of feature B come from the SAME observation. This is the
    statistically correct precondition for Pearson/Spearman: correlating
    independently null-dropped, then position-truncated, columns fabricates
    coefficients because the paired observations no longer correspond.

    Args:
        rows: iterable of dict-like records (one per dataset).
        feature_cols: ordered list of feature names to align on.

    Returns:
        (columns, n) where ``columns`` is a dict {feature: list_of_values} with
        every list the same length ``n`` (the number of complete-case rows) and
        the i-th entry of every column belonging to the same source row.
    """
    columns = {col: [] for col in feature_cols}
    n = 0
    for r in rows:
        vals = [r.get(col) for col in feature_cols]
        if any(v is None for v in vals):
            continue
        for col, v in zip(feature_cols, vals):
            columns[col].append(float(v))
        n += 1
    return columns, n


def stratified_split_counts(class_count, train_pct, val_pct, test_pct):
    """Allocate train/val/test counts for a single class that sum EXACTLY to
    the class size while respecting the requested ratios.

    Uses the largest-remainder (Hamilton) method: floor each ideal allocation,
    then hand the leftover units one at a time to the splits with the largest
    fractional remainders. This guarantees:
        train + val + test == class_count   (no over/under allocation)
        every split is >= 0 and <= class_count.

    The previous implementation rounded each split independently and then added
    a floor of 1, which over-allocated small classes (e.g. count=2 produced
    1/1/1 == 3 > 2).
    """
    c = int(class_count)
    if c <= 0:
        return 0, 0, 0
    ratios = [train_pct, val_pct, test_pct]
    ideal = [c * r for r in ratios]
    floors = [int(x) for x in ideal]
    remainder = c - sum(floors)
    # Distribute the remaining units to the largest fractional parts.
    frac_order = sorted(range(3), key=lambda i: (ideal[i] - floors[i]), reverse=True)
    for k in range(remainder):
        floors[frac_order[k % 3]] += 1
    return floors[0], floors[1], floors[2]


AI_CONFIG = getattr(settings, "AI_INSIGHT", {})
DB_CONFIG = AI_CONFIG.get("DATABASE", {})
AGENT_CONFIG = AI_CONFIG.get("AGENT", {})
SECURITY_CONFIG = AI_CONFIG.get("SECURITY", {})


# Use Docker DB ('db') when available, otherwise Django's default
def _get_data_insights_db_uri():
    """Use Docker DB ('db') when available, otherwise Django's default."""
    # Check if Docker DB is available (POSTGRES_HOST=db in env)
    postgres_host = os.getenv("POSTGRES_HOST")
    if postgres_host == "db":
        # We're in Docker, use the Docker DB
        return (
            f"postgresql://{os.getenv('POSTGRES_USER', 'postgres')}:"
            f"{os.getenv('POSTGRES_PASSWORD', '')}@"
            f"db:{os.getenv('POSTGRES_PORT', 5432)}/"
            f"{os.getenv('POSTGRES_DB', 'iheardatadb')}"
        )

    # Otherwise use Django's default database (works for local dev)
    db = settings.DATABASES.get("default", {})
    if "postgresql" in (db.get("ENGINE") or ""):
        return (
            f"postgresql://{db.get('USER', 'postgres')}:"
            f"{db.get('PASSWORD', '')}@"
            f"{db.get('HOST', 'localhost')}:"
            f"{db.get('PORT', 5432)}/"
            f"{db.get('NAME', 'postgres')}"
        )
    return (
        f"postgresql://{DB_CONFIG.get('USER', 'postgres')}:"
        f"{DB_CONFIG.get('PASSWORD', '')}@"
        f"{DB_CONFIG.get('HOST', 'localhost')}:"
        f"{DB_CONFIG.get('PORT', 5432)}/"
        f"{DB_CONFIG.get('NAME', 'iheardatadb')}"
    )


DB_URI = _get_data_insights_db_uri()
from data.models import (
    Dataset as AudioDataset,
    NoiseDataset,
    AudioFeature,
    NoiseAnalysis,
    VisualizationPreset,
    BulkAudioUpload,
    BulkReprocessingTask,
)
from core.models import Region, Community, Category, Class, SubClass


class PercentileCont(models.Aggregate):
    """Postgres ``percentile_cont`` ordered-set aggregate (continuous percentile).

    Renders ``percentile_cont(<fraction>) WITHIN GROUP (ORDER BY <expression>)``.
    Modeled as a Django ``Aggregate`` (rather than RawSQL) so the ORM treats it
    as an aggregate and does NOT add it to the GROUP BY clause. ``percentile_cont``
    interpolates linearly between adjacent ordered values, which is exactly the
    method used by the frontend / widget_composer ``_quartile``, so server-side
    quartiles/median match the previous "materialize every row then quartile in
    Python" result. Used by P6-4 to bound distribution-analysis memory.
    """

    function = "PERCENTILE_CONT"
    name = "PercentileCont"
    template = "%(function)s(%(percentile)s) WITHIN GROUP (ORDER BY %(expressions)s)"
    output_field = models.FloatField()

    def __init__(self, expression, percentile, **extra):
        super().__init__(expression, percentile=percentile, **extra)


# -----------------------------------------------------------------------------
# Security decision: org-wide (cross-collector) AGGREGATE analytics is INTENTIONAL
# -----------------------------------------------------------------------------
# The data-insights tools below deliberately analyze data across ALL collectors
# and contributors (counts, averages, breakdowns by region/category/device, and
# the username-only top_collectors leaderboard). This is a product requirement:
# researchers need org-wide views of the data bank. Rows are NOT scoped per-user
# and must not be — do not "re-fix" this as per-user row scoping.
#
# The two safeguards that make this safe are:
#   1. Read-only SQL execution (Phase 2): the SQL engine only runs SELECTs
#      against an allow-listed set of data/core tables; no writes, no PII tables.
#   2. PII + raw-storage-path minimization in tool output (Phase 3): tools return
#      collector id / username only — never email, phone, full legal name,
#      physical address, or precise GPS — and never the raw `audio` storage
#      path / object-storage key / signed URL (only a `has_audio` presence
#      signal where presence matters).
# -----------------------------------------------------------------------------


class NoiseDatasetSearchInput(BaseModel):
    """input for noise dataset search"""

    filter_criteria: Optional[
        Dict[
            Literal[
                "name",
                "collector",
                "category",
                "region",
                "recording_date",
                "time_of_day",
                "community",
                "class_name",
                "microphone_type",
                "subclass",
            ],
            Any,
        ]
    ] = Field(default_factory=dict, description="Filter criteria for noise dataset")


class AudioFeatureSearchInput(BaseModel):
    """Input schema for searching/filtering audio features"""

    # Time-domain filters
    min_rms_energy: Optional[float] = Field(
        default=None, description="Minimum RMS energy"
    )
    max_rms_energy: Optional[float] = Field(
        default=None, description="Maximum RMS energy"
    )
    min_zero_crossing_rate: Optional[float] = Field(
        default=None, description="Minimum Zero Crossing Rate"
    )
    max_zero_crossing_rate: Optional[float] = Field(
        default=None, description="Maximum Zero Crossing Rate"
    )

    # Frequency-domain filters
    min_spectral_centroid: Optional[float] = Field(
        default=None, description="Minimum Spectral Centroid"
    )
    max_spectral_centroid: Optional[float] = Field(
        default=None, description="Maximum Spectral Centroid"
    )
    min_spectral_bandwidth: Optional[float] = Field(
        default=None, description="Minimum Spectral Bandwidth"
    )
    max_spectral_bandwidth: Optional[float] = Field(
        default=None, description="Maximum Spectral Bandwidth"
    )
    min_spectral_rolloff: Optional[float] = Field(
        default=None, description="Minimum Spectral Rolloff"
    )
    max_spectral_rolloff: Optional[float] = Field(
        default=None, description="Maximum Spectral Rolloff"
    )
    min_spectral_flatness: Optional[float] = Field(
        default=None, description="Minimum Spectral Flatness"
    )
    max_spectral_flatness: Optional[float] = Field(
        default=None, description="Maximum Spectral Flatness"
    )

    # Harmonic/percussive filters
    min_harmonic_ratio: Optional[float] = Field(
        default=None, description="Minimum Harmonic Ratio"
    )
    max_harmonic_ratio: Optional[float] = Field(
        default=None, description="Maximum Harmonic Ratio"
    )
    min_percussive_ratio: Optional[float] = Field(
        default=None, description="Minimum Percussive Ratio"
    )
    max_percussive_ratio: Optional[float] = Field(
        default=None, description="Maximum Percussive Ratio"
    )

    # Pagination
    limit: int = Field(default=50, description="Maximum number of results")
    offset: int = Field(default=0, description="Offset for pagination")


class NoiseDetailInput(BaseModel):
    """Input schema for getting detailed noise dataset information"""

    noise_id: str = Field(description="Unique Noise Dataset ID to retrieve details for")

    include_audio_features: bool = Field(
        default=True, description="Include extracted audio features"
    )
    include_analysis: bool = Field(
        default=True, description="Include noise analysis results"
    )
    include_visualizations: bool = Field(
        default=False, description="Include visualization presets"
    )
    include_collector: bool = Field(
        default=True, description="Include collector (user) details"
    )
    include_metadata: bool = Field(
        default=True,
        description="Include region, category, class, and related metadata",
    )


# Tool implementation for searching Noise Datasets
class NoiseDatasetSearchTool(BaseTool):
    name: str = "search_noise_datasets"
    description: str = """Search for noise datasets based on criteria.
    Returns a query handle for large result sets.
    Use this to find datasets by name, region, community, recording_date, category, etc."""

    def _run(
        self,
        filter_criteria: Optional[Dict[str, Any]] = None,
        limit: int = 50,
        offset: int = 0,
        include_features: bool = False,
    ) -> Dict[str, Any]:
        try:
            filter_criteria = filter_criteria or {}

            # Build queryset
            queryset = NoiseDataset.objects.select_related(
                "region",
                "community",
                "category",
                "time_of_day",
                "class_name",
                "subclass",
                "microphone_type",
                "dataset_type",
                "collector",
            )

            # Apply filters
            if "name" in filter_criteria:
                queryset = queryset.filter(name__icontains=filter_criteria["name"])

            if "region" in filter_criteria:
                # Search by region
                queryset = queryset.filter(
                    models.Q(region__name__icontains=filter_criteria["region"])
                )

            if "community" in filter_criteria:
                # Search by community
                queryset = queryset.filter(
                    models.Q(community__name__icontains=filter_criteria["community"])
                )

            # Note: noise_level filtering would need to be done through NoiseAnalysis model
            # For now, we'll skip this filtering until we implement proper joins

            if "date_from" in filter_criteria:
                queryset = queryset.filter(
                    recording_date__gte=filter_criteria["date_from"]
                )

            if "date_to" in filter_criteria:
                queryset = queryset.filter(
                    recording_date__lte=filter_criteria["date_to"]
                )

            # Get total count
            total_count = queryset.count()

            # If large result set, create query handle
            if total_count > 100:
                query_id = f"noise_{uuid.uuid4().hex[:12]}"

                cache_entry = QueryCacheModel.objects.create(
                    query_id=query_id,
                    query_type="noise_dataset_search",
                    query_sql=str(queryset.query),
                    result_count=total_count,
                    created_by_id=_current_user_id.get(),
                    metadata={
                        "filter_criteria": filter_criteria,
                        "include_features": include_features,
                    },
                )

                # Return sample preview
                sample_data = list(
                    queryset[:5].values(
                        "id",
                        "name",
                        "region__name",
                        "community__name",
                        "recording_date",
                        "category__name",
                    )
                )

                return {
                    "query_id": query_id,
                    "total_count": total_count,
                    "sample_data": sample_data,
                    "rows": sample_data,
                    "columns": list(sample_data[0].keys()) if sample_data else [],
                    "message": f'Found {total_count} noise datasets. Use query_id "{query_id}" for bulk operations.',
                    "analysis_type": "dataset_search_sample",
                    "filter_criteria": filter_criteria,
                    "limit": limit,
                    "offset": offset,
                    "has_more": total_count > (offset + limit),
                    "pagination": {
                        "limit": limit,
                        "offset": offset,
                        "has_more": total_count > (offset + limit),
                        "total_count": total_count,
                        "query_kind": "dataset_search",
                        "filter_criteria": filter_criteria,
                    },
                    "skip_visualization": True,
                }

            else:
                # Return paginated data directly
                datasets = queryset[offset : offset + limit]

                result_data = []
                for dataset in datasets:
                    dataset_data = {
                        "id": dataset.id,
                        "name": dataset.name,
                        "region": dataset.region.name if dataset.region else None,
                        "community": (
                            dataset.community.name if dataset.community else None
                        ),
                        "category": dataset.category.name if dataset.category else None,
                        "recording_date": dataset.recording_date,
                        "recording_device": dataset.recording_device,
                    }

                    # Add noise analysis data if available
                    if hasattr(dataset, "noise_analysis") and dataset.noise_analysis:
                        analysis = dataset.noise_analysis
                        dataset_data.update(
                            {
                                "mean_db": analysis.mean_db,
                                "max_db": analysis.max_db,
                                "min_db": analysis.min_db,
                                "dominant_frequency": analysis.dominant_frequency,
                                "event_count": analysis.event_count,
                            }
                        )

                    if include_features and hasattr(dataset, "audio_features"):
                        features = dataset.audio_features
                        dataset_data["features"] = {
                            "rms_energy": getattr(features, "rms_energy", None),
                            "spectral_centroid": getattr(
                                features, "spectral_centroid", None
                            ),
                            "zero_crossing_rate": getattr(
                                features, "zero_crossing_rate", None
                            ),
                            "duration": getattr(features, "duration", None),
                        }

                    result_data.append(dataset_data)

                return {
                    "datasets": result_data,
                    "rows": result_data,
                    "columns": list(result_data[0].keys()) if result_data else [],
                    "total_count": total_count,
                    "returned_count": len(result_data),
                    "analysis_type": "dataset_search_results",
                    "filter_criteria": filter_criteria,
                    "limit": limit,
                    "offset": offset,
                    "has_more": total_count > (offset + limit),
                    "pagination": {
                        "limit": limit,
                        "offset": offset,
                        "has_more": total_count > (offset + limit),
                        "total_count": total_count,
                        "query_kind": "dataset_search",
                        "filter_criteria": filter_criteria,
                    },
                    "skip_visualization": True,
                }

        except Exception as e:
            logger.error(f"Error in noise dataset search: {str(e)}")
            return {"error": f"Noise dataset search failed: {str(e)}"}


class AudioFeatureSearchTool(BaseTool):
    name: str = "search_audio_features"
    description: str = """Search audio features based on criteria such as RMS, ZCR, Spectral Centroid, and Bandwidth.
    Returns either a direct result set or a query handle for large sets."""

    def _run(
        self,
        filter_criteria: Optional[Dict[str, Any]] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        try:
            filter_criteria = filter_criteria or {}

            # Build queryset
            queryset = AudioFeature.objects.select_related(
                "noise_dataset",
                "noise_dataset__region",
                "noise_dataset__category",
                "noise_dataset__community",
            )

            # Apply filters with correct field names
            if "rms_energy" in filter_criteria:
                queryset = queryset.filter(
                    rms_energy__gte=filter_criteria["rms_energy"]
                )

            if "zero_crossing_rate" in filter_criteria:
                queryset = queryset.filter(
                    zero_crossing_rate__gte=filter_criteria["zero_crossing_rate"]
                )

            if "spectral_centroid" in filter_criteria:
                queryset = queryset.filter(
                    spectral_centroid__gte=filter_criteria["spectral_centroid"]
                )

            if "spectral_bandwidth" in filter_criteria:
                queryset = queryset.filter(
                    spectral_bandwidth__gte=filter_criteria["spectral_bandwidth"]
                )

            if "duration" in filter_criteria:
                queryset = queryset.filter(duration__gte=filter_criteria["duration"])

            # Add filters for related dataset fields
            if "region" in filter_criteria:
                queryset = queryset.filter(
                    noise_dataset__region__name__icontains=filter_criteria["region"]
                )

            if "category" in filter_criteria:
                queryset = queryset.filter(
                    noise_dataset__category__name__icontains=filter_criteria["category"]
                )

            # Count results
            total_count = queryset.count()

            if total_count > 100:  # return query handle for big results
                query_id = f"audio_features_{uuid.uuid4().hex[:12]}"

                QueryCacheModel.objects.create(
                    query_id=query_id,
                    query_type="audio_feature_search",
                    query_sql=str(queryset.query),
                    result_count=total_count,
                    created_by_id=_current_user_id.get(),
                    metadata={"filter_criteria": filter_criteria},
                )

                sample_features = list(
                    queryset[:5].values(
                        "id",
                        "rms_energy",
                        "zero_crossing_rate",
                        "spectral_centroid",
                        "spectral_bandwidth",
                        "duration",
                    )
                )

                return {
                    "query_id": query_id,
                    "total_count": total_count,
                    "sample_data": sample_features,
                    "message": f'Found {total_count} audio features. Use query_id "{query_id}" for bulk operations.',
                }

            else:  # small result set → return directly
                features = queryset[offset : offset + limit]

                result_data = []
                for feature in features:
                    result_data.append(
                        {
                            "id": feature.id,
                            "rms_energy": feature.rms_energy,
                            "zero_crossing_rate": feature.zero_crossing_rate,
                            "spectral_centroid": feature.spectral_centroid,
                            "spectral_bandwidth": feature.spectral_bandwidth,
                            "duration": feature.duration,
                            "dataset_name": (
                                feature.noise_dataset.name
                                if feature.noise_dataset
                                else None
                            ),
                            "region": (
                                feature.noise_dataset.region.name
                                if feature.noise_dataset
                                and feature.noise_dataset.region
                                else None
                            ),
                        }
                    )

                return {
                    "audio_features": result_data,
                    "total_count": total_count,
                    "returned_count": len(result_data),
                }

        except Exception as e:
            logger.error(f"Error in audio feature search: {str(e)}")
            user_friendly_error = get_user_friendly_error(
                str(e), "searching audio features"
            )
            return {"error": user_friendly_error, "technical_details": str(e)}


class AudioAnalysisInput(BaseModel):
    query: str = Field(description="The natural language question about audio data")
    analysis_type: Optional[str] = Field(
        default=None,
        description=(
            "Optional. Leave null/omit — the tool auto-detects the right analysis from the query. "
            "Only set this if you are certain: 'energy', 'spectral', 'frequency', 'correlation', "
            "'statistical', 'temporal', or 'overview'."
        ),
    )
    force_analysis_type: Optional[str] = Field(
        default=None,
        description=(
            "When set, skip auto-detection entirely and use this as the analysis_type. "
            "Set this ONLY when the system's clarification gate has resolved the analysis "
            "type with the user (clarification_dimension='analysis_type'). "
            "Valid values: 'energy', 'spectral', 'frequency', 'correlation', "
            "'statistical', 'temporal', 'overview'."
        ),
    )
    group_by: Optional[str] = Field(
        default=None,
        description=(
            "Dimension to break results down by. Pick the one mentioned in the query: "
            "'region', 'category', 'community', 'microphone_type', 'time_of_day'. "
            "Omit (null) if no grouping is requested."
        ),
    )
    force_entity: Optional[str] = Field(
        default=None,
        description=(
            "When set by the clarification gate (dimension='entity'), use this "
            "entity name as a hard constraint for group_by. Set to 'all' if the "
            "user chose 'All areas'. Overrides group_by."
        ),
    )
    force_time_range: Optional[str] = Field(
        default=None,
        description=(
            "When set by the clarification gate (dimension='time_range'), use "
            "this as a hard constraint for temporal analysis. Values: "
            "'last_7_days', 'last_30_days', 'last_3_months', 'last_year', "
            "'all_time'."
        ),
    )


class AudioAnalysisTool(BaseTool):
    name: str = "analyze_audio_data"
    description: str = """Comprehensive audio analysis tool for energy, spectral, frequency, and statistical analysis.
    Use this for questions about:
    - Energy levels: RMS energy, decibel analysis, cumulative energy
    - Spectral features: centroid, bandwidth, rolloff, flatness trends
    - Frequency analysis: dominant frequencies, zero crossing rates
    - Statistical analysis: distributions, correlations, comparisons
    - Grouped analysis: by region, category, microphone type, community
    - Temporal analysis: trends over time, cumulative metrics"""
    args_schema: type[BaseModel] = AudioAnalysisInput

    def _run(
        self,
        query: str,
        analysis_type: str = "overview",
        group_by: Optional[str] = None,
        force_analysis_type: Optional[str] = None,
        force_entity: Optional[str] = None,
        force_time_range: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        try:
            from django.db import models
            from django.db.models import Avg, Max, Min, Count, StdDev, Sum
            from django.db.models.functions import TruncMonth, TruncDate

            # Get base queryset with all related data
            queryset = NoiseDataset.objects.select_related(
                "audio_features",
                "noise_analysis",
                "region",
                "category",
                "community",
                "microphone_type",
                "time_of_day",
            ).filter(audio_features__isnull=False, noise_analysis__isnull=False)

            # force_analysis_type takes absolute precedence — set by the
            # clarification gate when the user explicitly picked an analysis type.
            # This prevents _auto_detect_analysis() from silently overriding the
            # user's answer.
            if force_analysis_type:
                analysis_type = force_analysis_type
            elif not analysis_type:
                analysis_type = self._auto_detect_analysis(query)
                if analysis_type is None:
                    v2_enabled = settings.AI_INSIGHT.get("AGENT", {}).get(
                        "CLARIFICATION_V2_ENABLED", False
                    )
                    if v2_enabled:
                        return {
                            "_tool_ambiguity": True,
                            "dimension": "analysis_type",
                            "query": query,
                            "message": (
                                "The analysis type could not be determined from the "
                                "query. Ask the user what kind of analysis they want."
                            ),
                        }
                    return {
                        "analysis_type": "unresolved",
                        "message": (
                            "The analysis type could not be determined from the query. "
                            "Ask the user what kind of analysis they want: energy, "
                            "spectral, frequency, correlation, statistical, temporal, "
                            "or overview."
                        ),
                        "skip_visualization": True,
                    }

            # force_entity / force_time_range (V2 only) take precedence over
            # LLM-supplied parameters when the gate resolved the dimension.
            v2_enabled = settings.AI_INSIGHT.get("AGENT", {}).get(
                "CLARIFICATION_V2_ENABLED", False
            )
            if v2_enabled:
                if force_entity and force_entity != "all":
                    group_by = "region"
                elif force_entity == "all":
                    group_by = None

                if force_time_range:
                    from datetime import timedelta

                    now = timezone.now()
                    time_deltas = {
                        "last_7_days": timedelta(days=7),
                        "last_30_days": timedelta(days=30),
                        "last_3_months": timedelta(days=90),
                        "last_year": timedelta(days=365),
                    }
                    delta = time_deltas.get(force_time_range)
                    if delta:
                        cutoff = now - delta
                        queryset = queryset.filter(recording_date__gte=cutoff)

            # Perform analysis based on type
            if analysis_type == "energy":
                return self._energy_analysis(queryset, group_by, query)
            elif analysis_type == "spectral":
                return self._spectral_analysis(queryset, group_by, query)
            elif analysis_type == "frequency":
                return self._frequency_analysis(queryset, group_by, query)
            elif analysis_type == "correlation":
                return self._correlation_analysis(queryset, query)
            elif analysis_type == "statistical":
                return self._statistical_analysis(queryset, group_by, query)
            elif analysis_type == "temporal":
                return self._temporal_analysis(queryset, query)
            else:
                return self._overview_analysis(queryset, query)

        except Exception as e:
            logger.error(f"Error in audio analysis: {e}")
            user_friendly_error = get_user_friendly_error(
                str(e), "analyzing audio data"
            )
            return {"error": user_friendly_error, "technical_details": str(e)}

    def _auto_detect_analysis(self, query: str) -> Optional[str]:
        """Detect the appropriate analysis type from the query text.

        Returns the single best-matching type, or None if no keywords match
        (zero confidence — caller should ask the user for clarification rather
        than silently defaulting).
        """
        q = query.lower()

        # Score each type by keyword matches
        scores = {
            "temporal": sum(
                1
                for w in [
                    "over time",
                    "trend",
                    "monthly",
                    "daily",
                    "timeline",
                    "change",
                    "progression",
                    "history",
                    "over the past",
                    "this month",
                    "this year",
                    "last month",
                    "last year",
                ]
                if w in q
            ),
            "correlation": sum(
                1
                for w in [
                    "correlation",
                    "relationship between",
                    " vs ",
                    "against",
                    "compared to",
                    "plotted against",
                ]
                if w in q
            ),
            "statistical": sum(
                1
                for w in [
                    "distribution",
                    "quartile",
                    "outlier",
                    "spread",
                    "range",
                    "statistical",
                    "median",
                    "standard deviation",
                    "variance",
                ]
                if w in q
            ),
            "energy": sum(
                1
                for w in [
                    "rms",
                    "energy",
                    "decibel",
                    "db",
                    "amplitude",
                    "loudness",
                    "loudest",
                    "quietest",
                    "cumulative energy",
                ]
                if w in q
            ),
            "spectral": sum(
                1
                for w in [
                    "spectral",
                    "centroid",
                    "bandwidth",
                    "rolloff",
                    "flatness",
                    "mfcc",
                    "chroma",
                    "mel",
                ]
                if w in q
            ),
            "frequency": sum(
                1
                for w in [
                    "frequency",
                    "zero crossing",
                    "zcr",
                    "hz",
                    "pitch",
                    "dominant",
                ]
                if w in q
            ),
        }

        best = max(scores, key=scores.get)
        # Under V2: require score >= 2 (at least two keyword hits or one
        # multi-word phrase). A single keyword is too ambiguous to route
        # on confidently. Under V1: keep old behavior (score > 0 = "overview").
        v2_enabled = settings.AI_INSIGHT.get("AGENT", {}).get(
            "CLARIFICATION_V2_ENABLED", False
        )
        if v2_enabled:
            return best if scores[best] >= 2 else None
        return best if scores[best] > 0 else "overview"

    def _energy_analysis(self, queryset, group_by, query):
        """Analyze energy-related metrics"""
        try:
            if group_by == "region":
                results = list(
                    queryset.values("region__name")
                    .annotate(
                        dataset_count=Count("id"),
                        avg_rms_energy=Avg("audio_features__rms_energy"),
                        max_rms_energy=Max("audio_features__rms_energy"),
                        min_rms_energy=Min("audio_features__rms_energy"),
                        avg_decibel=Avg("noise_analysis__mean_db"),
                        max_decibel=Max("noise_analysis__max_db"),
                        min_decibel=Min("noise_analysis__min_db"),
                        cumulative_energy=Sum("audio_features__rms_energy"),
                    )
                    .order_by("-avg_rms_energy")
                )

            elif group_by == "category":
                results = list(
                    queryset.values("category__name")
                    .annotate(
                        dataset_count=Count("id"),
                        avg_rms_energy=Avg("audio_features__rms_energy"),
                        max_rms_energy=Max("audio_features__rms_energy"),
                        avg_decibel=Avg("noise_analysis__mean_db"),
                        max_decibel=Max("noise_analysis__max_db"),
                        cumulative_energy=Sum("audio_features__rms_energy"),
                    )
                    .order_by("-avg_decibel")
                )

            elif group_by == "microphone_type":
                results = list(
                    queryset.values("microphone_type__name")
                    .annotate(
                        dataset_count=Count("id"),
                        avg_rms_energy=Avg("audio_features__rms_energy"),
                        avg_decibel=Avg("noise_analysis__mean_db"),
                        max_decibel=Max("noise_analysis__max_db"),
                    )
                    .order_by("-avg_decibel")
                )

            else:
                # Overall energy statistics
                stats = queryset.aggregate(
                    total_datasets=Count("id"),
                    avg_rms_energy=Avg("audio_features__rms_energy"),
                    max_rms_energy=Max("audio_features__rms_energy"),
                    min_rms_energy=Min("audio_features__rms_energy"),
                    std_rms_energy=StdDev("audio_features__rms_energy"),
                    avg_decibel=Avg("noise_analysis__mean_db"),
                    max_decibel=Max("noise_analysis__max_db"),
                    min_decibel=Min("noise_analysis__min_db"),
                    cumulative_energy=Sum("audio_features__rms_energy"),
                    total_duration=Sum("audio_features__duration"),
                )
                results = [stats]

            return {
                "analysis_type": "energy_analysis",
                "grouped_by": group_by,
                "results": results,
                "rows": results if group_by else [],
                "columns": list(results[0].keys()) if group_by and results else [],
                "chart_hint": (
                    {"x": f"{group_by}__name", "y": "avg_decibel"} if group_by else None
                ),
                "skip_visualization": not bool(group_by),
                "query": query,
                "summary": f"Energy analysis shows audio power levels and decibel measurements",
            }

        except Exception as e:
            return {"error": f"Energy analysis failed: {str(e)}"}

    def _spectral_analysis(self, queryset, group_by, query):
        """Analyze spectral characteristics"""
        try:
            if group_by == "region":
                results = list(
                    queryset.values("region__name")
                    .annotate(
                        dataset_count=Count("id"),
                        avg_spectral_centroid=Avg("audio_features__spectral_centroid"),
                        avg_spectral_bandwidth=Avg(
                            "audio_features__spectral_bandwidth"
                        ),
                        avg_spectral_rolloff=Avg("audio_features__spectral_rolloff"),
                        avg_spectral_flatness=Avg("audio_features__spectral_flatness"),
                    )
                    .order_by("-avg_spectral_centroid")
                )

            elif group_by == "category":
                results = list(
                    queryset.values("category__name")
                    .annotate(
                        dataset_count=Count("id"),
                        avg_spectral_centroid=Avg("audio_features__spectral_centroid"),
                        avg_spectral_bandwidth=Avg(
                            "audio_features__spectral_bandwidth"
                        ),
                        avg_spectral_rolloff=Avg("audio_features__spectral_rolloff"),
                    )
                    .order_by("-avg_spectral_centroid")
                )

            else:
                stats = queryset.aggregate(
                    total_datasets=Count("id"),
                    avg_spectral_centroid=Avg("audio_features__spectral_centroid"),
                    max_spectral_centroid=Max("audio_features__spectral_centroid"),
                    min_spectral_centroid=Min("audio_features__spectral_centroid"),
                    avg_spectral_bandwidth=Avg("audio_features__spectral_bandwidth"),
                    avg_spectral_rolloff=Avg("audio_features__spectral_rolloff"),
                    avg_spectral_flatness=Avg("audio_features__spectral_flatness"),
                )
                results = [stats]

            return {
                "analysis_type": "spectral_analysis",
                "grouped_by": group_by,
                "results": results,
                "rows": results if group_by else [],
                "columns": list(results[0].keys()) if group_by and results else [],
                "chart_hint": (
                    {"x": f"{group_by}__name", "y": "avg_spectral_centroid"}
                    if group_by
                    else None
                ),
                "skip_visualization": not bool(group_by),
                "query": query,
                "summary": f"Spectral analysis shows frequency distribution characteristics",
            }

        except Exception as e:
            return {"error": f"Spectral analysis failed: {str(e)}"}

    def _frequency_analysis(self, queryset, group_by, query):
        """Analyze frequency characteristics"""
        try:
            if group_by == "region":
                results = list(
                    queryset.values("region__name")
                    .annotate(
                        dataset_count=Count("id"),
                        avg_dominant_frequency=Avg(
                            "noise_analysis__dominant_frequency"
                        ),
                        max_dominant_frequency=Max(
                            "noise_analysis__dominant_frequency"
                        ),
                        avg_zero_crossing_rate=Avg(
                            "audio_features__zero_crossing_rate"
                        ),
                    )
                    .order_by("-avg_dominant_frequency")
                )

            elif group_by == "category":
                results = list(
                    queryset.values("category__name")
                    .annotate(
                        dataset_count=Count("id"),
                        avg_dominant_frequency=Avg(
                            "noise_analysis__dominant_frequency"
                        ),
                        avg_zero_crossing_rate=Avg(
                            "audio_features__zero_crossing_rate"
                        ),
                    )
                    .order_by("-avg_dominant_frequency")
                )

            else:
                stats = queryset.aggregate(
                    total_datasets=Count("id"),
                    avg_dominant_frequency=Avg("noise_analysis__dominant_frequency"),
                    max_dominant_frequency=Max("noise_analysis__dominant_frequency"),
                    min_dominant_frequency=Min("noise_analysis__dominant_frequency"),
                    avg_zero_crossing_rate=Avg("audio_features__zero_crossing_rate"),
                )
                results = [stats]

            return {
                "analysis_type": "frequency_analysis",
                "grouped_by": group_by,
                "results": results,
                "rows": results if group_by else [],
                "columns": list(results[0].keys()) if group_by and results else [],
                "chart_hint": (
                    {"x": f"{group_by}__name", "y": "avg_dominant_frequency"}
                    if group_by
                    else None
                ),
                "skip_visualization": not bool(group_by),
                "query": query,
                "summary": f"Frequency analysis shows dominant frequencies and zero crossing rates",
            }

        except Exception as e:
            return {"error": f"Frequency analysis failed: {str(e)}"}

    def _correlation_analysis(self, queryset, query):
        """Analyze correlations between different features"""
        try:
            # Get sample data for correlation
            data = list(
                queryset.values(
                    "audio_features__rms_energy",
                    "audio_features__spectral_centroid",
                    "audio_features__spectral_bandwidth",
                    "audio_features__zero_crossing_rate",
                    "noise_analysis__mean_db",
                    "noise_analysis__dominant_frequency",
                    "audio_features__duration",
                )[:100]
            )  # Limit for performance

            # Extract specific correlations based on query
            correlations = {}
            if "rms" in query.lower() and "spectral" in query.lower():
                rms_data = [
                    d["audio_features__rms_energy"]
                    for d in data
                    if d["audio_features__rms_energy"] is not None
                ]
                centroid_data = [
                    d["audio_features__spectral_centroid"]
                    for d in data
                    if d["audio_features__spectral_centroid"] is not None
                ]

                correlations["rms_vs_spectral_centroid"] = {
                    "rms_energy_samples": rms_data[:20],
                    "spectral_centroid_samples": centroid_data[:20],
                    "sample_count": min(len(rms_data), len(centroid_data)),
                }

            if "frequency" in query.lower() and "amplitude" in query.lower():
                freq_data = [
                    d["noise_analysis__dominant_frequency"]
                    for d in data
                    if d["noise_analysis__dominant_frequency"] is not None
                ]
                db_data = [
                    d["noise_analysis__mean_db"]
                    for d in data
                    if d["noise_analysis__mean_db"] is not None
                ]

                correlations["frequency_vs_amplitude"] = {
                    "frequency_samples": freq_data[:20],
                    "decibel_samples": db_data[:20],
                    "sample_count": min(len(freq_data), len(db_data)),
                }

            return {
                "analysis_type": "correlation_analysis",
                "correlations": correlations,
                "total_samples": len(data),
                "query": query,
                "summary": f"Correlation analysis between audio features",
            }

        except Exception as e:
            return {"error": f"Correlation analysis failed: {str(e)}"}

    def _decibel_distribution_by_group(self, queryset, group_field):
        """Compute the per-group decibel distribution entirely in the database.

        Returns ``{group_name: {decibel_values, count, avg, max, min, box_stats}}``
        — the exact dict shape the charts (widget_composer ``_decompose_statistical``)
        and the table builder (views) consume.

        P6-4: the previous implementation pulled EVERY ``mean_db`` row for the
        filtered dataset into Python and grouped/aggregated it there, so memory
        scaled with the row count. Here, ``count``/``avg``/``max``/``min`` come
        from ``GROUP BY`` aggregates and ``q1``/``median``/``q3`` come from
        ``percentile_cont`` (continuous percentile = linear interpolation), which
        matches the frontend / ``_box_stats`` quartile method exactly — so the
        rendered box plot is numerically unchanged. ``decibel_values`` is kept
        for backward compatibility but is now a BOUNDED representative sample
        (capped at ``DISTRIBUTION_SAMPLE_LIMIT`` per group); the rendered box
        statistics come from the exact precomputed ``box_stats``, not from this
        sample, so capping it changes no statistical result.
        """
        from django.db.models import Avg, Count, Max, Min

        # percentile_cont(<q>) WITHIN GROUP (ORDER BY mean_db) is an ordered-set
        # aggregate. It MUST be modeled as a Django Aggregate (not RawSQL) so the
        # ORM recognizes it as an aggregate and keeps it OUT of the GROUP BY
        # clause — RawSQL is opaque to Django and gets grouped on, producing
        # invalid SQL. PercentileCont (defined at module scope) renders the
        # WITHIN GROUP ordered-set form correctly for Postgres.
        def _pct(q):
            return PercentileCont("noise_analysis__mean_db", q)

        agg_qs = (
            queryset.filter(
                **{
                    f"{group_field}__isnull": False,
                    "noise_analysis__mean_db__isnull": False,
                }
            )
            .values(group_field)
            .annotate(
                count=Count("id"),
                avg=Avg("noise_analysis__mean_db"),
                max=Max("noise_analysis__mean_db"),
                min=Min("noise_analysis__mean_db"),
                q1=_pct(0.25),
                median=_pct(0.5),
                q3=_pct(0.75),
            )
        )

        distribution_data = {}
        for grp in agg_qs:
            name = grp[group_field]
            count = grp["count"]
            min_v = float(grp["min"]) if grp["min"] is not None else 0.0
            max_v = float(grp["max"]) if grp["max"] is not None else 0.0
            box_stats = {
                "min": min_v,
                "q1": float(grp["q1"]) if grp["q1"] is not None else min_v,
                "median": (
                    float(grp["median"]) if grp["median"] is not None else min_v
                ),
                "q3": float(grp["q3"]) if grp["q3"] is not None else max_v,
                "max": max_v,
                "count": count,
            }
            # Bounded representative sample of raw values (not used for the box
            # statistics, which are exact above). Capped so memory/transfer do
            # not scale with the group's true row count.
            sample = list(
                queryset.filter(
                    **{
                        group_field: name,
                        "noise_analysis__mean_db__isnull": False,
                    }
                ).values_list("noise_analysis__mean_db", flat=True)[
                    :DISTRIBUTION_SAMPLE_LIMIT
                ]
            )
            decibel_values = [float(v) for v in sample if v is not None]
            distribution_data[name] = {
                "decibel_values": decibel_values,
                "count": count,
                "avg": float(grp["avg"]) if grp["avg"] is not None else 0.0,
                "max": max_v,
                "min": min_v,
                # Exact server-computed five-number summary; consumed directly by
                # the box-plot renderer so the bounded sample above never affects
                # the drawn box.
                "box_stats": box_stats,
            }
        return distribution_data

    def _statistical_analysis(self, queryset, group_by, query):
        """Statistical distribution analysis with actual data for box plots"""
        try:
            query_lower = query.lower()

            # For distribution queries, provide box-plot statistics for charts.
            # P6-4: compute the per-group distribution SERVER-SIDE (count/avg/
            # max/min plus q1/median/q3 via Postgres percentile_cont) instead of
            # pulling every mean_db row into Python and grouping it there. Memory
            # and transfer are now bounded by the number of GROUPS, not the row
            # count. The precomputed box statistics use percentile_cont with the
            # SAME linear-interpolation method as the frontend / widget_composer
            # _quartile, so the rendered box plot is numerically identical to the
            # previous "materialize every row" approach.
            if "distribution" in query_lower and group_by:
                if group_by == "category":
                    distribution_data = self._decibel_distribution_by_group(
                        queryset, "category__name"
                    )
                    return {
                        "analysis_type": "statistical_distribution",
                        "grouped_by": group_by,
                        "distribution_data": distribution_data,
                        "categories": list(distribution_data.keys()),
                        "query": query,
                        "summary": f"Decibel level distribution across {len(distribution_data)} categories with actual values for box plot visualization",
                    }

                elif group_by == "region":
                    distribution_data = self._decibel_distribution_by_group(
                        queryset, "region__name"
                    )
                    return {
                        "analysis_type": "statistical_distribution",
                        "grouped_by": group_by,
                        "distribution_data": distribution_data,
                        "regions": list(distribution_data.keys()),
                        "query": query,
                        "summary": f"Decibel level distribution across {len(distribution_data)} regions with actual values for box plot visualization",
                    }

            # Fallback to summary statistics if not a distribution query
            if group_by == "category":
                results = list(
                    queryset.values("category__name")
                    .annotate(
                        dataset_count=Count("id"),
                        avg_decibel=Avg("noise_analysis__mean_db"),
                        std_decibel=StdDev("noise_analysis__mean_db"),
                        max_decibel=Max("noise_analysis__max_db"),
                        min_decibel=Min("noise_analysis__min_db"),
                        avg_rms=Avg("audio_features__rms_energy"),
                        std_rms=StdDev("audio_features__rms_energy"),
                    )
                    .order_by("-avg_decibel")
                )

            elif group_by == "region":
                results = list(
                    queryset.values("region__name")
                    .annotate(
                        dataset_count=Count("id"),
                        avg_decibel=Avg("noise_analysis__mean_db"),
                        std_decibel=StdDev("noise_analysis__mean_db"),
                        quartile_range=Max("noise_analysis__max_db")
                        - Min("noise_analysis__min_db"),
                        avg_rms=Avg("audio_features__rms_energy"),
                    )
                    .order_by("-avg_decibel")
                )

            else:
                # Overall statistics
                stats = queryset.aggregate(
                    total_datasets=Count("id"),
                    avg_decibel=Avg("noise_analysis__mean_db"),
                    std_decibel=StdDev("noise_analysis__mean_db"),
                    max_decibel=Max("noise_analysis__max_db"),
                    min_decibel=Min("noise_analysis__min_db"),
                    avg_rms=Avg("audio_features__rms_energy"),
                    std_rms=StdDev("audio_features__rms_energy"),
                    avg_spectral_centroid=Avg("audio_features__spectral_centroid"),
                    std_spectral_centroid=StdDev("audio_features__spectral_centroid"),
                )
                results = [stats]

            return {
                "analysis_type": "statistical_analysis",
                "grouped_by": group_by,
                "results": results,
                "query": query,
                "summary": f"Statistical analysis of audio features grouped by {group_by}",
            }

        except Exception as e:
            return {"error": f"Statistical analysis failed: {str(e)}"}

    def _temporal_analysis(self, queryset, query):
        """Analyze trends over time"""
        try:
            # Monthly trends.
            # NOTE (P4-3): Sum(...) over a TruncMonth group is the PER-PERIOD
            # energy total, not a cumulative one. "cumulative" must be a running
            # total across ordered periods: cumulative_energy[i] = sum of every
            # period_energy up to and including period i. We keep the per-period
            # value as ``period_energy`` (for charts that plot it) and overwrite
            # ``cumulative_energy`` with the true running prefix sum.
            monthly_trends = list(
                queryset.annotate(month=TruncMonth("recording_date"))
                .values("month")
                .annotate(
                    dataset_count=Count("id"),
                    avg_decibel=Avg("noise_analysis__mean_db"),
                    avg_energy=Avg("audio_features__rms_energy"),
                    period_energy=Sum("audio_features__rms_energy"),
                )
                .order_by("month")
            )
            monthly_cumulative = running_cumulative(
                [row["period_energy"] for row in monthly_trends]
            )
            for row, cum in zip(monthly_trends, monthly_cumulative):
                row["cumulative_energy"] = cum

            # Daily trends (last 30 days)
            from datetime import datetime, timedelta

            thirty_days_ago = timezone.now() - timedelta(days=30)

            daily_trends = list(
                queryset.filter(recording_date__gte=thirty_days_ago)
                .annotate(date=TruncDate("recording_date"))
                .values("date")
                .annotate(
                    dataset_count=Count("id"),
                    avg_decibel=Avg("noise_analysis__mean_db"),
                    period_energy=Sum("audio_features__rms_energy"),
                )
                .order_by("date")
            )
            daily_cumulative = running_cumulative(
                [row["period_energy"] for row in daily_trends]
            )
            for row, cum in zip(daily_trends, daily_cumulative):
                row["cumulative_energy"] = cum

            return {
                "analysis_type": "temporal_analysis",
                "monthly_trends": monthly_trends,
                "daily_trends": daily_trends,
                "query": query,
                "summary": f"Temporal analysis showing trends over time",
            }

        except Exception as e:
            return {"error": f"Temporal analysis failed: {str(e)}"}

    def _overview_analysis(self, queryset, query):
        """General overview analysis"""
        try:
            # Overall statistics
            overview = queryset.aggregate(
                total_datasets=Count("id"),
                avg_rms_energy=Avg("audio_features__rms_energy"),
                avg_decibel=Avg("noise_analysis__mean_db"),
                max_decibel=Max("noise_analysis__max_db"),
                avg_duration=Avg("audio_features__duration"),
                total_duration=Sum("audio_features__duration"),
                avg_spectral_centroid=Avg("audio_features__spectral_centroid"),
                avg_dominant_frequency=Avg("noise_analysis__dominant_frequency"),
            )

            # Regional breakdown
            regional_stats = list(
                queryset.values("region__name")
                .annotate(count=Count("id"), avg_decibel=Avg("noise_analysis__mean_db"))
                .order_by("-count")
            )

            # Category breakdown
            category_stats = list(
                queryset.values("category__name")
                .annotate(count=Count("id"), avg_decibel=Avg("noise_analysis__mean_db"))
                .order_by("-count")
            )

            return {
                "analysis_type": "overview_analysis",
                "overview_statistics": overview,
                "regional_breakdown": regional_stats,
                "category_breakdown": category_stats,
                "query": query,
                "summary": f"Comprehensive overview of audio data characteristics",
            }

        except Exception as e:
            return {"error": f"Overview analysis failed: {str(e)}"}


class NoiseDetailTool(BaseTool):
    name: str = "get_noise_dataset_details"
    description: str = """Get detailed information about a specific noise dataset
    including metadata, audio features, analysis results, and visualizations."""

    def _run(
        self,
        noise_id: int,
        include_audio_features: bool = True,
        include_analysis: bool = True,
        include_visualizations: bool = False,
        include_collector: bool = True,
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        try:
            dataset = NoiseDataset.objects.get(id=noise_id)

            # Basic dataset info
            dataset_data = {
                "id": dataset.id,
                "name": dataset.name,
                "region": dataset.region.name if dataset.region else None,
                "community": dataset.community.name if dataset.community else None,
                "mean_db": (
                    getattr(dataset.noise_analysis, "mean_db", None)
                    if hasattr(dataset, "noise_analysis") and dataset.noise_analysis
                    else None
                ),
                "recording_date": dataset.recording_date,
                "recording_device": dataset.recording_device,
                # Raw-storage minimization: expose only a presence signal, never
                # the raw `audio` filesystem path / object-storage key / signed
                # URL (e.g. DO Spaces). The path itself is sensitive and is not
                # useful to the chat/LLM.
                "has_audio": bool(getattr(dataset, "audio", None)),
            }

            # Include metadata
            if include_metadata:
                dataset_data["metadata"] = {
                    "category": getattr(dataset, "category", None),
                    "region": getattr(dataset, "region", None),
                    "community": getattr(dataset, "community", None),
                    "class_name": getattr(dataset, "class_name", None),
                    "subclass": getattr(dataset, "subclass", None),
                    "microphone_type": getattr(dataset, "microphone_type", None),
                    "time_of_day": getattr(dataset, "time_of_day", None),
                }

            # Include collector (uploader/owner) — email omitted: PII, not needed by LLM
            if include_collector and hasattr(dataset, "collector"):
                dataset_data["collector"] = {
                    "id": dataset.collector.id,
                    "username": getattr(dataset.collector, "username", None),
                }

            # Include extracted audio features
            if include_audio_features and hasattr(dataset, "audio_features"):
                features = dataset.audio_features
                dataset_data["audio_features"] = {
                    "sample_rate": features.sample_rate,
                    "num_samples": features.num_samples,
                    "duration": features.duration,
                    "rms_energy": features.rms_energy,
                    "zero_crossing_rate": features.zero_crossing_rate,
                    "spectral_centroid": features.spectral_centroid,
                    "spectral_bandwidth": features.spectral_bandwidth,
                    "spectral_rolloff": features.spectral_rolloff,
                    "spectral_flatness": features.spectral_flatness,
                    "mfccs": features.mfccs,
                    "chroma_stft": features.chroma_stft,
                    "harmonic_ratio": features.harmonic_ratio,
                    "percussive_ratio": features.percussive_ratio,
                }

            # Include analysis results (assuming you store them in related models/fields)
            if include_analysis and hasattr(dataset, "analysis_results"):
                dataset_data["analysis_results"] = dataset.analysis_results

            # Include visualization presets
            if include_visualizations and hasattr(dataset, "audio_features"):
                features = dataset.audio_features
                dataset_data["visualizations"] = {
                    "mel_spectrogram": features.mel_spectrogram,
                    "waveform_data": features.waveform_data,
                }

            return dataset_data

        except NoiseDataset.DoesNotExist:
            return {"error": f"Noise dataset with ID {noise_id} not found"}
        except Exception as e:
            logger.error(f"Error getting noise dataset details: {str(e)}")
            return {"error": f"Failed to get noise dataset details: {str(e)}"}


_llm: Optional[ChatOpenAI] = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        _llm = ChatOpenAI(
            model=AGENT_CONFIG.get("MODEL", "gpt-4"),
            api_key=api_key,
            # Cap output per call to bound worst-case SQL-agent spend. Mirrors
            # the agent/dashboard LLMs in views.py (AI_INSIGHT MAX_TOKENS).
            max_tokens=AGENT_CONFIG.get("MAX_TOKENS", 2000),
            streaming=True,
        )
    return _llm


allowed_tables = SECURITY_CONFIG.get("DEFAULT_ALLOWED_TABLES", [])


def _get_default_allowed_tables() -> List[str]:
    """Discover data/core tables for SQL tools when not explicitly configured.

    PII reachability (Phase 3): only the ``data`` and ``core`` apps are
    discovered here. The ``authentication`` app — which owns the only direct
    PII columns (CustomUser.email / phone_number / first_name / last_name on
    ``authentication_customuser``, plus ``authentication_userotp``) — is NOT in
    scope, so the read-only SQL agent cannot reach contact PII. Dataset FKs
    (collector_id / contributor_id) are opaque integers, not PII.

    Audio storage paths: the allowed ``data_noisedataset`` / ``data_recording``
    / ``data_cleanspeechdataset`` tables carry an ``audio`` column holding the
    raw storage path / object-storage key. These tables are kept (excluding them
    would break org-wide aggregate analytics, the product's core feature). The
    ``audio`` column is masked at the schema layer: SQLDatabaseWrapper strips it
    (see ``SENSITIVE_COLUMN_NAMES`` in sql_agent.py) from BOTH the CREATE TABLE
    schema and the sample-row preview injected into {table_info}, so its raw
    values never reach the LLM even before the agent generates SQL. The
    SQL_SYSTEM_TEMPLATE prompt rule (never SELECT ``audio``) is a secondary,
    defense-in-depth layer on top of that schema-level masking.
    """
    # Tables that should never be exposed to the SQL agent — admin/internal
    # tables whose FKs point outside the allowed schema or that aren't useful
    # for data insights queries.
    _TABLE_BLOCKLIST = {
        "data_bulkaudioupload",
        "data_bulkreprocessingtask",
    }
    try:
        from django.apps import apps

        tables: List[str] = []
        for app_label in ("data", "core"):
            try:
                app_config = apps.get_app_config(app_label)
            except LookupError:
                continue
            for model in app_config.get_models():
                tables.append(model._meta.db_table)
        return sorted(set(tables) - _TABLE_BLOCKLIST)
    except Exception as e:
        logger.warning(f"Failed to discover allowed tables: {e}")
        return []


if not allowed_tables:
    allowed_tables = _get_default_allowed_tables()


_DATA_ANALYSIS_GROUP_FIELD_MAP: Dict[str, str] = {
    "region": "region__name",
    "community": "community__name",
    "category": "category__name",
    "class": "class_name__name",
    "recording_device": "recording_device",
}


class DataAnalysisInput(BaseModel):
    query: str = Field(
        description="The natural language question about audio dataset records"
    )
    query_type: Literal[
        "recent_datasets",
        "top_collectors_monthly",
        "category_class_count",
        "group_count",
        "dataset_count",
        "decibel_ranked",
        "decibel_grouped",
        "sql",
    ] = Field(
        description=(
            "The query category — pick the closest match:\n"
            "- 'recent_datasets': latest/most recent N recordings\n"
            "- 'top_collectors_monthly': who contributed the most datasets this month\n"
            "- 'category_class_count': dataset counts broken down by category × class\n"
            "- 'group_count': dataset counts grouped by one dimension (region, community, category, class, device)\n"
            "- 'dataset_count': scalar count, optionally filtered by region, category, date, or device\n"
            "- 'decibel_ranked': top or bottom N recordings ordered by decibel level\n"
            "- 'decibel_grouped': average decibel aggregated by a group dimension\n"
            "- 'sql': complex, multi-condition, or free-form question — routes to the SQL agent"
        )
    )
    limit: Optional[int] = Field(
        default=None,
        description="Row cap for 'recent_datasets' and 'decibel_ranked'. Defaults to 10 if omitted.",
    )
    group_by: Optional[
        Literal["region", "community", "category", "class", "recording_device"]
    ] = Field(
        default=None,
        description="Required for 'group_count' and 'decibel_grouped'. Which dimension to aggregate by.",
    )
    sort_direction: Optional[Literal["highest", "lowest"]] = Field(
        default="highest",
        description="For 'decibel_ranked': 'highest' returns loudest recordings, 'lowest' returns quietest.",
    )
    force_entity: Optional[str] = Field(
        default=None,
        description=(
            "When set by the clarification gate (dimension='entity'), use this "
            "entity name as a hard constraint for filtering or grouping. "
            "Set to 'all' if the user chose 'All areas'. Overrides group_by "
            "and any entity name extraction from the query."
        ),
    )
    force_time_range: Optional[str] = Field(
        default=None,
        description=(
            "When set by the clarification gate (dimension='time_range'), use "
            "this as a hard constraint. Values: 'last_7_days', 'last_30_days', "
            "'last_3_months', 'last_year', 'all_time'. Overrides any date filter "
            "extraction from the query."
        ),
    )


class DataAnalysisTool(BaseTool):
    name: str = "data_analysis"
    description: str = """Queries audio dataset records — counts, rankings, groupings, and recent uploads.
    Use this for:
    - Dataset counts by region, category, class, community, or device
    - Highest/lowest decibel recordings
    - Recent/latest N datasets
    - Top collectors this month
    - Dataset counts grouped by any dimension
    - Complex or filtered questions that require SQL"""
    agent: Any = None
    top_k: int = 10
    args_schema: type[BaseModel] = DataAnalysisInput

    def _get_agent(self):
        """Lazily build and cache the compiled TextToSQL agent.

        Built on first SQL invocation rather than at construction time: the
        TextToSQLAgent opens a DB connection (schema reflection) when created, so
        building it eagerly made merely *constructing* this tool — and therefore
        loading the whole agent tool set — fail whenever the DB was unreachable.
        Deferring it keeps tool construction DB-free and avoids paying the
        engine/reflection cost until an SQL query is actually needed.
        """
        if self.agent is None:
            from .prompt import SQL_SYSTEM_TEMPLATE

            self.agent = TextToSQLAgent(
                llm=_get_llm(),
                system_prompt=SQL_SYSTEM_TEMPLATE,
                include_tables=allowed_tables,
                top_k=self.top_k,
                ai_answer=False,
                sample_rows_in_table_info=3,
                max_string_length=40,
                max_retries=2,
            ).compile_workflow()
        return self.agent

    def _match_entity_name(self, query_lower: str, names: List[str]) -> Optional[str]:
        if not names:
            return None
        for name in sorted([n for n in names if n], key=len, reverse=True):
            name_lower = str(name).lower()
            if name_lower and name_lower in query_lower:
                return str(name)
        return None

    def _match_dataset_type(self, query_lower: str) -> Optional[str]:
        try:
            for value, label in getattr(AudioDataset, "DATASET_TYPES", []):
                tokens = []
                if value:
                    tokens.append(str(value).lower())
                    tokens.append(str(value).lower().replace("_", " "))
                if label:
                    tokens.append(str(label).lower())
                if any(token and token in query_lower for token in tokens):
                    return str(value)
        except Exception:
            return None
        return None

    def _extract_device_value(self, query_lower: str) -> Optional[str]:
        device_match = re.search(
            r"(?:recording device|device|recorded with|recorded on|using)\s+([a-z0-9\s\-\+\.]+?)(?:\b(in|from|during|between|after|before|on|at|for|with)\b|$)",
            query_lower,
        )
        if device_match:
            return device_match.group(1).strip()

        device_keywords = [
            "iphone",
            "samsung",
            "pixel",
            "android",
            "huawei",
            "tecno",
            "infinix",
            "xiaomi",
            "oppo",
            "vivo",
        ]
        for keyword in device_keywords:
            if keyword in query_lower:
                return keyword
        return None

    def _extract_date_filters(self, query_lower: str) -> Dict[str, Any]:
        date_filters: Dict[str, Any] = {}
        date_matches = re.findall(r"\b(20\d{2}-\d{1,2}-\d{1,2})\b", query_lower)
        if len(date_matches) >= 2:
            try:
                start = datetime.strptime(date_matches[0], "%Y-%m-%d").date()
                end = datetime.strptime(date_matches[1], "%Y-%m-%d").date()
                date_filters["recording_date__date__range"] = (start, end)
                return date_filters
            except Exception:
                return date_filters
        if len(date_matches) == 1:
            try:
                single_date = datetime.strptime(date_matches[0], "%Y-%m-%d").date()
                date_filters["recording_date__date"] = single_date
                return date_filters
            except Exception:
                return date_filters

        year_matches = re.findall(r"\b(20\d{2})\b", query_lower)
        if year_matches:
            years = [int(y) for y in year_matches]
            if any(
                word in query_lower for word in ["between", "from", "to", "through"]
            ):
                date_filters["recording_date__year__gte"] = min(years)
                date_filters["recording_date__year__lte"] = max(years)
            else:
                date_filters["recording_date__year"] = years[0]
        return date_filters

    def _run(
        self,
        query: str,
        query_type: str = "sql",
        limit: Optional[int] = None,
        group_by: Optional[str] = None,
        sort_direction: str = "highest",
        force_entity: Optional[str] = None,
        force_time_range: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        # force_entity takes precedence over LLM-supplied group_by.
        if force_entity and force_entity != "all":
            group_by = "region"
        elif force_entity == "all":
            group_by = None

        # force_time_range — gate-resolved time range. Map known values
        # to recency limits for _recent_datasets or date-filtered queries.
        _time_range_limit_map = {
            "last_7_days": 7,
            "last_30_days": 30,
            "last_3_months": None,  # handled by date filter, not recency count
            "last_year": None,
            "all_time": None,
        }
        if force_time_range and force_time_range in _time_range_limit_map:
            forced_limit = _time_range_limit_map[force_time_range]
            if forced_limit is not None and limit is None:
                limit = forced_limit
            # For "last_3_months" and "last_year": the LLM should set
            # date filters in the query — this is documented via the
            # CLARIFICATION_CONTEXT_TEMPLATE.

        try:
            if query_type == "recent_datasets":
                return self._recent_datasets(limit or 10)
            if query_type == "top_collectors_monthly":
                return self._top_collectors_monthly()
            if query_type == "category_class_count":
                return self._category_class_count()
            if query_type == "group_count":
                return self._group_count(group_by)
            if query_type == "dataset_count":
                return self._dataset_count(query)
            if query_type == "decibel_ranked":
                return self._decibel_ranked(limit or 10, sort_direction)
            if query_type == "decibel_grouped":
                return self._decibel_grouped(group_by)
            return self._invoke_sql_agent(query)
        except Exception as e:
            error_str = str(e) if e else repr(e)
            logger.error(f"Error in data analysis tool: {error_str}")
            return {
                "error": True,
                "message": (
                    f"The query could not be processed directly. "
                    f"Ask the user to rephrase with a specific dimension — "
                    f"for example: 'by region', 'by category', 'over time', "
                    f"or name a specific recording or device."
                ),
            }

    def _recent_datasets(self, limit: int) -> Dict[str, Any]:
        limit = max(1, min(limit, 200))
        qs = NoiseDataset.objects.select_related(
            "region", "community", "category", "dataset_type"
        ).order_by("-recording_date", "-created_at")
        total_count = qs.count()
        rows = [
            {
                "name": item.name,
                "region": item.region.name if item.region else None,
                "community": item.community.name if item.community else None,
                "category": item.category.name if item.category else None,
                "recording_date": item.recording_date,
                "recording_device": item.recording_device,
            }
            for item in qs[:limit]
        ]
        return {
            "analysis_type": "recent_datasets",
            "rows": rows,
            "columns": [
                "name",
                "region",
                "community",
                "category",
                "recording_date",
                "recording_device",
            ],
            "row_count": len(rows),
            "limit": limit,
            "offset": 0,
            "total_count": total_count,
            "has_more": total_count > limit,
            "skip_visualization": False,
            "chart_hint": {
                "x": "recording_date",
                "y": "count",
                "group_by": "category",
            },
        }

    def _top_collectors_monthly(self) -> Dict[str, Any]:
        now = timezone.now()
        # PII minimization: select only id + username for the leaderboard.
        # first_name/last_name (full legal name) are direct PII and must not be
        # surfaced to the chat/LLM. username is an acceptable display handle.
        #
        # P4-6: a "this month" collector leaderboard measures who CONTRIBUTED
        # (uploaded/collected) data this month, so it must filter on created_at
        # (upload/collection timestamp, auto_now_add) — NOT recording_date,
        # which is when the audio was originally recorded and can be far in the
        # past (or null) for a dataset uploaded today.
        collector_rows = list(
            NoiseDataset.objects.filter(
                collector__isnull=False,
                created_at__year=now.year,
                created_at__month=now.month,
            )
            .values(
                "collector__id",
                "collector__username",
            )
            .annotate(dataset_count=models.Count("id"))
            .order_by("-dataset_count")[:10]
        )
        rows = []
        for row in collector_rows:
            username = row.get("collector__username") or "Unknown"
            rows.append(
                {
                    "collector_id": row.get("collector__id"),
                    "collector": username,
                    "username": username,
                    "dataset_count": row.get("dataset_count", 0),
                }
            )
        return {
            "analysis_type": "top_collectors_monthly",
            "rows": rows,
            "columns": ["collector_id", "collector", "username", "dataset_count"],
            "row_count": len(rows),
            "limit": 10,
            "offset": 0,
            "has_more": False,
            "period": now.strftime("%Y-%m"),
            "skip_visualization": False,
            "chart_hint": {
                "x": "collector",
                "y": "dataset_count",
            },
        }

    def _category_class_count(self) -> Dict[str, Any]:
        qs = (
            NoiseDataset.objects.values("category__name", "class_name__name")
            .annotate(count=models.Count("id"))
            .order_by("category__name", "-count")
        )
        rows = [
            {
                "category": r["category__name"],
                "class": r["class_name__name"],
                "count": r["count"],
            }
            for r in qs
            if r["category__name"] and r["class_name__name"]
        ]
        return {
            "analysis_type": "category_class_count",
            "rows": rows,
            "columns": ["category", "class", "count"],
            "row_count": len(rows),
            "limit": len(rows),
            "offset": 0,
            "has_more": False,
            "skip_visualization": False,
            "chart_hint": {
                "x": "category",
                "y": "count",
                "group_by": "class",
            },
        }

    def _group_count(self, group_by: Optional[str]) -> Dict[str, Any]:
        field = _DATA_ANALYSIS_GROUP_FIELD_MAP.get(group_by or "") if group_by else None
        if not field:
            return {
                "error": f"Unknown group_by: {group_by!r}. Must be one of {list(_DATA_ANALYSIS_GROUP_FIELD_MAP)}"
            }
        rows = list(
            NoiseDataset.objects.values(field)
            .annotate(count=models.Count("id"))
            .order_by("-count")[:25]
        )
        return {
            "analysis_type": "group_count",
            "rows": rows,
            "columns": [field, "count"],
            "row_count": len(rows),
            "limit": 25,
            "offset": 0,
            "has_more": False,
            "chart_hint": {
                "x": field,
                "y": "count",
            },
        }

    def _dataset_count(self, query: str) -> Dict[str, Any]:
        query_lower = query.lower()
        filters: Dict[str, Any] = {}
        filter_meta: Dict[str, Any] = {}

        region_name = self._match_entity_name(
            query_lower, list(Region.objects.values_list("name", flat=True))
        )
        if region_name:
            filters["region__name__iexact"] = region_name
            filter_meta["region"] = region_name

        community_name = self._match_entity_name(
            query_lower, list(Community.objects.values_list("name", flat=True))
        )
        if community_name:
            filters["community__name__iexact"] = community_name
            filter_meta["community"] = community_name

        category_name = self._match_entity_name(
            query_lower, list(Category.objects.values_list("name", flat=True))
        )
        if category_name:
            filters["category__name__iexact"] = category_name
            filter_meta["category"] = category_name

        class_name = self._match_entity_name(
            query_lower, list(Class.objects.values_list("name", flat=True))
        )
        if class_name:
            filters["class_name__name__iexact"] = class_name
            filter_meta["class"] = class_name

        subclass_name = self._match_entity_name(
            query_lower, list(SubClass.objects.values_list("name", flat=True))
        )
        if subclass_name:
            filters["subclass__name__iexact"] = subclass_name
            filter_meta["subclass"] = subclass_name

        dataset_type_value = self._match_dataset_type(query_lower)
        if dataset_type_value:
            filters["dataset_type__name__iexact"] = dataset_type_value
            filter_meta["dataset_type"] = dataset_type_value

        date_filters = self._extract_date_filters(query_lower)
        if date_filters:
            filters.update(date_filters)
            filter_meta["recording_date"] = True

        device_value = self._extract_device_value(query_lower)
        qs = (
            NoiseDataset.objects.filter(**filters)
            if filters
            else NoiseDataset.objects.all()
        )
        if device_value:
            filter_meta["recording_device"] = device_value
            try:
                from django.contrib.postgres.search import TrigramSimilarity

                qs = qs.annotate(
                    similarity=TrigramSimilarity("recording_device", device_value)
                ).filter(similarity__gt=0.2)
            except Exception:
                qs = qs.filter(recording_device__icontains=device_value)

        return {
            "analysis_type": "dataset_count",
            "total_count": qs.count(),
            "filters": filter_meta,
            "limit": 0,
            "offset": 0,
            "has_more": False,
            "skip_visualization": True,
        }

    def _decibel_ranked(self, limit: int, sort_direction: str) -> Dict[str, Any]:
        limit = max(1, min(limit, 50))
        order = "-mean_db" if sort_direction == "highest" else "mean_db"
        qs = (
            NoiseAnalysis.objects.select_related(
                "noise_dataset",
                "noise_dataset__region",
                "noise_dataset__community",
                "noise_dataset__category",
            )
            .exclude(mean_db__isnull=True)
            .order_by(order)
        )[:limit]
        rows = []
        for item in qs:
            ds = item.noise_dataset
            rows.append(
                {
                    "name": ds.name if ds else None,
                    "mean_db": item.mean_db,
                    "region": ds.region.name if ds and ds.region else None,
                    "community": ds.community.name if ds and ds.community else None,
                    "category": ds.category.name if ds and ds.category else None,
                    "recording_date": ds.recording_date if ds else None,
                    "recording_device": ds.recording_device if ds else None,
                }
            )
        return {
            "analysis_type": f"{sort_direction}_decibel",
            "merge_group": "decibel_extremes",
            "rows": rows,
            "columns": [
                "name",
                "mean_db",
                "region",
                "community",
                "category",
                "recording_date",
                "recording_device",
            ],
            "row_count": len(rows),
            "limit": limit,
            "offset": 0,
            "has_more": False,
            "chart_hint": {
                "x": "name",
                "y": "mean_db",
                "group_by": "category",
            },
        }

    def _decibel_grouped(self, group_by: Optional[str]) -> Dict[str, Any]:
        field = _DATA_ANALYSIS_GROUP_FIELD_MAP.get(group_by or "") if group_by else None
        if not field:
            return {
                "error": f"Unknown group_by: {group_by!r}. Must be one of {list(_DATA_ANALYSIS_GROUP_FIELD_MAP)}"
            }
        grouped = (
            NoiseDataset.objects.exclude(**{f"{field}__isnull": True})
            .values(field)
            .annotate(
                avg_db=models.Avg("noise_analysis__mean_db"),
                # P4-4: sample_count must equal the denominator of avg_db.
                # avg_db only averages rows whose noise_analysis.mean_db is
                # non-null (mean_db is nullable), so counting datasets (Count(id))
                # overstates how many values backed the average. Count the
                # non-null mean_db values instead so the count matches the mean's
                # denominator exactly.
                sample_count=models.Count("noise_analysis__mean_db"),
            )
            .exclude(avg_db__isnull=True)
            .order_by("-avg_db")
        )
        rows = [
            {
                group_by: r[field],
                "avg_db": r["avg_db"],
                "sample_count": r["sample_count"],
            }
            for r in grouped
        ]
        return {
            "analysis_type": f"avg_decibel_by_{group_by}",
            "merge_group": "decibel_by_dimension",
            "rows": rows,
            "columns": [group_by, "avg_db", "sample_count"],
            "row_count": len(rows),
            "limit": len(rows),
            "offset": 0,
            "has_more": False,
            "skip_visualization": False,
            "chart_hint": {
                "x": group_by,
                "y": "avg_db",
            },
        }

    def _invoke_sql_agent(self, query: str) -> Dict[str, Any]:
        # Bound the SQL agent's llm<->tool loop depth so a runaway loop cannot
        # rack up unbounded LLM spend. Mirrors RECURSION_LIMIT in agent_workflow.
        config = {"recursion_limit": AGENT_CONFIG.get("RECURSION_LIMIT", 15)}
        response = self._get_agent().invoke(
            {"messages": [HumanMessage(content=query)]}, config=config
        )
        if response and "messages" in response and response["messages"]:
            last_message = response["messages"][-1]
            msg = (
                str(last_message.content)
                if hasattr(last_message, "content") and last_message.content
                else str(last_message)
            )
        else:
            msg = "No response received"
        return {"message": msg}


class VisualizationAnalysisInput(BaseModel):
    """Input for visualization analysis"""

    query: str = Field(description="The user's query about data visualization")
    data_summary: Optional[str] = Field(
        default=None, description="Summary of the data to be visualized"
    )


class VisualizationAnalysisTool(BaseTool):
    """Tool for analyzing data and recommending the best visualization type.

    Delegates chart type selection to chart_builder's pure-Python decision tree.
    No LLM calls — deterministic, instant, and debuggable.
    """

    name: str = "visualization_analysis"
    description: str = """
    Analyzes audio data and recommends the best visualization type for the given audio data and query.
    Supports: pie chart, bar chart, line chart, heatmap, scatter plot, box plot, area chart.
    Specializes in audio-specific visualizations like frequency analysis, decibel trends, and spectral characteristics.
    Returns both the recommended chart type and a template for creating the visualization.
    """
    args_schema: Optional[type[BaseModel]] = VisualizationAnalysisInput

    def _run(
        self,
        query: str,
        data_summary: Optional[str] = None,
        data_block: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        from .chart_builder import resolve_chart

        if data_block and isinstance(data_block, dict):
            chart_config = resolve_chart(data_block)
            if chart_config:
                return {
                    "chart": chart_config,
                    "visualization_type": chart_config.get(
                        "visualization_type", "bar_chart"
                    ),
                    "visualization_name": chart_config.get(
                        "visualization_name", "Bar Chart"
                    ),
                    "frontend_data": chart_config.get("frontend_data", {}),
                    "message": f"Recommended {chart_config.get('visualization_name', 'Bar Chart')} for this data",
                }

        return {
            "chart": None,
            "visualization_type": "bar_chart",
            "visualization_name": "Bar Chart",
            "message": "No data available for visualization analysis",
        }


# Lazy initialization to avoid database connection at import time
_all_tools_cache: Optional[List[BaseTool]] = None


class MLProfileInput(BaseModel):
    """Input for ML dataset profile"""

    group_by: Optional[Literal["category", "region", "class", "subclass"]] = Field(
        default="category", description="Field to group label distribution by"
    )
    top_k: int = Field(default=10, description="Top groups to return")


class MLDatasetProfileTool(BaseTool):
    name: str = "ml_dataset_profile"
    description: str = (
        "Profile dataset readiness for ML: dataset size, label distribution, "
        "feature coverage, and missingness for key metadata fields."
    )
    args_schema: Optional[type[BaseModel]] = MLProfileInput

    def _run(
        self, group_by: Optional[str] = "category", top_k: int = 10, **kwargs
    ) -> Dict[str, Any]:
        try:
            total = NoiseDataset.objects.count()
            with_features = NoiseDataset.objects.filter(
                audio_features__isnull=False
            ).count()
            with_analysis = NoiseDataset.objects.filter(
                noise_analysis__isnull=False
            ).count()

            dataset_types = (
                NoiseDataset.objects.values("dataset_type").distinct().count()
            )

            missing = {
                "region": NoiseDataset.objects.filter(region__isnull=True).count(),
                "category": NoiseDataset.objects.filter(category__isnull=True).count(),
                "community": NoiseDataset.objects.filter(
                    community__isnull=True
                ).count(),
                "recording_date": NoiseDataset.objects.filter(
                    recording_date__isnull=True
                ).count(),
                "recording_device": NoiseDataset.objects.filter(
                    recording_device__isnull=True
                ).count(),
                "microphone_type": NoiseDataset.objects.filter(
                    microphone_type__isnull=True
                ).count(),
            }

            group_field = "category__name"
            if group_by == "region":
                group_field = "region__name"
            elif group_by == "class":
                group_field = "class_name__name"
            elif group_by == "subclass":
                group_field = "subclass__name"

            distribution = (
                NoiseDataset.objects.values(group_field)
                .annotate(count=models.Count("id"))
                .order_by("-count")[: max(1, int(top_k))]
            )

            return {
                "analysis_type": "ml_dataset_profile",
                "totals": {
                    "noise_datasets": total,
                    "dataset_types": dataset_types,
                    "with_audio_features": with_features,
                    "with_noise_analysis": with_analysis,
                },
                "coverage": {
                    "features_pct": (with_features / total * 100) if total else 0,
                    "analysis_pct": (with_analysis / total * 100) if total else 0,
                },
                "missingness": missing,
                "label_distribution": {
                    "group_by": group_by,
                    "top_k": top_k,
                    "counts": list(distribution),
                },
            }
        except Exception as e:
            logger.error(f"Error in ML dataset profile: {e}")
            return {"error": get_user_friendly_error(str(e), "profiling ML dataset")}


class MLFeatureStatsInput(BaseModel):
    """Input for ML feature stats"""

    include_decibels: bool = Field(
        default=True, description="Include noise analysis decibel stats"
    )


class MLFeatureStatsTool(BaseTool):
    name: str = "ml_feature_stats"
    description: str = "Compute summary statistics for ML-relevant audio features."
    args_schema: Optional[type[BaseModel]] = MLFeatureStatsInput

    def _run(self, include_decibels: bool = True, **kwargs) -> Dict[str, Any]:
        try:
            feature_stats = AudioFeature.objects.aggregate(
                count=models.Count("id"),
                avg_duration=models.Avg("duration"),
                std_duration=models.StdDev("duration"),
                avg_rms=models.Avg("rms_energy"),
                std_rms=models.StdDev("rms_energy"),
                avg_centroid=models.Avg("spectral_centroid"),
                std_centroid=models.StdDev("spectral_centroid"),
                avg_bandwidth=models.Avg("spectral_bandwidth"),
                std_bandwidth=models.StdDev("spectral_bandwidth"),
                avg_zcr=models.Avg("zero_crossing_rate"),
                std_zcr=models.StdDev("zero_crossing_rate"),
            )

            response = {
                "analysis_type": "ml_feature_stats",
                "audio_features": feature_stats,
            }

            if include_decibels:
                decibel_stats = NoiseAnalysis.objects.aggregate(
                    count=models.Count("id"),
                    avg_mean_db=models.Avg("mean_db"),
                    std_mean_db=models.StdDev("mean_db"),
                    max_db=models.Max("max_db"),
                    min_db=models.Min("min_db"),
                )
                response["noise_analysis"] = decibel_stats

            return response
        except Exception as e:
            logger.error(f"Error in ML feature stats: {e}")
            return {
                "error": get_user_friendly_error(str(e), "computing ML feature stats")
            }


class MLClassBalanceInput(BaseModel):
    """Input for ML class balance analysis."""

    label_column: Literal["category", "region", "class", "subclass"] = Field(
        default="category", description="Label column to analyze class balance for"
    )


class MLClassBalanceTool(BaseTool):
    name: str = "ml_class_balance"
    description: str = (
        "Analyze class balance for a label column. Returns per-class counts, "
        "percentages, minority/majority ratio, imbalance severity, Shannon entropy, "
        "and stratified split recommendations. Use when the user asks about class "
        "distribution, balance, or dataset readiness for classification."
    )
    args_schema: Optional[type[BaseModel]] = MLClassBalanceInput

    def _run(self, label_column: str = "category", **kwargs) -> Dict[str, Any]:
        import math

        try:
            field_map = {
                "category": "category__name",
                "region": "region__name",
                "class": "class_name__name",
                "subclass": "subclass__name",
            }
            field = field_map.get(label_column, "category__name")

            distribution = list(
                NoiseDataset.objects.values(field)
                .annotate(count=models.Count("id"))
                .order_by("-count")
            )

            if not distribution:
                return {
                    "error": "insufficient_data",
                    "reason": "No rows found.",
                    "rows_available": 0,
                    "skip_visualization": True,
                }

            total = sum(r["count"] for r in distribution)
            if total < 10:
                return {
                    "error": "insufficient_data",
                    "reason": f"Only {total} rows found. Minimum is 10.",
                    "rows_available": total,
                    "skip_visualization": True,
                }

            n_classes = len(distribution)
            if n_classes < 2:
                return {
                    "error": "insufficient_data",
                    "reason": f"Only {n_classes} class present. Need at least 2 classes.",
                    "rows_available": total,
                    "skip_visualization": True,
                }

            # Per-class stats
            per_class = []
            for r in distribution:
                name = r.get(field) or "Unknown"
                count = r["count"]
                pct = (count / total * 100) if total else 0
                per_class.append(
                    {
                        "label": str(name),
                        "count": count,
                        "percentage": round(pct, 1),
                    }
                )

            # Minority / majority
            max_count = max(r["count"] for r in per_class)
            min_count = min(r["count"] for r in per_class)
            minority_ratio = min_count / max_count if max_count > 0 else 0
            minority_pct = (min_count / total * 100) if total > 0 else 0

            # Imbalance severity
            if minority_pct < 5:
                severity = "severe"
            elif minority_pct < 15:
                severity = "moderate"
            elif minority_pct < 30:
                severity = "mild"
            else:
                severity = "balanced"

            # Shannon entropy (normalized)
            entropy = 0.0
            for r in per_class:
                p = r["count"] / total if total > 0 else 0
                if p > 0:
                    entropy -= p * math.log2(p)
            max_entropy = math.log2(n_classes) if n_classes > 1 else 1
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

            # Stratified split recommendation (70/15/15).
            # P4-7: use largest-remainder allocation so train + val + test == c
            # exactly and never exceeds the class size. Independent rounding with
            # max(1, round(...)) over-allocated small classes (e.g. c=2 produced
            # 1/1/1 == 3 > 2), recommending more rows than the class contains.
            split_recs = []
            for r in per_class:
                c = r["count"]
                train, val, test = stratified_split_counts(c, 0.70, 0.15, 0.15)
                split_recs.append(
                    {
                        "label": r["label"],
                        "train": train,
                        "val": val,
                        "test": test,
                        "total": c,
                    }
                )

            # Warn if any class has < 30 in val+test combined
            low_sample_classes = [
                r["label"] for r in split_recs if (r["val"] + r["test"]) < 30
            ]

            return {
                "analysis_type": "ml_class_balance",
                "label_column": label_column,
                "total_rows": total,
                "n_classes": n_classes,
                "minority_ratio": round(minority_ratio, 4),
                "minority_pct": round(minority_pct, 1),
                "imbalance_severity": severity,
                "shannon_entropy": round(entropy, 3),
                "normalized_entropy": round(normalized_entropy, 3),
                "per_class": per_class,
                "split_recommendation": split_recs,
                "low_sample_classes": low_sample_classes,
                "warnings": (
                    [
                        f"Classes with <30 samples in val+test combined: {', '.join(low_sample_classes)}"
                    ]
                    if low_sample_classes
                    else []
                ),
            }
        except Exception as e:
            logger.error(f"Error in ml_class_balance: {e}")
            return {"error": get_user_friendly_error(str(e), "analyzing class balance")}


class MLTrainTestSplitInput(BaseModel):
    """Input for ML train/test split calculation."""

    label_column: Literal["category", "region", "class", "subclass"] = Field(
        default="category", description="Label column to stratify by"
    )
    train_pct: float = Field(
        default=0.70,
        ge=0.1,
        le=0.9,
        description="Fraction of data for training (0.1-0.9)",
    )
    val_pct: float = Field(
        default=0.15,
        ge=0.05,
        le=0.5,
        description="Fraction of data for validation (0.05-0.5)",
    )
    test_pct: float = Field(
        default=0.15,
        ge=0.05,
        le=0.5,
        description="Fraction of data for testing (0.05-0.5)",
    )
    split_strategy: Literal["stratified", "random", "time_based"] = Field(
        default="stratified",
        description="Split strategy: stratified preserves class proportions, "
        "random shuffles uniformly, time_based splits by recording date",
    )
    seed: int = Field(default=42, description="Random seed for reproducibility")


class MLTrainTestSplitTool(BaseTool):
    name: str = "ml_train_test_split"
    description: str = (
        "Compute exact row counts for train/val/test splits. Supports stratified "
        "(preserves class proportions), random (uniform shuffle), and time-based "
        "(split by recording date) strategies. Returns per-class breakdown per split. "
        "Uses a fixed seed for reproducibility."
    )
    args_schema: Optional[type[BaseModel]] = MLTrainTestSplitInput

    def _run(
        self,
        label_column: str = "category",
        train_pct: float = 0.70,
        val_pct: float = 0.15,
        test_pct: float = 0.15,
        split_strategy: str = "stratified",
        seed: int = 42,
        **kwargs,
    ) -> Dict[str, Any]:
        import random as _random

        try:
            field_map = {
                "category": "category__name",
                "region": "region__name",
                "class": "class_name__name",
                "subclass": "subclass__name",
            }
            field = field_map.get(label_column, "category__name")

            total = NoiseDataset.objects.count()
            if total < 10:
                return {
                    "error": "insufficient_data",
                    "reason": f"Only {total} rows found. Minimum is 10.",
                    "rows_available": total,
                    "skip_visualization": True,
                }

            # Verify splits sum to ~1
            total_pct = train_pct + val_pct + test_pct
            if abs(total_pct - 1.0) > 0.01:
                return {
                    "error": "invalid_splits",
                    "reason": f"Split percentages sum to {total_pct}, should be 1.0",
                    "skip_visualization": True,
                }

            if split_strategy == "stratified":
                distribution = list(
                    NoiseDataset.objects.values(field)
                    .annotate(count=models.Count("id"))
                    .order_by("-count")
                )

                splits_per_class = []
                for r in distribution:
                    c = r["count"]
                    # P4-7: largest-remainder allocation guarantees
                    # train + val + test == c exactly and never exceeds the
                    # class size. The previous code rounded each split
                    # independently and floored to >=1, which over-allocated
                    # small classes (e.g. c=2 -> 1/1/1 == 3 > 2).
                    train, val, test = stratified_split_counts(
                        c, train_pct, val_pct, test_pct
                    )
                    splits_per_class.append(
                        {
                            "label": str(r.get(field) or "Unknown"),
                            "total": c,
                            "train": train,
                            "val": val,
                            "test": test,
                        }
                    )

                train_total = sum(s["train"] for s in splits_per_class)
                val_total = sum(s["val"] for s in splits_per_class)
                test_total = sum(s["test"] for s in splits_per_class)

            elif split_strategy == "random":
                _random.seed(seed)
                train_total = max(1, round(total * train_pct))
                test_total = max(1, round(total * test_pct))
                val_total = max(1, total - train_total - test_total)
                splits_per_class = []

            elif split_strategy == "time_based":
                # Sort by recording date, earliest → train, middle → val, latest → test
                all_ids = list(
                    NoiseDataset.objects.order_by("recording_date").values_list(
                        "id", flat=True
                    )
                )
                n = len(all_ids)
                train_end = max(1, round(n * train_pct))
                val_end = max(train_end + 1, round(n * (train_pct + val_pct)))
                train_total = train_end
                val_total = val_end - train_end
                test_total = n - val_end
                splits_per_class = []

            else:
                return {
                    "error": "invalid_strategy",
                    "reason": f"Unknown split strategy: {split_strategy}",
                    "skip_visualization": True,
                }

            # Unstratifiable class check
            unstratifiable = []
            if splits_per_class:
                for s in splits_per_class:
                    if s["total"] < 3:
                        unstratifiable.append(s["label"])

            return {
                "analysis_type": "ml_train_test_split",
                "split_strategy": split_strategy,
                "seed": seed,
                "total_rows": total,
                "label_column": label_column,
                "train_count": train_total,
                "val_count": val_total,
                "test_count": test_total,
                "train_pct": round(train_total / total * 100, 1) if total else 0,
                "val_pct": round(val_total / total * 100, 1) if total else 0,
                "test_pct": round(test_total / total * 100, 1) if total else 0,
                "splits_per_class": splits_per_class if splits_per_class else None,
                "unstratifiable_classes": unstratifiable,
                "warnings": (
                    [
                        f"Classes with <3 samples — stratification impossible: {', '.join(unstratifiable)}"
                    ]
                    if unstratifiable
                    else []
                ),
            }
        except Exception as e:
            logger.error(f"Error in ml_train_test_split: {e}")
            return {
                "error": get_user_friendly_error(str(e), "computing train/test split")
            }


class MLCorrelationMatrixInput(BaseModel):
    """Input for ML correlation matrix."""

    method: Literal["spearman", "pearson", "both"] = Field(
        default="both", description="Correlation method: spearman, pearson, or both"
    )
    label_column: Optional[Literal["category", "region", "class", "subclass"]] = Field(
        default=None,
        description="Optional label column (accepted for backward compatibility; "
        "sampling is uniform random, not stratified)",
    )
    sample_size: int = Field(
        default=5000,
        ge=100,
        le=50000,
        description="Max rows to sample (capped for performance)",
    )
    seed: int = Field(default=42, description="Random seed for sampling")


class MLCorrelationMatrixTool(BaseTool):
    name: str = "ml_correlation_matrix"
    description: str = (
        "Compute pairwise Spearman ρ and/or Pearson r between all numeric audio "
        "feature columns. Returns N×N matrix, top-10 strongest pairs, and p-values. "
        "Correlations use complete-case rows (every selected feature non-null). "
        "Randomly samples up to 5,000 rows for performance."
    )
    args_schema: Optional[type[BaseModel]] = MLCorrelationMatrixInput

    def _run(
        self,
        method: str = "both",
        label_column: Optional[str] = None,
        sample_size: int = 5000,
        seed: int = 42,
        **kwargs,
    ) -> Dict[str, Any]:
        try:
            import numpy as np
            from scipy import stats as sp_stats
        except ImportError:
            return {
                "error": "Correlation computation requires scipy. Please install scipy.",
                "skip_visualization": True,
            }

        try:
            feature_cols = [
                "rms_energy",
                "zero_crossing_rate",
                "spectral_centroid",
                "spectral_bandwidth",
                "spectral_rolloff",
                "spectral_flatness",
                "duration",
                "harmonic_ratio",
                "percussive_ratio",
            ]

            # Fetch data
            qs = AudioFeature.objects.filter(
                rms_energy__isnull=False,
                spectral_centroid__isnull=False,
            ).select_related("noise_dataset")

            total = qs.count()
            if total < 5:
                return {
                    "error": "insufficient_data",
                    "reason": f"Only {total} rows with features found. Minimum is 5.",
                    "rows_available": total,
                    "skip_visualization": True,
                }

            # Sample
            actual_sample = min(sample_size, total)

            # P4-6: sampling is order_by("?") = uniform RANDOM in every branch
            # (no per-class quotas), so it must not be reported as "stratified".
            # The label_column branch only changes which columns are selected; it
            # does not stratify. We keep it as honest random sampling and label
            # it accurately as "random" in the output metadata below.
            ids = list(qs.values_list("id", flat=True).order_by("?")[:actual_sample])

            if not ids:
                return {
                    "error": "insufficient_data",
                    "reason": "No rows found after sampling.",
                    "rows_available": 0,
                    "skip_visualization": True,
                }

            # Build data matrix (ids is a flat list of AudioFeature primary keys)
            features_qs = AudioFeature.objects.filter(id__in=ids).values(*feature_cols)
            rows = list(features_qs)

            # P4-1: select which features can participate (each must have enough
            # non-null observations on its own), then align with COMPLETE-CASE
            # rows so row i of every feature comes from the SAME dataset.
            # The previous code dropped nulls per-column independently and then
            # position-truncated with [:min_len] + column_stack — pairing row i
            # of feature A with an unrelated row i of feature B and fabricating
            # every coefficient/p-value. Correct correlation requires the paired
            # observations to correspond to the same source row.
            non_null_counts = {
                col: sum(1 for r in rows if r.get(col) is not None)
                for col in feature_cols
            }
            valid_features = [c for c in feature_cols if non_null_counts[c] >= 5]
            if len(valid_features) < 2:
                return {
                    "error": "insufficient_data",
                    "reason": f"Only {len(valid_features)} features with sufficient data.",
                    "rows_available": len(rows),
                    "skip_visualization": True,
                }

            # Complete-case alignment over the selected features.
            aligned_columns, min_len = complete_case_matrix(rows, valid_features)
            if min_len < 5:
                return {
                    "error": "insufficient_data",
                    "reason": (
                        f"Only {min_len} complete-case rows (every selected "
                        "feature non-null) found. Minimum is 5."
                    ),
                    "rows_available": len(rows),
                    "skip_visualization": True,
                }

            # Compute correlations
            n_features = len(valid_features)
            spearman_matrix = None
            pearson_matrix = None
            spearman_pvals = None
            pearson_pvals = None

            aligned_data = np.column_stack(
                [np.asarray(aligned_columns[f], dtype=float) for f in valid_features]
            )

            if method in ("spearman", "both"):
                spearman_matrix = np.zeros((n_features, n_features))
                spearman_pvals = np.zeros((n_features, n_features))
                for i in range(n_features):
                    for j in range(i, n_features):
                        if i == j:
                            spearman_matrix[i][j] = 1.0
                            spearman_pvals[i][j] = 0.0
                        else:
                            rho, p = sp_stats.spearmanr(
                                aligned_data[:, i], aligned_data[:, j]
                            )
                            spearman_matrix[i][j] = spearman_matrix[j][i] = float(rho)
                            spearman_pvals[i][j] = spearman_pvals[j][i] = float(p)

            if method in ("pearson", "both"):
                pearson_matrix = np.zeros((n_features, n_features))
                pearson_pvals = np.zeros((n_features, n_features))
                for i in range(n_features):
                    for j in range(i, n_features):
                        if i == j:
                            pearson_matrix[i][j] = 1.0
                            pearson_pvals[i][j] = 0.0
                        else:
                            r, p = sp_stats.pearsonr(
                                aligned_data[:, i], aligned_data[:, j]
                            )
                            pearson_matrix[i][j] = pearson_matrix[j][i] = float(r)
                            pearson_pvals[i][j] = pearson_pvals[j][i] = float(p)

            # Top-10 strongest pairs, ranked by the matrix matching the requested
            # method. P4-2: a `primary_matrix` was previously assigned here but
            # never used (dead code / wrong-matrix hazard). Only the matrix(es)
            # for the requested ``method`` are computed (the others stay None),
            # so reading spearman first and falling back to pearson below is
            # already consistent with the request; the dead variable is removed.
            top_pairs = []
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    top_pairs.append(
                        {
                            "feature_a": valid_features[i],
                            "feature_b": valid_features[j],
                            "spearman_rho": (
                                round(float(spearman_matrix[i][j]), 4)
                                if spearman_matrix is not None
                                else None
                            ),
                            "spearman_p": (
                                round(float(spearman_pvals[i][j]), 6)
                                if spearman_pvals is not None
                                else None
                            ),
                            "pearson_r": (
                                round(float(pearson_matrix[i][j]), 4)
                                if pearson_matrix is not None
                                else None
                            ),
                            "pearson_p": (
                                round(float(pearson_pvals[i][j]), 6)
                                if pearson_pvals is not None
                                else None
                            ),
                        }
                    )
            top_pairs.sort(
                key=lambda x: abs(x.get("spearman_rho") or x.get("pearson_r") or 0),
                reverse=True,
            )
            top_pairs = top_pairs[:10]

            return {
                "analysis_type": "ml_correlation_matrix",
                "method": method,
                "features": valid_features,
                "n_features": n_features,
                # sample_size = number of complete-case rows actually correlated
                # (every selected feature non-null), so users see the real n.
                "sample_size": min_len,
                "n": min_len,
                "total_available": total,
                # P4-6: order_by("?") is uniform random sampling, not stratified.
                "sampling_method": "random",
                "seed": seed,
                "spearman_matrix": (
                    spearman_matrix.tolist() if spearman_matrix is not None else None
                ),
                "spearman_pvals": (
                    spearman_pvals.tolist() if spearman_pvals is not None else None
                ),
                "pearson_matrix": (
                    pearson_matrix.tolist() if pearson_matrix is not None else None
                ),
                "pearson_pvals": (
                    pearson_pvals.tolist() if pearson_pvals is not None else None
                ),
                "top_pairs": top_pairs,
            }
        except Exception as e:
            logger.error(f"Error in ml_correlation_matrix: {e}")
            return {
                "error": get_user_friendly_error(str(e), "computing correlation matrix")
            }


class MLStatisticalTestInput(BaseModel):
    """Input for ML statistical tests comparing two groups."""

    group_a_value: str = Field(description="Name of group A (e.g. 'Greater Accra')")
    group_b_value: str = Field(description="Name of group B (e.g. 'Ashanti')")
    group_by: Literal["region", "category", "class", "community"] = Field(
        default="region", description="Dimension to split groups by"
    )
    feature: str = Field(
        default="mean_db",
        description="Numeric feature to compare (mean_db, rms_energy, spectral_centroid, zero_crossing_rate, duration)",
    )


class MLStatisticalTestTool(BaseTool):
    name: str = "ml_statistical_test"
    description: str = (
        "Compare two groups on a numeric feature. Returns Welch's t-test "
        "(t, p, df), Mann-Whitney U (U, p), Cohen's d effect size, and "
        "a plain-English interpretation. E.g. 'Is region A louder than region B?'"
    )
    args_schema: Optional[type[BaseModel]] = MLStatisticalTestInput

    def _run(
        self,
        group_a_value: str,
        group_b_value: str,
        group_by: str = "region",
        feature: str = "mean_db",
        **kwargs,
    ) -> Dict[str, Any]:
        try:
            import numpy as np
            from scipy import stats as sp_stats
        except ImportError:
            return {
                "error": "Statistical test requires scipy. Please install scipy.",
                "skip_visualization": True,
            }

        try:
            field_map = {
                "region": "region__name",
                "category": "category__name",
                "class": "class_name__name",
                "community": "community__name",
            }
            group_field = field_map.get(group_by, "region__name")

            # Map feature name to ORM path
            if feature in (
                "mean_db",
                "max_db",
                "min_db",
                "std_db",
                "dominant_frequency",
            ):
                feature_path = f"noise_analysis__{feature}"
                feature_filter = {f"{feature_path}__isnull": False}
            elif feature == "duration":
                feature_path = f"audio_features__{feature}"
                feature_filter = {"audio_features__duration__isnull": False}
            else:
                feature_path = f"audio_features__{feature}"
                feature_filter = {f"{feature_path}__isnull": False}

            # Fetch group data
            filter_a = {f"{group_field}__iexact": group_a_value, **feature_filter}
            filter_b = {f"{group_field}__iexact": group_b_value, **feature_filter}

            values_a_raw = list(
                NoiseDataset.objects.filter(**filter_a).values_list(
                    feature_path, flat=True
                )[:10000]
            )
            values_b_raw = list(
                NoiseDataset.objects.filter(**filter_b).values_list(
                    feature_path, flat=True
                )[:10000]
            )

            values_a = np.array(
                [float(v) for v in values_a_raw if v is not None], dtype=float
            )
            values_b = np.array(
                [float(v) for v in values_b_raw if v is not None], dtype=float
            )

            n_a = len(values_a)
            n_b = len(values_b)

            if n_a < 3 or n_b < 3:
                return {
                    "error": "insufficient_data",
                    "reason": f"Group A has {n_a} samples, Group B has {n_b}. Both need at least 3.",
                    "rows_available": {"group_a": n_a, "group_b": n_b},
                    "skip_visualization": True,
                }

            # Welch's t-test
            t_stat, t_p = sp_stats.ttest_ind(values_a, values_b, equal_var=False)

            # Mann-Whitney U
            u_stat, u_p = sp_stats.mannwhitneyu(
                values_a, values_b, alternative="two-sided"
            )

            # Cohen's d
            mean_a = float(np.mean(values_a))
            mean_b = float(np.mean(values_b))
            std_a = float(np.std(values_a, ddof=1))
            std_b = float(np.std(values_b, ddof=1))
            pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
            cohens_d = float((mean_a - mean_b) / pooled_std) if pooled_std > 0 else 0.0

            # Effect size interpretation
            abs_d = abs(cohens_d)
            if abs_d < 0.2:
                effect_label = "negligible"
            elif abs_d < 0.5:
                effect_label = "small"
            elif abs_d < 0.8:
                effect_label = "medium"
            else:
                effect_label = "large"

            # Plain-English summary
            significance = "significantly" if t_p < 0.05 else "not significantly"
            direction = "higher" if mean_a > mean_b else "lower"
            summary = (
                f"{group_a_value} has {direction} {feature} than {group_b_value} "
                f"({significance} different: p={t_p:.4f}, d={cohens_d:.2f} {effect_label} effect)"
            )

            small_sample = n_a < 10 or n_b < 10

            return {
                "analysis_type": "ml_statistical_test",
                "group_by": group_by,
                "feature": feature,
                "group_a": {
                    "name": group_a_value,
                    "n": n_a,
                    "mean": round(mean_a, 4),
                    "std": round(std_a, 4),
                },
                "group_b": {
                    "name": group_b_value,
                    "n": n_b,
                    "mean": round(mean_b, 4),
                    "std": round(std_b, 4),
                },
                "welch_t_test": {
                    "t_statistic": round(float(t_stat), 4),
                    "p_value": round(float(t_p), 6),
                    "degrees_of_freedom": None,  # Welch's doesn't give a clean integer df
                },
                "mann_whitney_u": {
                    "u_statistic": round(float(u_stat), 4),
                    "p_value": round(float(u_p), 6),
                },
                "cohens_d": round(cohens_d, 4),
                "effect_size": effect_label,
                "plain_english": summary,
                "small_sample_warning": small_sample,
                "warnings": (
                    [
                        f"Small sample sizes (A: {n_a}, B: {n_b}) — results are underpowered"
                    ]
                    if small_sample
                    else []
                ),
            }
        except Exception as e:
            logger.error(f"Error in ml_statistical_test: {e}")
            return {
                "error": get_user_friendly_error(str(e), "running statistical test")
            }


class MLExportFeaturesInput(BaseModel):
    """Input for ML feature export."""

    label_column: Literal["category", "region", "class", "subclass"] = Field(
        default="category", description="Label column for y vector"
    )
    features: Optional[List[str]] = Field(
        default=None, description="Feature columns to include (defaults to all)"
    )
    format: Literal["flat", "split"] = Field(
        default="flat",
        description="'flat' = X+y in one table, 'split' = separate X and y",
    )
    max_rows: int = Field(
        default=10000, ge=100, le=100000, description="Max rows to export"
    )


class MLExportFeaturesTool(BaseTool):
    name: str = "ml_export_features"
    description: str = (
        "Export a feature matrix (X) and label vector (y) from the database. "
        "Returns CSV-formatted data for use in notebooks. Bridge from chatbot "
        "to ML training."
    )
    args_schema: Optional[type[BaseModel]] = MLExportFeaturesInput

    def _run(
        self,
        label_column: str = "category",
        features: Optional[List[str]] = None,
        format: str = "flat",
        max_rows: int = 10000,
        **kwargs,
    ) -> Dict[str, Any]:
        try:
            field_map = {
                "category": "category__name",
                "region": "region__name",
                "class": "class_name__name",
                "subclass": "subclass__name",
            }
            label_field = field_map.get(label_column, "category__name")

            if features is None:
                features = [
                    "rms_energy",
                    "zero_crossing_rate",
                    "spectral_centroid",
                    "spectral_bandwidth",
                    "spectral_rolloff",
                    "spectral_flatness",
                    "duration",
                ]

            # Build queryset with all joins
            qs = NoiseDataset.objects.select_related(
                "audio_features",
                "noise_analysis",
                "region",
                "category",
                "class_name",
                "subclass",
            ).filter(audio_features__isnull=False)

            total = qs.count()
            actual_rows = min(max_rows, total)

            rows_data = []
            for ds in qs[:actual_rows]:
                row = {}
                af = ds.audio_features
                if af:
                    for f in features:
                        row[f] = getattr(af, f, None)
                # Label
                if label_column == "region":
                    row["label"] = ds.region.name if ds.region else None
                elif label_column == "category":
                    row["label"] = ds.category.name if ds.category else None
                elif label_column == "class":
                    row["label"] = ds.class_name.name if ds.class_name else None
                elif label_column == "subclass":
                    row["label"] = ds.subclass.name if ds.subclass else None
                else:
                    row["label"] = ds.category.name if ds.category else None
                rows_data.append(row)

            # Build CSV
            all_columns = features + ["label"]
            csv_lines = [",".join(all_columns)]
            for row in rows_data:
                csv_lines.append(
                    ",".join(
                        str(row.get(c, "")) if row.get(c) is not None else ""
                        for c in all_columns
                    )
                )
            csv_content = "\n".join(csv_lines)

            # Count rows with non-null labels
            labeled_rows = sum(1 for r in rows_data if r.get("label") is not None)

            return {
                "analysis_type": "ml_export_features",
                "format": format,
                "columns": all_columns,
                "feature_count": len(features),
                "total_rows": total,
                "exported_rows": actual_rows,
                "labeled_rows": labeled_rows,
                "csv_data": csv_content,
                "rows": rows_data,
                "columns_meta": all_columns,
                "total_count": actual_rows,
            }
        except Exception as e:
            logger.error(f"Error in ml_export_features: {e}")
            return {"error": get_user_friendly_error(str(e), "exporting features")}


class MLFeatureImportanceInput(BaseModel):
    """Input for ML feature importance ranking."""

    label_column: Literal["category", "region", "class", "subclass"] = Field(
        default="category", description="Label column to measure importance against"
    )
    sample_size: int = Field(
        default=5000,
        ge=100,
        le=50000,
        description="Max rows to sample (capped for performance)",
    )
    seed: int = Field(default=42, description="Random seed for sampling")


class MLFeatureImportanceTool(BaseTool):
    name: str = "ml_feature_importance"
    description: str = (
        "Rank audio features by how well they predict a label. Uses sklearn's "
        "mutual_info_classif (k-NN estimator, n_neighbors=3) as primary score "
        "and ANOVA F-value as secondary cross-check. Returns ranked features "
        "with caveats about estimator behavior on small samples."
    )
    args_schema: Optional[type[BaseModel]] = MLFeatureImportanceInput

    def _run(
        self,
        label_column: str = "category",
        sample_size: int = 5000,
        seed: int = 42,
        **kwargs,
    ) -> Dict[str, Any]:
        try:
            import numpy as np
            from sklearn.feature_selection import mutual_info_classif, f_classif
        except ImportError:
            return {
                "error": "Feature importance requires scikit-learn. Please install scikit-learn.",
                "skip_visualization": True,
            }

        try:
            field_map = {
                "category": "category__name",
                "region": "region__name",
                "class": "class_name__name",
                "subclass": "subclass__name",
            }
            label_field = field_map.get(label_column, "category__name")

            feature_cols = [
                "rms_energy",
                "zero_crossing_rate",
                "spectral_centroid",
                "spectral_bandwidth",
                "spectral_rolloff",
                "spectral_flatness",
                "duration",
            ]

            qs = (
                NoiseDataset.objects.select_related(
                    "audio_features", "category", "region", "class_name", "subclass"
                )
                .filter(audio_features__isnull=False)
                .exclude(**{f"{label_field}__isnull": True})
            )

            total = qs.count()
            if total < 10:
                return {
                    "error": "insufficient_data",
                    "reason": f"Only {total} labeled rows found. Minimum is 10.",
                    "rows_available": total,
                    "skip_visualization": True,
                }

            # Stratified sample
            actual_sample = min(sample_size, total)
            rows = []
            labels = []
            for ds in qs[:actual_sample]:
                af = ds.audio_features
                if af:
                    row = [getattr(af, c, None) for c in feature_cols]
                    if all(v is not None for v in row):
                        rows.append(row)
                        if label_column == "region":
                            labels.append(ds.region.name if ds.region else None)
                        elif label_column == "class":
                            labels.append(ds.class_name.name if ds.class_name else None)
                        elif label_column == "subclass":
                            labels.append(ds.subclass.name if ds.subclass else None)
                        else:
                            labels.append(ds.category.name if ds.category else None)

            X = np.array(rows, dtype=float)
            y = np.array(labels)

            if len(X) < 10 or len(np.unique(y)) < 2:
                return {
                    "error": "insufficient_data",
                    "reason": f"Only {len(X)} valid rows with {len(np.unique(y))} unique labels.",
                    "rows_available": len(X),
                    "skip_visualization": True,
                }

            # Mutual information (k-NN estimator, n_neighbors=3)
            mi_scores = mutual_info_classif(X, y, n_neighbors=3, random_state=seed)

            # ANOVA F-value
            f_scores, f_pvalues = f_classif(X, y)

            # Check per-class sample sizes for caveats
            unique_labels, counts = np.unique(y, return_counts=True)
            small_classes = [
                str(label) for label, count in zip(unique_labels, counts) if count < 500
            ]

            # Rank features
            ranked = []
            for i, col in enumerate(feature_cols):
                ranked.append(
                    {
                        "feature": col,
                        "mi_score": round(float(mi_scores[i]), 4),
                        "f_score": round(float(f_scores[i]), 4),
                        "f_pvalue": round(float(f_pvalues[i]), 6),
                        "rank_by_mi": 0,  # filled below
                    }
                )

            ranked.sort(key=lambda x: x["mi_score"], reverse=True)
            for idx, item in enumerate(ranked):
                item["rank_by_mi"] = idx + 1

            return {
                "analysis_type": "ml_feature_importance",
                "label_column": label_column,
                "n_features": len(feature_cols),
                "n_samples": len(X),
                "total_available": total,
                "n_classes": len(unique_labels),
                "estimator": "mutual_info_classif (k-NN, n_neighbors=3)",
                "small_sample_classes": small_classes,
                "features": ranked,
                "warnings": (
                    [
                        f"Classes with <500 samples (MI estimates may be biased upward): {', '.join(small_classes)}"
                    ]
                    if small_classes
                    else []
                ),
            }
        except Exception as e:
            logger.error(f"Error in ml_feature_importance: {e}")
            return {
                "error": get_user_friendly_error(str(e), "computing feature importance")
            }


class ListMLSchemaInput(BaseModel):
    """Input for listing ML schema columns (no parameters needed)."""

    pass


class ListMLSchemaTool(BaseTool):
    name: str = "list_ml_schema"
    description: str = (
        "Returns available label columns, feature columns, metadata columns, "
        "and current row counts for ML analysis. Call this before any ML tool "
        "that references columns by name to verify which columns exist."
    )
    args_schema: Optional[type[BaseModel]] = ListMLSchemaInput

    def _run(self, **kwargs) -> Dict[str, Any]:
        try:
            return {
                "analysis_type": "ml_schema",
                "label_columns": ["category", "class", "subclass", "region"],
                "feature_columns": [
                    "rms_energy",
                    "zero_crossing_rate",
                    "spectral_centroid",
                    "spectral_bandwidth",
                    "spectral_rolloff",
                    "spectral_flatness",
                    "duration",
                    "harmonic_ratio",
                    "percussive_ratio",
                ],
                "noise_analysis_columns": [
                    "mean_db",
                    "max_db",
                    "min_db",
                    "std_db",
                    "dominant_frequency",
                    "frequency_range",
                    "event_count",
                    "peak_count",
                ],
                "metadata_columns": [
                    "name",
                    "collector",
                    "recording_date",
                    "recording_device",
                    "microphone_type",
                    "time_of_day",
                    "community",
                ],
                "row_counts": {
                    "noise_dataset": NoiseDataset.objects.count(),
                    "audio_feature": AudioFeature.objects.count(),
                    "noise_analysis": NoiseAnalysis.objects.count(),
                },
                "skip_visualization": True,
            }
        except Exception as e:
            logger.error(f"Error in list_ml_schema: {e}")
            return {"error": str(e), "skip_visualization": True}


def get_agent_tools(mode: str = "analysis"):
    """Get agent tools with lazy initialization.

    Both modes receive ALL tools. Mode only controls the system prompt
    and dashboard finalizer — the tool list is unified to avoid drift.
    The mode parameter is kept for backward compatibility but ignored.
    """
    global _all_tools_cache

    if _all_tools_cache is None:
        # P4-5: the full advanced ML toolset must be registered on the HAPPY
        # path. Previously the success branch registered only a subset and the
        # complete ML toolset existed only in the except/fallback, so in normal
        # operation (when WebFetchTool imports fine) those advanced ML tools
        # were never reachable. The advanced ML tools do not depend on
        # WebFetchTool, so they belong here unconditionally; only WebFetchTool
        # (the import that can fail) is added inside the guarded section and is
        # the sole tool dropped by the fallback.
        _all_tools_cache = [
            # Data tools (both modes)
            NoiseDatasetSearchTool(),
            DataAnalysisTool(),
            AudioAnalysisTool(),
            AudioFeatureSearchTool(),
            NoiseDetailTool(),
            # ML tools (full advanced toolset)
            MLDatasetProfileTool(),
            MLFeatureStatsTool(),
            ListMLSchemaTool(),
            MLClassBalanceTool(),
            MLTrainTestSplitTool(),
            MLCorrelationMatrixTool(),
            MLStatisticalTestTool(),
            MLExportFeaturesTool(),
            MLFeatureImportanceTool(),
        ]
        try:
            from chatbot.services.web_fetch_tool import WebFetchTool

            # Insert the web-fetch tool next to the other data tools.
            _all_tools_cache.insert(5, WebFetchTool())
        except Exception as e:
            logger.warning(f"WebFetchTool unavailable, continuing without it: {e}")
    return _all_tools_cache


# For backward compatibility - use property to make it truly lazy
class LazyAgentTools:
    def __getitem__(self, index):
        return get_agent_tools()[index]

    def __iter__(self):
        return iter(get_agent_tools())

    def __len__(self):
        return len(get_agent_tools())


AGENT_TOOLS = LazyAgentTools()


def get_tool_by_name(tool_name: str) -> Optional[BaseTool]:
    """Get a tool by its name from the unified tool list."""
    for tool in get_agent_tools():
        if tool.name == tool_name:
            return tool
    return None


def get_all_tool_schemas() -> List[Dict[str, Any]]:
    """Get all tool schemas for LLM binding"""
    schemas = []
    for tool in get_agent_tools():
        schemas.append(
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.args_schema.schema() if tool.args_schema else {},
            }
        )
    return schemas
