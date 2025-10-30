from pydantic import BaseModel, Field, model_validator
from typing import Dict, List, Any, Literal, Optional
from typing import Optional, Dict, Any
from pydantic import BaseModel
from django.db.models import Q
import uuid
import logging
import os
from langchain_core.tools import BaseTool
from django.utils import timezone
from datetime import timedelta
from data_insights.models import QueryCacheModel
from langchain_openai import ChatOpenAI
from django.conf import settings
from langchain_core.tools.base import ArgsSchema
from langchain_core.messages import HumanMessage
from .sql_agent import TextToSQLAgent

logger = logging.getLogger(__name__)

# Performance optimization settings
QUERY_TIMEOUT = 30  # seconds
MAX_RESULTS_LIMIT = 1000
DEFAULT_RESULTS_LIMIT = 100

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
    "collector__username": True,
    "collector__first_name": True,
    "collector__last_name": True,
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


AI_CONFIG = getattr(settings, "AI_INSIGHT", {})
DB_CONFIG = AI_CONFIG.get("DATABASE", {})
AGENT_CONFIG = AI_CONFIG.get("AGENT", {})
SECURITY_CONFIG = AI_CONFIG.get("SECURITY", {})

DB_URI = (
    f"postgresql://{DB_CONFIG.get('USER', 'admin')}:"
    f"{DB_CONFIG.get('PASSWORD', 'localhost')}@"
    f"{DB_CONFIG.get('HOST', 'db')}:"
    f"{DB_CONFIG.get('PORT', 5432)}/"
    f"{DB_CONFIG.get('NAME', 'database')}"
)
from data.models import (
    NoiseDataset,
    AudioFeature,
    NoiseAnalysis,
    VisualizationPreset,
    BulkAudioUpload,
    BulkReprocessingTask,
)


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
            queryset = NoiseDataset.objects.all()

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
                    created_by_id=1,  # TODO: pull from context/session
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
                    "message": f'Found {total_count} noise datasets. Use query_id "{query_id}" for bulk operations.',
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
                    "total_count": total_count,
                    "returned_count": len(result_data),
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
            queryset = AudioFeature.objects.all()

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
                    created_by_id=1,  # TODO: replace with real user context
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

    def _run(
        self,
        query: str,
        analysis_type: Optional[str] = None,
        group_by: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        try:
            from django.db import models
            from django.db.models import Avg, Max, Min, Count, StdDev, Sum
            from django.db.models.functions import TruncMonth, TruncDate

            # Determine analysis type from query if not specified
            query_lower = query.lower()

            if not analysis_type:
                # Prioritize statistical analysis for distribution queries
                if any(
                    word in query_lower
                    for word in [
                        "distribution",
                        "statistical",
                        "quartile",
                        "outlier",
                        "spread",
                        "range",
                    ]
                ):
                    analysis_type = "statistical"
                elif any(
                    word in query_lower
                    for word in ["trend", "over time", "month", "date", "timeline"]
                ):
                    analysis_type = "temporal"
                elif any(
                    word in query_lower
                    for word in ["correlation", "relationship", "vs", "against"]
                ):
                    analysis_type = "correlation"
                elif any(
                    word in query_lower
                    for word in ["cumulative", "total over", "area under"]
                ):
                    analysis_type = "temporal"  # Cumulative is temporal
                elif any(
                    word in query_lower
                    for word in ["energy", "rms", "decibel", "db", "amplitude"]
                ):
                    analysis_type = "energy"
                elif any(
                    word in query_lower
                    for word in [
                        "spectral",
                        "centroid",
                        "bandwidth",
                        "rolloff",
                        "flatness",
                    ]
                ):
                    analysis_type = "spectral"
                elif any(
                    word in query_lower
                    for word in ["frequency", "dominant", "hz", "crossing"]
                ):
                    analysis_type = "frequency"
                else:
                    analysis_type = "overview"

            # Determine grouping from query
            if not group_by:
                if "region" in query_lower:
                    group_by = "region"
                elif "category" in query_lower:
                    group_by = "category"
                elif "community" in query_lower:
                    group_by = "community"
                elif "microphone" in query_lower or "device" in query_lower:
                    group_by = "microphone_type"
                elif "time" in query_lower and "day" in query_lower:
                    group_by = "time_of_day"

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

    def _statistical_analysis(self, queryset, group_by, query):
        """Statistical distribution analysis with actual data for box plots"""
        try:
            query_lower = query.lower()

            # For distribution queries, provide actual data values for box plots
            if "distribution" in query_lower and group_by:
                if group_by == "category":
                    # Get actual decibel values grouped by category for box plot
                    categories = queryset.values_list(
                        "category__name", flat=True
                    ).distinct()
                    distribution_data = {}

                    for category in categories:
                        if category:  # Skip null categories
                            decibel_values = list(
                                queryset.filter(
                                    category__name=category,
                                    noise_analysis__mean_db__isnull=False,
                                ).values_list("noise_analysis__mean_db", flat=True)
                            )

                            if decibel_values:  # Only include categories with data
                                distribution_data[category] = {
                                    "decibel_values": decibel_values,
                                    "count": len(decibel_values),
                                    "avg": sum(decibel_values) / len(decibel_values),
                                    "max": max(decibel_values),
                                    "min": min(decibel_values),
                                }

                    return {
                        "analysis_type": "statistical_distribution",
                        "grouped_by": group_by,
                        "distribution_data": distribution_data,
                        "categories": list(distribution_data.keys()),
                        "query": query,
                        "summary": f"Decibel level distribution across {len(distribution_data)} categories with actual values for box plot visualization",
                    }

                elif group_by == "region":
                    # Get actual decibel values grouped by region for box plot
                    regions = queryset.values_list("region__name", flat=True).distinct()
                    distribution_data = {}

                    for region in regions:
                        if region:  # Skip null regions
                            decibel_values = list(
                                queryset.filter(
                                    region__name=region,
                                    noise_analysis__mean_db__isnull=False,
                                ).values_list("noise_analysis__mean_db", flat=True)
                            )

                            if decibel_values:  # Only include regions with data
                                distribution_data[region] = {
                                    "decibel_values": decibel_values,
                                    "count": len(decibel_values),
                                    "avg": sum(decibel_values) / len(decibel_values),
                                    "max": max(decibel_values),
                                    "min": min(decibel_values),
                                }

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
            # Monthly trends
            monthly_trends = list(
                queryset.annotate(month=TruncMonth("recording_date"))
                .values("month")
                .annotate(
                    dataset_count=Count("id"),
                    avg_decibel=Avg("noise_analysis__mean_db"),
                    avg_energy=Avg("audio_features__rms_energy"),
                    cumulative_energy=Sum("audio_features__rms_energy"),
                )
                .order_by("month")
            )

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
                    cumulative_energy=Sum("audio_features__rms_energy"),
                )
                .order_by("date")
            )

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

            # Include collector (uploader/owner)
            if include_collector and hasattr(dataset, "collector"):
                dataset_data["collector"] = {
                    "id": dataset.collector.id,
                    "username": getattr(dataset.collector, "username", None),
                    "email": getattr(dataset.collector, "email", None),
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


llm = ChatOpenAI(
    model=AGENT_CONFIG.get("MODEL", "gpt-4"), api_key=os.getenv("OPENAI_API_KEY")
)

allowed_tables = SECURITY_CONFIG.get("DEFAULT_ALLOWED_TABLES", [])


class DataAnalysisInput(BaseModel):
    query: str = Field(description="The natural language query to analyst")


class DataAnalysisTool(BaseTool):
    name: str = "data_analysis"
    description: str = """Use this tool to analyze data based on a natural language query.
    It can be used to analyze data from a database or any other source."""
    agent: Any = None
    top_k: int = 10
    args_schema: Optional[type[BaseModel]] = DataAnalysisInput

    @model_validator(mode="before")
    def add_agent(cls, data: Dict[str, Any]):
        """Inject a TextToSQL agent before initialization"""
        from .prompt import SQL_SYSTEM_TEMPLATE

        top_k = data.get("top_k", 10)
        data["agent"] = TextToSQLAgent(
            llm=llm,
            system_prompt=SQL_SYSTEM_TEMPLATE,
            include_tables=allowed_tables,
            top_k=top_k,
            ai_answer=False,
        ).compile_workflow()
        return data

    def _run(self, query: str, **kwargs) -> Dict[str, Any]:
        try:
            response = self.agent.invoke({"messages": [HumanMessage(content=query)]})

            # Safely extract message content
            if response and "messages" in response and response["messages"]:
                last_message = response["messages"][-1]
                # Handle different message types safely
                if hasattr(last_message, "content"):
                    msg = str(last_message.content) if last_message.content else ""
                else:
                    msg = str(last_message) if last_message else ""
            else:
                msg = "No response received"

            if "no results found" in msg.lower():
                return {"message": "No results found"}
            if "error" in msg.lower():
                return {"message": "Error in data analysis tool"}

            return {"message": msg}

        except Exception as e:
            # Ensure error message is JSON serializable
            try:
                error_str = str(e) if e else "Unknown error occurred"
            except Exception:
                error_str = "Error occurred but could not be converted to string"
            logger.error(f"Error in data analysis tool: {error_str}")
            return {"message": "Error in data analysis tool"}


class VisualizationAnalysisInput(BaseModel):
    """Input for visualization analysis"""

    query: str = Field(description="The user's query about data visualization")
    data_summary: Optional[str] = Field(
        default=None, description="Summary of the data to be visualized"
    )


class VisualizationAnalysisTool(BaseTool):
    """Tool for analyzing data and recommending the best visualization type"""

    name: str = "visualization_analysis"
    description: str = """
    Analyzes audio data and recommends the best visualization type for the given audio data and query.
    Supports: pie chart, bar chart, line chart, heatmap, scatter plot, box plot, area chart.
    Specializes in audio-specific visualizations like frequency analysis, decibel trends, and spectral characteristics.
    Returns both the recommended chart type and a template for creating the visualization.
    """
    args_schema: Optional[type[BaseModel]] = VisualizationAnalysisInput

    def _run(
        self, query: str, data_summary: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        try:
            # First, analyze the query characteristics
            query_analysis = self._analyze_query_characteristics(query, data_summary)

            # Create enhanced analysis prompt for the LLM
            analysis_prompt = f"""
            You are an expert data visualization analyst specializing in audio data. Your task is to recommend the BEST chart type for this specific audio data query.

            QUERY: "{query}"
            DATA SUMMARY: {data_summary or "No specific data provided"}

            QUERY ANALYSIS:
            - Query Type: {query_analysis['query_type']}
            - Data Dimensions: {query_analysis['dimensions']}
            - Temporal Aspect: {query_analysis['temporal']}
            - Comparison Needed: {query_analysis['comparison']}
            - Statistical Focus: {query_analysis['statistical']}

            CHART SELECTION RULES (FOLLOW STRICTLY):

            🥧 PIE CHART - Use ONLY when:
            - Query asks for proportions, percentages, or parts of a whole
            - Keywords: "distribution of", "percentage of", "proportion", "share"
            - Data shows how categories make up 100% of something
            - Example: "What percentage of audio files are in each category?"

            📊 BAR CHART - Use when:
            - Comparing discrete values across categories
            - Keywords: "compare", "which has higher", "levels across", "by region/category"
            - Data has distinct categories with values to compare
            - Example: "Compare decibel levels across different regions"

            📈 LINE CHART - Use when:
            - Showing trends, changes, or progression over time
            - Keywords: "over time", "trends", "changes", "timeline", "progression"
            - Data has temporal or sequential component
            - Example: "Show me audio recording trends over time"

            🔥 HEATMAP - Use when:
            - Showing correlations, patterns, or 2D relationships
            - Keywords: "correlation", "pattern", "relationship between", "matrix"
            - Data has two dimensions that interact
            - Example: "Show correlation between frequency and amplitude"

            🔵 SCATTER PLOT - Use when:
            - Showing relationship between two continuous variables
            - Keywords: "relationship between X and Y", "vs", "against", "correlation"
            - Data points need to show individual relationships
            - Example: "Plot RMS energy vs spectral centroid"

            📦 BOX PLOT - Use when:
            - Showing statistical distributions, quartiles, outliers
            - Keywords: "distribution", "outliers", "quartiles", "statistical", "range"
            - Data needs to show statistical properties
            - Example: "Show distribution of decibel levels across categories"

            🏔️ AREA CHART - Use when:
            - Showing cumulative data or area under curves
            - Keywords: "cumulative", "total over time", "area under", "spectrum"
            - Data shows accumulation or stacked composition
            - Example: "Show cumulative audio energy over time"

            RESPOND WITH ONLY JSON:
            {{
                "recommended_chart": "chart_type",
                "reasoning": "Detailed explanation why this specific chart type is optimal for this audio data query",
                "data_requirements": ["field1", "field2"],
                "confidence": "high"
            }}
            """

            # Use LLM to analyze and recommend
            llm = ChatOpenAI(
                model=AGENT_CONFIG.get("MODEL", "gpt-5-nano"),
                temperature=0.1,  # Low temperature for consistent reasoning
                api_key=os.getenv("OPENAI_API_KEY"),
            )
            response = llm.invoke([HumanMessage(content=analysis_prompt)])

            # Parse the response and create chart template
            recommendation = self._parse_visualization_recommendation(
                response.content, query
            )

            # Validate and potentially override the recommendation
            final_recommendation = self._validate_recommendation(
                recommendation, query_analysis, query
            )

            chart_type = final_recommendation["recommended_chart"]
            chart_template = self._generate_chart_template(chart_type)

            return {
                "visualization_type": chart_type,
                "visualization_name": self._get_visualization_name(chart_type),
                "chart_template": chart_template,
                "recommendation": final_recommendation,
                "frontend_data": {
                    "type": chart_type,
                    "name": self._get_visualization_name(chart_type),
                    "config": chart_template["config"],
                    "data_structure": self._get_data_structure(chart_type),
                    "description": final_recommendation.get(
                        "reasoning", "Audio data visualization"
                    ),
                },
                "message": f"Recommended {self._get_visualization_name(chart_type)} for this data analysis",
            }

        except Exception as e:
            logger.error(f"Error in visualization analysis: {e}")
            chart_type = "bar_chart"
            chart_template = self._generate_chart_template(chart_type)

            return {
                "visualization_type": chart_type,
                "visualization_name": self._get_visualization_name(chart_type),
                "chart_template": chart_template,
                "recommendation": {
                    "recommended_chart": chart_type,
                    "reasoning": "Default recommendation due to analysis error",
                    "data_requirements": ["category", "value"],
                },
                "frontend_data": {
                    "type": chart_type,
                    "name": self._get_visualization_name(chart_type),
                    "config": chart_template["config"],
                    "data_structure": self._get_data_structure(chart_type),
                    "description": "Default bar chart for audio data visualization",
                },
                "message": "Error in visualization analysis, using default bar chart",
            }

    def _analyze_query_characteristics(
        self, query: str, data_summary: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze query characteristics to inform visualization choice"""
        query_lower = query.lower()

        # Determine query type
        query_type = "unknown"
        if any(
            word in query_lower
            for word in ["distribution", "percentage", "proportion", "share"]
        ):
            query_type = "distribution"
        elif any(
            word in query_lower
            for word in ["compare", "comparison", "vs", "against", "between"]
        ):
            query_type = "comparison"
        elif any(
            word in query_lower for word in ["trend", "over time", "timeline", "change"]
        ):
            query_type = "temporal"
        elif any(
            word in query_lower for word in ["correlation", "relationship", "pattern"]
        ):
            query_type = "correlation"
        elif any(
            word in query_lower for word in ["cumulative", "total", "sum", "area under"]
        ):
            query_type = "cumulative"
        elif any(
            word in query_lower
            for word in ["outlier", "quartile", "statistical", "distribution"]
        ):
            query_type = "statistical"

        # Determine data dimensions
        dimensions = 1
        if any(
            word in query_lower
            for word in ["vs", "against", "correlation", "relationship between"]
        ):
            dimensions = 2
        elif any(
            word in query_lower
            for word in ["by region and category", "cross", "matrix"]
        ):
            dimensions = 3

        # Check for temporal aspect
        temporal = any(
            word in query_lower
            for word in ["time", "date", "month", "day", "timeline", "trend"]
        )

        # Check for comparison needs
        comparison = any(
            word in query_lower
            for word in ["compare", "vs", "higher", "lower", "best", "worst"]
        )

        # Check for statistical focus
        statistical = any(
            word in query_lower
            for word in [
                "distribution",
                "outlier",
                "quartile",
                "statistical",
                "median",
                "average",
            ]
        )

        # Suggest chart type based on analysis
        suggested_type = self._suggest_chart_type(
            query_type, dimensions, temporal, comparison, statistical
        )

        return {
            "query_type": query_type,
            "dimensions": dimensions,
            "temporal": temporal,
            "comparison": comparison,
            "statistical": statistical,
            "suggested_type": suggested_type,
            "reasoning": f"Query appears to be {query_type} analysis with {dimensions} dimensions",
        }

    def _suggest_chart_type(
        self,
        query_type: str,
        dimensions: int,
        temporal: bool,
        comparison: bool,
        statistical: bool,
    ) -> str:
        """Suggest chart type based on query characteristics"""

        # Temporal data almost always needs line or area charts
        if temporal and query_type == "cumulative":
            return "area_chart"
        elif temporal:
            return "line_chart"

        # Statistical analysis usually needs box plots
        if statistical and query_type == "statistical":
            return "box_plot"

        # Correlation analysis needs scatter plots or heatmaps
        if query_type == "correlation":
            if dimensions == 2:
                return "scatter_plot"
            else:
                return "heatmap"

        # Distribution analysis - pie only for parts of whole
        if query_type == "distribution":
            return "pie_chart"

        # Comparison analysis needs bar charts
        if query_type == "comparison" or comparison:
            return "bar_chart"

        # Cumulative data needs area charts
        if query_type == "cumulative":
            return "area_chart"

        # Default to bar chart for general comparisons
        return "bar_chart"

    def _validate_recommendation(
        self, recommendation: Dict[str, Any], query_analysis: Dict[str, Any], query: str
    ) -> Dict[str, Any]:
        """Validate and potentially override LLM recommendation based on query analysis"""

        llm_chart = recommendation.get("recommended_chart", "bar_chart")
        suggested_chart = query_analysis["suggested_type"]

        # Override pie chart if it's not truly a distribution query
        if llm_chart == "pie_chart" and query_analysis["query_type"] != "distribution":
            query_lower = query.lower()
            if not any(
                word in query_lower
                for word in ["percentage", "proportion", "share", "part of"]
            ):
                logger.info(f"Overriding pie chart recommendation for query: {query}")
                return {
                    "recommended_chart": suggested_chart,
                    "reasoning": f"Changed from pie chart to {suggested_chart} because this query is about {query_analysis['query_type']}, not proportions",
                    "data_requirements": recommendation.get(
                        "data_requirements", ["category", "value"]
                    ),
                    "confidence": "high",
                    "override_reason": f"Query type '{query_analysis['query_type']}' is better suited for {suggested_chart}",
                }

        # Override if temporal data is using wrong chart type
        if query_analysis["temporal"] and llm_chart not in ["line_chart", "area_chart"]:
            logger.info(f"Overriding {llm_chart} for temporal query: {query}")
            chart_type = "area_chart" if "cumulative" in query.lower() else "line_chart"
            return {
                "recommended_chart": chart_type,
                "reasoning": f"Changed to {chart_type} because this query involves temporal data which is best shown with time-based charts",
                "data_requirements": ["time", "value"],
                "confidence": "high",
                "override_reason": "Temporal data requires time-based visualization",
            }

        # Override if correlation data is using wrong chart type
        if query_analysis["query_type"] == "correlation" and llm_chart not in [
            "scatter_plot",
            "heatmap",
        ]:
            logger.info(f"Overriding {llm_chart} for correlation query: {query}")
            chart_type = (
                "scatter_plot" if query_analysis["dimensions"] == 2 else "heatmap"
            )
            return {
                "recommended_chart": chart_type,
                "reasoning": f"Changed to {chart_type} because correlation analysis requires charts that show relationships between variables",
                "data_requirements": ["variable_x", "variable_y"],
                "confidence": "high",
                "override_reason": "Correlation data requires relationship visualization",
            }

        # If LLM recommendation is good, use it
        return recommendation

    def _parse_visualization_recommendation(
        self, llm_response: str, query: str
    ) -> Dict[str, Any]:
        """Parse LLM response and extract visualization recommendation with intelligent fallback"""
        try:
            # Try to parse JSON response from LLM first
            import json
            import re

            # Look for JSON in the response
            json_match = re.search(
                r'\{[^}]*"recommended_chart"[^}]*\}', llm_response, re.DOTALL
            )
            if json_match:
                try:
                    parsed_json = json.loads(json_match.group())
                    if "recommended_chart" in parsed_json:
                        logger.info(
                            f"Successfully parsed LLM recommendation: {parsed_json['recommended_chart']}"
                        )
                        return parsed_json
                except json.JSONDecodeError as je:
                    logger.warning(f"JSON parsing failed: {je}")

            # If JSON parsing fails, use intelligent keyword analysis
            logger.info(f"Using fallback keyword analysis for query: {query}")
            return self._intelligent_keyword_analysis(query, llm_response)

        except Exception as e:
            logger.error(f"Error parsing visualization recommendation: {e}")
            return self._intelligent_keyword_analysis(query, "")

    def _intelligent_keyword_analysis(
        self, query: str, llm_response: str = ""
    ) -> Dict[str, Any]:
        """Intelligent keyword-based chart selection with strict rules to avoid pie chart bias"""
        query_lower = query.lower()

        # 1. TEMPORAL ANALYSIS - Line/Area Charts (HIGH PRIORITY)
        if any(
            word in query_lower
            for word in [
                "over time",
                "timeline",
                "trend",
                "change",
                "month",
                "date",
                "day",
                "progression",
            ]
        ):
            if any(
                word in query_lower
                for word in ["cumulative", "total", "sum", "area under"]
            ):
                return {
                    "recommended_chart": "area_chart",
                    "reasoning": "Temporal query with cumulative aspect - area chart shows accumulation over time",
                    "data_requirements": ["time", "cumulative_value"],
                    "confidence": "high",
                }
            else:
                return {
                    "recommended_chart": "line_chart",
                    "reasoning": "Temporal query - line chart best shows trends and changes over time",
                    "data_requirements": ["time", "value"],
                    "confidence": "high",
                }

        # 2. CORRELATION/RELATIONSHIP ANALYSIS - Scatter/Heatmap (HIGH PRIORITY)
        elif any(
            word in query_lower
            for word in [
                "relationship between",
                " vs ",
                " against ",
                "correlation between",
                "plot",
            ]
        ):
            return {
                "recommended_chart": "scatter_plot",
                "reasoning": "Query asks for relationship between two specific variables - scatter plot shows individual data point correlations",
                "data_requirements": ["variable_x", "variable_y"],
                "confidence": "high",
            }
        elif any(
            word in query_lower
            for word in ["correlation", "pattern", "matrix", "spectral pattern"]
        ):
            return {
                "recommended_chart": "heatmap",
                "reasoning": "Query asks for correlations or patterns - heatmap shows complex multi-dimensional relationships",
                "data_requirements": ["dimension_x", "dimension_y", "intensity"],
                "confidence": "high",
            }

        # 3. STATISTICAL ANALYSIS - Box Plots (HIGH PRIORITY)
        elif any(
            word in query_lower
            for word in [
                "outlier",
                "quartile",
                "statistical",
                "range",
                "spread",
                "median",
            ]
        ):
            return {
                "recommended_chart": "box_plot",
                "reasoning": "Statistical analysis query - box plot shows distribution, quartiles, and outliers",
                "data_requirements": ["category", "value_distribution"],
                "confidence": "high",
            }

        # 4. CUMULATIVE ANALYSIS - Area Charts (HIGH PRIORITY)
        elif any(
            word in query_lower
            for word in [
                "cumulative",
                "total over",
                "area under",
                "spectrum analysis",
                "energy over",
            ]
        ):
            return {
                "recommended_chart": "area_chart",
                "reasoning": "Cumulative analysis query - area chart shows accumulation and total values",
                "data_requirements": ["sequence", "cumulative_value"],
                "confidence": "high",
            }

        # 5. STRICT PROPORTION ANALYSIS - Pie Charts (VERY STRICT CRITERIA)
        elif (
            any(
                word in query_lower
                for word in ["percentage", "proportion", "share", "part of"]
            )
            and any(word in query_lower for word in ["of", "by", "across"])
            and not any(
                word in query_lower
                for word in ["compare", "vs", "against", "higher", "lower"]
            )
        ):
            return {
                "recommended_chart": "pie_chart",
                "reasoning": "Query specifically asks for proportions or percentages of a whole - pie chart shows parts-to-whole relationships",
                "data_requirements": ["category", "percentage"],
                "confidence": "high",
            }

        # 6. COMPARISON ANALYSIS - Bar Charts (DEFAULT FOR MOST QUERIES)
        elif any(
            word in query_lower
            for word in [
                "compare",
                "comparison",
                "which",
                "higher",
                "lower",
                "across",
                "between",
                "levels",
            ]
        ):
            return {
                "recommended_chart": "bar_chart",
                "reasoning": "Comparison query - bar chart clearly shows differences between categories",
                "data_requirements": ["category", "value"],
                "confidence": "high",
            }

        # 7. DISTRIBUTION BY COUNT (NOT PERCENTAGE) - Bar Charts
        elif any(word in query_lower for word in ["distribution", "breakdown"]):
            return {
                "recommended_chart": "bar_chart",
                "reasoning": "Distribution query by count - bar chart shows quantities across categories (not percentages)",
                "data_requirements": ["category", "count"],
                "confidence": "medium",
            }

        # 8. DEFAULT - Bar Chart (NEVER default to pie!)
        else:
            return {
                "recommended_chart": "bar_chart",
                "reasoning": "General audio data query - bar chart provides clear comparison of values across categories",
                "data_requirements": ["category", "value"],
                "confidence": "medium",
            }

    def _generate_chart_template(self, chart_type: str) -> Dict[str, Any]:
        """Generate chart template based on chart type"""
        templates = {
            "pie_chart": {
                "type": "pie",
                "config": {
                    "data": {
                        "labels": [],
                        "datasets": [
                            {
                                "data": [],
                                "backgroundColor": [
                                    "#FF6384",
                                    "#36A2EB",
                                    "#FFCE56",
                                    "#4BC0C0",
                                    "#9966FF",
                                    "#FF9F40",
                                    "#FF6384",
                                    "#C9CBCF",
                                ],
                            }
                        ],
                    },
                    "options": {
                        "responsive": True,
                        "plugins": {
                            "legend": {"position": "bottom"},
                            "title": {"display": True, "text": "Data Distribution"},
                        },
                    },
                },
            },
            "bar_chart": {
                "type": "bar",
                "config": {
                    "data": {
                        "labels": [],
                        "datasets": [
                            {
                                "label": "Values",
                                "data": [],
                                "backgroundColor": "#36A2EB",
                                "borderColor": "#36A2EB",
                                "borderWidth": 1,
                            }
                        ],
                    },
                    "options": {
                        "responsive": True,
                        "scales": {"y": {"beginAtZero": True}},
                        "plugins": {
                            "title": {"display": True, "text": "Data Comparison"}
                        },
                    },
                },
            },
            "line_chart": {
                "type": "line",
                "config": {
                    "data": {
                        "labels": [],
                        "datasets": [
                            {
                                "label": "Trend",
                                "data": [],
                                "borderColor": "#36A2EB",
                                "backgroundColor": "rgba(54, 162, 235, 0.1)",
                                "tension": 0.1,
                            }
                        ],
                    },
                    "options": {
                        "responsive": True,
                        "scales": {"y": {"beginAtZero": True}},
                        "plugins": {
                            "title": {"display": True, "text": "Trend Analysis"}
                        },
                    },
                },
            },
            "heatmap": {
                "type": "heatmap",
                "config": {
                    "data": {
                        "datasets": [
                            {
                                "label": "Heatmap",
                                "data": [],
                                "backgroundColor": "rgba(54, 162, 235, 0.8)",
                            }
                        ]
                    },
                    "options": {
                        "responsive": True,
                        "plugins": {
                            "title": {"display": True, "text": "Data Patterns"}
                        },
                        "scales": {
                            "x": {"type": "category"},
                            "y": {"type": "category"},
                        },
                    },
                },
            },
            "scatter_plot": {
                "type": "scatter",
                "config": {
                    "data": {
                        "datasets": [
                            {
                                "label": "Data Points",
                                "data": [],
                                "backgroundColor": "#36A2EB",
                                "borderColor": "#36A2EB",
                            }
                        ]
                    },
                    "options": {
                        "responsive": True,
                        "scales": {
                            "x": {"type": "linear", "position": "bottom"},
                            "y": {"type": "linear"},
                        },
                        "plugins": {
                            "title": {"display": True, "text": "Relationship Analysis"}
                        },
                    },
                },
            },
            "box_plot": {
                "type": "boxplot",
                "config": {
                    "data": {
                        "labels": [],
                        "datasets": [
                            {
                                "label": "Distribution",
                                "data": [],
                                "backgroundColor": "rgba(54, 162, 235, 0.5)",
                                "borderColor": "#36A2EB",
                            }
                        ],
                    },
                    "options": {
                        "responsive": True,
                        "plugins": {
                            "title": {"display": True, "text": "Data Distribution"}
                        },
                    },
                },
            },
            "area_chart": {
                "type": "line",
                "config": {
                    "data": {
                        "labels": [],
                        "datasets": [
                            {
                                "label": "Cumulative",
                                "data": [],
                                "borderColor": "#36A2EB",
                                "backgroundColor": "rgba(54, 162, 235, 0.3)",
                                "fill": True,
                                "tension": 0.1,
                            }
                        ],
                    },
                    "options": {
                        "responsive": True,
                        "scales": {"y": {"beginAtZero": True}},
                        "plugins": {
                            "title": {"display": True, "text": "Cumulative Analysis"}
                        },
                    },
                },
            },
        }

        return templates.get(chart_type, templates["bar_chart"])

    def _get_visualization_name(self, chart_type: str) -> str:
        """Get human-readable name for chart type"""
        names = {
            "pie_chart": "Pie Chart",
            "bar_chart": "Bar Chart",
            "line_chart": "Line Chart",
            "heatmap": "Heatmap",
            "scatter_plot": "Scatter Plot",
            "box_plot": "Box Plot",
            "area_chart": "Area Chart",
        }
        return names.get(chart_type, "Bar Chart")

    def _get_data_structure(self, chart_type: str) -> Dict[str, Any]:
        """Get expected data structure for each chart type"""
        structures = {
            "pie_chart": {
                "labels": "Array of category names",
                "data": "Array of values corresponding to labels",
                "description": "For showing proportions/percentages of audio categories",
            },
            "bar_chart": {
                "labels": "Array of category names",
                "data": "Array of values for each category",
                "description": "For comparing audio metrics across categories",
            },
            "line_chart": {
                "labels": "Array of time points or frequency values",
                "data": "Array of values over time/frequency",
                "description": "For showing audio trends over time or frequency analysis",
            },
            "heatmap": {
                "x_labels": "Array of x-axis categories",
                "y_labels": "Array of y-axis categories",
                "data": "2D array of intensity values",
                "description": "For showing audio correlations and patterns",
            },
            "scatter_plot": {
                "x_data": "Array of x-axis values",
                "y_data": "Array of y-axis values",
                "description": "For showing relationships between audio variables",
            },
            "box_plot": {
                "labels": "Array of category names",
                "data": "Array of arrays containing numerical values for each category",
                "description": "For showing distribution of audio metrics",
            },
            "area_chart": {
                "labels": "Array of time points or frequency values",
                "data": "Array of cumulative values",
                "description": "For showing cumulative audio energy or spectrum analysis",
            },
        }
        return structures.get(chart_type, structures["bar_chart"])


# Lazy initialization to avoid database connection at import time
_agent_tools = None


def get_agent_tools():
    """Get agent tools with lazy initialization"""
    global _agent_tools
    if _agent_tools is None:
        try:
            _agent_tools = [
                NoiseDatasetSearchTool(),
                DataAnalysisTool(),
                VisualizationAnalysisTool(),
                AudioFeatureSearchTool(),
                AudioAnalysisTool(),
                NoiseDetailTool(),
            ]
        except Exception as e:
            logger.warning(f"Failed to initialize some tools: {e}")
            # Fallback to tools that don't require database connection
            _agent_tools = [
                NoiseDatasetSearchTool(),
                VisualizationAnalysisTool(),
                AudioFeatureSearchTool(),
                NoiseDetailTool(),
            ]
    return _agent_tools


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
    """Get a tool by its name"""
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
