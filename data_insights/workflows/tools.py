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

def get_user_friendly_error(error_msg: str, context: str = "") -> str:
    """Convert technical errors to user-friendly messages"""
    error_lower = str(error_msg).lower()
    
    if "timeout" in error_lower:
        return "The query is taking too long to process. Please try a more specific search or contact support."
    elif "connection" in error_lower or "database" in error_lower:
        return "Unable to connect to the audio database. Please check your connection and try again."
    elif "permission" in error_lower or "access" in error_lower:
        return "You don't have permission to access this audio data. Please contact your administrator."
    elif "not found" in error_lower or "does not exist" in error_lower:
        return f"The requested audio data was not found. Please check your search criteria."
    elif "invalid" in error_lower or "malformed" in error_lower:
        return "Invalid search parameters provided. Please check your input and try again."
    elif "memory" in error_lower or "limit" in error_lower:
        return "The request is too large to process. Please try filtering your search to fewer results."
    else:
        return f"An unexpected error occurred while processing your audio data request. {context}"

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
                queryset = queryset.filter(recording_date__lte=filter_criteria["date_to"])

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
                        "community": dataset.community.name if dataset.community else None,
                        "category": dataset.category.name if dataset.category else None,
                        "recording_date": dataset.recording_date,
                        "recording_device": dataset.recording_device,
                    }
                    
                    # Add noise analysis data if available
                    if hasattr(dataset, 'noise_analysis') and dataset.noise_analysis:
                        analysis = dataset.noise_analysis
                        dataset_data.update({
                            "mean_db": analysis.mean_db,
                            "max_db": analysis.max_db,
                            "min_db": analysis.min_db,
                            "dominant_frequency": analysis.dominant_frequency,
                            "event_count": analysis.event_count,
                        })

                    if include_features and hasattr(dataset, "features"):
                        features = dataset.features
                        dataset_data["features"] = {
                            "rms": getattr(features, "rms", None),
                            "spectral_centroid": getattr(
                                features, "spectral_centroid", None
                            ),
                            "zero_crossing_rate": getattr(
                                features, "zero_crossing_rate", None
                            ),
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
                queryset = queryset.filter(rms_energy__gte=filter_criteria["rms_energy"])

            if "zero_crossing_rate" in filter_criteria:
                queryset = queryset.filter(zero_crossing_rate__gte=filter_criteria["zero_crossing_rate"])

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
                        "id", "rms", "zcr", "spectral_centroid", "spectral_bandwidth"
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
                            "dataset_name": feature.noise_dataset.name if feature.noise_dataset else None,
                            "region": feature.noise_dataset.region.name if feature.noise_dataset and feature.noise_dataset.region else None,
                        }
                    )

                return {
                    "audio_features": result_data,
                    "total_count": total_count,
                    "returned_count": len(result_data),
                }

        except Exception as e:
            logger.error(f"Error in audio feature search: {str(e)}")
            return {"error": f"Audio feature search failed: {str(e)}"}


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
        **kwargs
    ) -> Dict[str, Any]:
        try:
            from django.db import models
            from django.db.models import Avg, Max, Min, Count, StdDev, Sum
            from django.db.models.functions import TruncMonth, TruncDate
            
            # Determine analysis type from query if not specified
            query_lower = query.lower()
            
            if not analysis_type:
                if any(word in query_lower for word in ['energy', 'rms', 'decibel', 'db', 'amplitude', 'cumulative']):
                    analysis_type = 'energy'
                elif any(word in query_lower for word in ['spectral', 'centroid', 'bandwidth', 'rolloff', 'flatness']):
                    analysis_type = 'spectral'
                elif any(word in query_lower for word in ['frequency', 'dominant', 'hz', 'crossing']):
                    analysis_type = 'frequency'
                elif any(word in query_lower for word in ['correlation', 'relationship', 'vs', 'against']):
                    analysis_type = 'correlation'
                elif any(word in query_lower for word in ['distribution', 'statistical', 'quartile', 'outlier']):
                    analysis_type = 'statistical'
                elif any(word in query_lower for word in ['trend', 'over time', 'month', 'date', 'timeline']):
                    analysis_type = 'temporal'
                else:
                    analysis_type = 'overview'
            
            # Determine grouping from query
            if not group_by:
                if 'region' in query_lower:
                    group_by = 'region'
                elif 'category' in query_lower:
                    group_by = 'category'
                elif 'community' in query_lower:
                    group_by = 'community'
                elif 'microphone' in query_lower or 'device' in query_lower:
                    group_by = 'microphone_type'
                elif 'time' in query_lower and 'day' in query_lower:
                    group_by = 'time_of_day'
            
            # Get base queryset with all related data
            queryset = NoiseDataset.objects.select_related(
                'audio_features', 'noise_analysis', 'region', 'category', 
                'community', 'microphone_type', 'time_of_day'
            ).filter(
                audio_features__isnull=False,
                noise_analysis__isnull=False
            )
            
            # Perform analysis based on type
            if analysis_type == 'energy':
                return self._energy_analysis(queryset, group_by, query)
            elif analysis_type == 'spectral':
                return self._spectral_analysis(queryset, group_by, query)
            elif analysis_type == 'frequency':
                return self._frequency_analysis(queryset, group_by, query)
            elif analysis_type == 'correlation':
                return self._correlation_analysis(queryset, query)
            elif analysis_type == 'statistical':
                return self._statistical_analysis(queryset, group_by, query)
            elif analysis_type == 'temporal':
                return self._temporal_analysis(queryset, query)
            else:
                return self._overview_analysis(queryset, query)
                
        except Exception as e:
            logger.error(f"Error in audio analysis: {e}")
            return {"error": f"Audio analysis failed: {str(e)}"}
    
    def _energy_analysis(self, queryset, group_by, query):
        """Analyze energy-related metrics"""
        try:
            if group_by == 'region':
                results = list(queryset.values('region__name').annotate(
                    dataset_count=Count('id'),
                    avg_rms_energy=Avg('audio_features__rms_energy'),
                    max_rms_energy=Max('audio_features__rms_energy'),
                    min_rms_energy=Min('audio_features__rms_energy'),
                    avg_decibel=Avg('noise_analysis__mean_db'),
                    max_decibel=Max('noise_analysis__max_db'),
                    min_decibel=Min('noise_analysis__min_db'),
                    cumulative_energy=Sum('audio_features__rms_energy')
                ).order_by('-avg_rms_energy'))
                
            elif group_by == 'category':
                results = list(queryset.values('category__name').annotate(
                    dataset_count=Count('id'),
                    avg_rms_energy=Avg('audio_features__rms_energy'),
                    max_rms_energy=Max('audio_features__rms_energy'),
                    avg_decibel=Avg('noise_analysis__mean_db'),
                    max_decibel=Max('noise_analysis__max_db'),
                    cumulative_energy=Sum('audio_features__rms_energy')
                ).order_by('-avg_decibel'))
                
            elif group_by == 'microphone_type':
                results = list(queryset.values('microphone_type__name').annotate(
                    dataset_count=Count('id'),
                    avg_rms_energy=Avg('audio_features__rms_energy'),
                    avg_decibel=Avg('noise_analysis__mean_db'),
                    max_decibel=Max('noise_analysis__max_db')
                ).order_by('-avg_decibel'))
                
            else:
                # Overall energy statistics
                stats = queryset.aggregate(
                    total_datasets=Count('id'),
                    avg_rms_energy=Avg('audio_features__rms_energy'),
                    max_rms_energy=Max('audio_features__rms_energy'),
                    min_rms_energy=Min('audio_features__rms_energy'),
                    std_rms_energy=StdDev('audio_features__rms_energy'),
                    avg_decibel=Avg('noise_analysis__mean_db'),
                    max_decibel=Max('noise_analysis__max_db'),
                    min_decibel=Min('noise_analysis__min_db'),
                    cumulative_energy=Sum('audio_features__rms_energy'),
                    total_duration=Sum('audio_features__duration')
                )
                results = [stats]
            
            return {
                "analysis_type": "energy_analysis",
                "grouped_by": group_by,
                "results": results,
                "query": query,
                "summary": f"Energy analysis shows audio power levels and decibel measurements"
            }
            
        except Exception as e:
            return {"error": f"Energy analysis failed: {str(e)}"}
    
    def _spectral_analysis(self, queryset, group_by, query):
        """Analyze spectral characteristics"""
        try:
            if group_by == 'region':
                results = list(queryset.values('region__name').annotate(
                    dataset_count=Count('id'),
                    avg_spectral_centroid=Avg('audio_features__spectral_centroid'),
                    avg_spectral_bandwidth=Avg('audio_features__spectral_bandwidth'),
                    avg_spectral_rolloff=Avg('audio_features__spectral_rolloff'),
                    avg_spectral_flatness=Avg('audio_features__spectral_flatness')
                ).order_by('-avg_spectral_centroid'))
                
            elif group_by == 'category':
                results = list(queryset.values('category__name').annotate(
                    dataset_count=Count('id'),
                    avg_spectral_centroid=Avg('audio_features__spectral_centroid'),
                    avg_spectral_bandwidth=Avg('audio_features__spectral_bandwidth'),
                    avg_spectral_rolloff=Avg('audio_features__spectral_rolloff')
                ).order_by('-avg_spectral_centroid'))
                
            else:
                stats = queryset.aggregate(
                    total_datasets=Count('id'),
                    avg_spectral_centroid=Avg('audio_features__spectral_centroid'),
                    max_spectral_centroid=Max('audio_features__spectral_centroid'),
                    min_spectral_centroid=Min('audio_features__spectral_centroid'),
                    avg_spectral_bandwidth=Avg('audio_features__spectral_bandwidth'),
                    avg_spectral_rolloff=Avg('audio_features__spectral_rolloff'),
                    avg_spectral_flatness=Avg('audio_features__spectral_flatness')
                )
                results = [stats]
            
            return {
                "analysis_type": "spectral_analysis",
                "grouped_by": group_by,
                "results": results,
                "query": query,
                "summary": f"Spectral analysis shows frequency distribution characteristics"
            }
            
        except Exception as e:
            return {"error": f"Spectral analysis failed: {str(e)}"}
    
    def _frequency_analysis(self, queryset, group_by, query):
        """Analyze frequency characteristics"""
        try:
            if group_by == 'region':
                results = list(queryset.values('region__name').annotate(
                    dataset_count=Count('id'),
                    avg_dominant_frequency=Avg('noise_analysis__dominant_frequency'),
                    max_dominant_frequency=Max('noise_analysis__dominant_frequency'),
                    avg_zero_crossing_rate=Avg('audio_features__zero_crossing_rate')
                ).order_by('-avg_dominant_frequency'))
                
            elif group_by == 'category':
                results = list(queryset.values('category__name').annotate(
                    dataset_count=Count('id'),
                    avg_dominant_frequency=Avg('noise_analysis__dominant_frequency'),
                    avg_zero_crossing_rate=Avg('audio_features__zero_crossing_rate')
                ).order_by('-avg_dominant_frequency'))
                
            else:
                stats = queryset.aggregate(
                    total_datasets=Count('id'),
                    avg_dominant_frequency=Avg('noise_analysis__dominant_frequency'),
                    max_dominant_frequency=Max('noise_analysis__dominant_frequency'),
                    min_dominant_frequency=Min('noise_analysis__dominant_frequency'),
                    avg_zero_crossing_rate=Avg('audio_features__zero_crossing_rate')
                )
                results = [stats]
            
            return {
                "analysis_type": "frequency_analysis", 
                "grouped_by": group_by,
                "results": results,
                "query": query,
                "summary": f"Frequency analysis shows dominant frequencies and zero crossing rates"
            }
            
        except Exception as e:
            return {"error": f"Frequency analysis failed: {str(e)}"}
    
    def _correlation_analysis(self, queryset, query):
        """Analyze correlations between different features"""
        try:
            # Get sample data for correlation
            data = list(queryset.values(
                'audio_features__rms_energy',
                'audio_features__spectral_centroid', 
                'audio_features__spectral_bandwidth',
                'audio_features__zero_crossing_rate',
                'noise_analysis__mean_db',
                'noise_analysis__dominant_frequency',
                'audio_features__duration'
            )[:100])  # Limit for performance
            
            # Extract specific correlations based on query
            correlations = {}
            if 'rms' in query.lower() and 'spectral' in query.lower():
                rms_data = [d['audio_features__rms_energy'] for d in data if d['audio_features__rms_energy'] is not None]
                centroid_data = [d['audio_features__spectral_centroid'] for d in data if d['audio_features__spectral_centroid'] is not None]
                
                correlations['rms_vs_spectral_centroid'] = {
                    'rms_energy_samples': rms_data[:20],
                    'spectral_centroid_samples': centroid_data[:20],
                    'sample_count': min(len(rms_data), len(centroid_data))
                }
            
            if 'frequency' in query.lower() and 'amplitude' in query.lower():
                freq_data = [d['noise_analysis__dominant_frequency'] for d in data if d['noise_analysis__dominant_frequency'] is not None]
                db_data = [d['noise_analysis__mean_db'] for d in data if d['noise_analysis__mean_db'] is not None]
                
                correlations['frequency_vs_amplitude'] = {
                    'frequency_samples': freq_data[:20],
                    'decibel_samples': db_data[:20],
                    'sample_count': min(len(freq_data), len(db_data))
                }
            
            return {
                "analysis_type": "correlation_analysis",
                "correlations": correlations,
                "total_samples": len(data),
                "query": query,
                "summary": f"Correlation analysis between audio features"
            }
            
        except Exception as e:
            return {"error": f"Correlation analysis failed: {str(e)}"}
    
    def _statistical_analysis(self, queryset, group_by, query):
        """Statistical distribution analysis"""
        try:
            if group_by == 'category':
                results = list(queryset.values('category__name').annotate(
                    dataset_count=Count('id'),
                    avg_decibel=Avg('noise_analysis__mean_db'),
                    std_decibel=StdDev('noise_analysis__mean_db'),
                    max_decibel=Max('noise_analysis__max_db'),
                    min_decibel=Min('noise_analysis__min_db'),
                    avg_rms=Avg('audio_features__rms_energy'),
                    std_rms=StdDev('audio_features__rms_energy')
                ).order_by('-avg_decibel'))
                
            elif group_by == 'region':
                results = list(queryset.values('region__name').annotate(
                    dataset_count=Count('id'),
                    avg_decibel=Avg('noise_analysis__mean_db'),
                    std_decibel=StdDev('noise_analysis__mean_db'),
                    quartile_range=Max('noise_analysis__max_db') - Min('noise_analysis__min_db'),
                    avg_rms=Avg('audio_features__rms_energy')
                ).order_by('-avg_decibel'))
                
            else:
                # Overall statistics
                stats = queryset.aggregate(
                    total_datasets=Count('id'),
                    avg_decibel=Avg('noise_analysis__mean_db'),
                    std_decibel=StdDev('noise_analysis__mean_db'),
                    max_decibel=Max('noise_analysis__max_db'),
                    min_decibel=Min('noise_analysis__min_db'),
                    avg_rms=Avg('audio_features__rms_energy'),
                    std_rms=StdDev('audio_features__rms_energy'),
                    avg_spectral_centroid=Avg('audio_features__spectral_centroid'),
                    std_spectral_centroid=StdDev('audio_features__spectral_centroid')
                )
                results = [stats]
            
            return {
                "analysis_type": "statistical_analysis",
                "grouped_by": group_by,
                "results": results,
                "query": query,
                "summary": f"Statistical distribution analysis of audio features"
            }
            
        except Exception as e:
            return {"error": f"Statistical analysis failed: {str(e)}"}
    
    def _temporal_analysis(self, queryset, query):
        """Analyze trends over time"""
        try:
            # Monthly trends
            monthly_trends = list(queryset.annotate(
                month=TruncMonth('recording_date')
            ).values('month').annotate(
                dataset_count=Count('id'),
                avg_decibel=Avg('noise_analysis__mean_db'),
                avg_energy=Avg('audio_features__rms_energy'),
                cumulative_energy=Sum('audio_features__rms_energy')
            ).order_by('month'))
            
            # Daily trends (last 30 days)
            from datetime import datetime, timedelta
            thirty_days_ago = timezone.now() - timedelta(days=30)
            
            daily_trends = list(queryset.filter(
                recording_date__gte=thirty_days_ago
            ).annotate(
                date=TruncDate('recording_date')
            ).values('date').annotate(
                dataset_count=Count('id'),
                avg_decibel=Avg('noise_analysis__mean_db'),
                cumulative_energy=Sum('audio_features__rms_energy')
            ).order_by('date'))
            
            return {
                "analysis_type": "temporal_analysis",
                "monthly_trends": monthly_trends,
                "daily_trends": daily_trends,
                "query": query,
                "summary": f"Temporal analysis showing trends over time"
            }
            
        except Exception as e:
            return {"error": f"Temporal analysis failed: {str(e)}"}
    
    def _overview_analysis(self, queryset, query):
        """General overview analysis"""
        try:
            # Overall statistics
            overview = queryset.aggregate(
                total_datasets=Count('id'),
                avg_rms_energy=Avg('audio_features__rms_energy'),
                avg_decibel=Avg('noise_analysis__mean_db'),
                max_decibel=Max('noise_analysis__max_db'),
                avg_duration=Avg('audio_features__duration'),
                total_duration=Sum('audio_features__duration'),
                avg_spectral_centroid=Avg('audio_features__spectral_centroid'),
                avg_dominant_frequency=Avg('noise_analysis__dominant_frequency')
            )
            
            # Regional breakdown
            regional_stats = list(queryset.values('region__name').annotate(
                count=Count('id'),
                avg_decibel=Avg('noise_analysis__mean_db')
            ).order_by('-count'))
            
            # Category breakdown
            category_stats = list(queryset.values('category__name').annotate(
                count=Count('id'),
                avg_decibel=Avg('noise_analysis__mean_db')
            ).order_by('-count'))
            
            return {
                "analysis_type": "overview_analysis",
                "overview_statistics": overview,
                "regional_breakdown": regional_stats,
                "category_breakdown": category_stats,
                "query": query,
                "summary": f"Comprehensive overview of audio data characteristics"
            }
            
        except Exception as e:
            return {"error": f"Overview analysis failed: {str(e)}"}


class AudioDataAggregationTool(BaseTool):
    name: str = "aggregate_audio_data"
    description: str = """Advanced aggregation tool for complex audio data summaries and rollups.
    Use this for questions about:
    - Multi-dimensional aggregations (region + category + time)
    - Percentile calculations and ranking
    - Data quality assessments and completeness
    - Advanced statistical summaries
    - Performance comparisons and benchmarking"""

    def _run(
        self,
        query: str,
        aggregation_type: Optional[str] = None,
        dimensions: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        try:
            from django.db import models
            from django.db.models import Avg, Max, Min, Count, StdDev, Sum, Q
            from django.db.models import Case, When, IntegerField, FloatField
            
            # Determine aggregation type from query
            query_lower = query.lower()
            
            if not aggregation_type:
                if any(word in query_lower for word in ['quality', 'completeness', 'missing', 'coverage']):
                    aggregation_type = 'data_quality'
                elif any(word in query_lower for word in ['ranking', 'top', 'highest', 'lowest', 'best', 'worst']):
                    aggregation_type = 'ranking'
                elif any(word in query_lower for word in ['percentile', 'quartile', 'median', 'distribution']):
                    aggregation_type = 'percentiles'
                elif any(word in query_lower for word in ['benchmark', 'compare', 'performance', 'standard']):
                    aggregation_type = 'benchmarking'
                elif any(word in query_lower for word in ['multi', 'cross', 'breakdown', 'pivot']):
                    aggregation_type = 'multi_dimensional'
                else:
                    aggregation_type = 'summary'
            
            # Get base queryset
            queryset = NoiseDataset.objects.select_related(
                'audio_features', 'noise_analysis', 'region', 'category', 
                'community', 'microphone_type', 'time_of_day'
            )
            
            # Perform aggregation based on type
            if aggregation_type == 'data_quality':
                return self._data_quality_analysis(queryset, query)
            elif aggregation_type == 'ranking':
                return self._ranking_analysis(queryset, query)
            elif aggregation_type == 'percentiles':
                return self._percentile_analysis(queryset, query)
            elif aggregation_type == 'benchmarking':
                return self._benchmarking_analysis(queryset, query)
            elif aggregation_type == 'multi_dimensional':
                return self._multi_dimensional_analysis(queryset, query)
            else:
                return self._summary_analysis(queryset, query)
                
        except Exception as e:
            logger.error(f"Error in audio data aggregation: {e}")
            return {"error": f"Audio data aggregation failed: {str(e)}"}
    
    def _data_quality_analysis(self, queryset, query):
        """Analyze data quality and completeness"""
        try:
            total_datasets = queryset.count()
            
            # Check completeness of key fields
            quality_metrics = {
                "total_datasets": total_datasets,
                "with_audio_features": queryset.filter(audio_features__isnull=False).count(),
                "with_noise_analysis": queryset.filter(noise_analysis__isnull=False).count(),
                "with_region": queryset.filter(region__isnull=False).count(),
                "with_category": queryset.filter(category__isnull=False).count(),
                "with_recording_date": queryset.filter(recording_date__isnull=False).count(),
            }
            
            # Calculate completeness percentages
            completeness = {}
            for key, count in quality_metrics.items():
                if key != "total_datasets" and total_datasets > 0:
                    completeness[f"{key}_percentage"] = round((count / total_datasets) * 100, 2)
            
            # Identify datasets with missing critical data
            incomplete_datasets = queryset.filter(
                Q(audio_features__isnull=True) | 
                Q(noise_analysis__isnull=True) |
                Q(region__isnull=True) |
                Q(category__isnull=True)
            ).count()
            
            return {
                "analysis_type": "data_quality",
                "quality_metrics": quality_metrics,
                "completeness_percentages": completeness,
                "incomplete_datasets": incomplete_datasets,
                "data_quality_score": round(((total_datasets - incomplete_datasets) / total_datasets) * 100, 2) if total_datasets > 0 else 0,
                "query": query,
                "summary": f"Data quality analysis shows {completeness.get('with_audio_features_percentage', 0)}% completeness for audio features"
            }
            
        except Exception as e:
            return {"error": f"Data quality analysis failed: {str(e)}"}
    
    def _ranking_analysis(self, queryset, query):
        """Rank datasets by various metrics"""
        try:
            query_lower = query.lower()
            
            # Determine ranking criteria
            if 'decibel' in query_lower or 'loudest' in query_lower:
                ranking = list(queryset.filter(
                    noise_analysis__isnull=False
                ).select_related('region', 'category').order_by(
                    '-noise_analysis__max_db'
                )[:10].values(
                    'name', 'region__name', 'category__name',
                    'noise_analysis__max_db', 'noise_analysis__mean_db'
                ))
                metric = "decibel_levels"
                
            elif 'energy' in query_lower or 'rms' in query_lower:
                ranking = list(queryset.filter(
                    audio_features__isnull=False
                ).select_related('region', 'category').order_by(
                    '-audio_features__rms_energy'
                )[:10].values(
                    'name', 'region__name', 'category__name',
                    'audio_features__rms_energy', 'audio_features__duration'
                ))
                metric = "rms_energy"
                
            elif 'frequency' in query_lower:
                ranking = list(queryset.filter(
                    noise_analysis__isnull=False
                ).select_related('region', 'category').order_by(
                    '-noise_analysis__dominant_frequency'
                )[:10].values(
                    'name', 'region__name', 'category__name',
                    'noise_analysis__dominant_frequency'
                ))
                metric = "dominant_frequency"
                
            else:
                # Default: rank by dataset completeness
                ranking = list(queryset.annotate(
                    completeness_score=Case(
                        When(audio_features__isnull=False, then=1),
                        default=0,
                        output_field=IntegerField()
                    ) + Case(
                        When(noise_analysis__isnull=False, then=1),
                        default=0,
                        output_field=IntegerField()
                    )
                ).select_related('region', 'category').order_by(
                    '-completeness_score'
                )[:10].values(
                    'name', 'region__name', 'category__name', 'completeness_score'
                ))
                metric = "completeness"
            
            return {
                "analysis_type": "ranking",
                "ranking_metric": metric,
                "top_10_results": ranking,
                "query": query,
                "summary": f"Top 10 datasets ranked by {metric}"
            }
            
        except Exception as e:
            return {"error": f"Ranking analysis failed: {str(e)}"}
    
    def _percentile_analysis(self, queryset, query):
        """Calculate percentiles and distribution statistics"""
        try:
            from django.db.models import Percentile
            
            # Get datasets with complete data
            complete_datasets = queryset.filter(
                audio_features__isnull=False,
                noise_analysis__isnull=False
            )
            
            # Calculate percentiles for key metrics
            percentiles = {}
            
            # Decibel percentiles
            db_percentiles = complete_datasets.aggregate(
                db_p25=Percentile('noise_analysis__mean_db', 0.25),
                db_p50=Percentile('noise_analysis__mean_db', 0.50),
                db_p75=Percentile('noise_analysis__mean_db', 0.75),
                db_p90=Percentile('noise_analysis__mean_db', 0.90),
                db_p95=Percentile('noise_analysis__mean_db', 0.95)
            )
            percentiles['decibel_levels'] = db_percentiles
            
            # RMS Energy percentiles
            energy_percentiles = complete_datasets.aggregate(
                energy_p25=Percentile('audio_features__rms_energy', 0.25),
                energy_p50=Percentile('audio_features__rms_energy', 0.50),
                energy_p75=Percentile('audio_features__rms_energy', 0.75),
                energy_p90=Percentile('audio_features__rms_energy', 0.90),
                energy_p95=Percentile('audio_features__rms_energy', 0.95)
            )
            percentiles['rms_energy'] = energy_percentiles
            
            # Duration percentiles
            duration_percentiles = complete_datasets.aggregate(
                duration_p25=Percentile('audio_features__duration', 0.25),
                duration_p50=Percentile('audio_features__duration', 0.50),
                duration_p75=Percentile('audio_features__duration', 0.75),
                duration_p90=Percentile('audio_features__duration', 0.90)
            )
            percentiles['duration'] = duration_percentiles
            
            return {
                "analysis_type": "percentiles",
                "percentile_distributions": percentiles,
                "total_datasets_analyzed": complete_datasets.count(),
                "query": query,
                "summary": f"Percentile analysis across {complete_datasets.count()} complete datasets"
            }
            
        except Exception as e:
            # Fallback if Percentile is not available
            return self._fallback_percentile_analysis(queryset, query, str(e))
    
    def _fallback_percentile_analysis(self, queryset, query, original_error):
        """Fallback percentile analysis using basic statistics"""
        try:
            complete_datasets = queryset.filter(
                audio_features__isnull=False,
                noise_analysis__isnull=False
            )
            
            # Basic statistical summary
            stats = complete_datasets.aggregate(
                db_avg=Avg('noise_analysis__mean_db'),
                db_max=Max('noise_analysis__mean_db'),
                db_min=Min('noise_analysis__mean_db'),
                db_std=StdDev('noise_analysis__mean_db'),
                energy_avg=Avg('audio_features__rms_energy'),
                energy_max=Max('audio_features__rms_energy'),
                energy_min=Min('audio_features__rms_energy'),
                energy_std=StdDev('audio_features__rms_energy'),
                duration_avg=Avg('audio_features__duration'),
                duration_max=Max('audio_features__duration'),
                duration_min=Min('audio_features__duration')
            )
            
            return {
                "analysis_type": "statistical_summary",
                "statistics": stats,
                "total_datasets": complete_datasets.count(),
                "note": f"Advanced percentile calculation not available: {original_error}",
                "query": query,
                "summary": f"Statistical summary of {complete_datasets.count()} complete datasets"
            }
            
        except Exception as e:
            return {"error": f"Fallback percentile analysis failed: {str(e)}"}
    
    def _benchmarking_analysis(self, queryset, query):
        """Compare performance against benchmarks"""
        try:
            # Calculate overall benchmarks
            overall_stats = queryset.filter(
                audio_features__isnull=False,
                noise_analysis__isnull=False
            ).aggregate(
                avg_decibel=Avg('noise_analysis__mean_db'),
                avg_energy=Avg('audio_features__rms_energy'),
                avg_duration=Avg('audio_features__duration'),
                avg_frequency=Avg('noise_analysis__dominant_frequency')
            )
            
            # Compare by region
            regional_comparison = list(queryset.filter(
                audio_features__isnull=False,
                noise_analysis__isnull=False
            ).values('region__name').annotate(
                avg_decibel=Avg('noise_analysis__mean_db'),
                avg_energy=Avg('audio_features__rms_energy'),
                dataset_count=Count('id'),
                above_avg_decibel=Count(Case(
                    When(noise_analysis__mean_db__gt=overall_stats['avg_decibel'], then=1),
                    output_field=IntegerField()
                )),
                above_avg_energy=Count(Case(
                    When(audio_features__rms_energy__gt=overall_stats['avg_energy'], then=1),
                    output_field=IntegerField()
                ))
            ).order_by('-avg_decibel'))
            
            return {
                "analysis_type": "benchmarking",
                "system_benchmarks": overall_stats,
                "regional_performance": regional_comparison,
                "query": query,
                "summary": f"Performance benchmarking across regions compared to system averages"
            }
            
        except Exception as e:
            return {"error": f"Benchmarking analysis failed: {str(e)}"}
    
    def _multi_dimensional_analysis(self, queryset, query):
        """Multi-dimensional cross-tabulation analysis"""
        try:
            # Region + Category breakdown
            region_category = list(queryset.values(
                'region__name', 'category__name'
            ).annotate(
                dataset_count=Count('id'),
                avg_decibel=Avg('noise_analysis__mean_db'),
                avg_energy=Avg('audio_features__rms_energy')
            ).order_by('region__name', 'category__name'))
            
            # Microphone + Time of Day breakdown
            mic_time = list(queryset.values(
                'microphone_type__name', 'time_of_day__name'
            ).annotate(
                dataset_count=Count('id'),
                avg_decibel=Avg('noise_analysis__mean_db')
            ).order_by('microphone_type__name', 'time_of_day__name'))
            
            return {
                "analysis_type": "multi_dimensional",
                "region_category_breakdown": region_category,
                "microphone_time_breakdown": mic_time,
                "query": query,
                "summary": f"Multi-dimensional analysis across regions, categories, microphones, and time periods"
            }
            
        except Exception as e:
            return {"error": f"Multi-dimensional analysis failed: {str(e)}"}
    
    def _summary_analysis(self, queryset, query):
        """General summary analysis"""
        try:
            # Overall system summary
            summary = {
                "total_datasets": queryset.count(),
                "with_complete_analysis": queryset.filter(
                    audio_features__isnull=False,
                    noise_analysis__isnull=False
                ).count(),
                "unique_regions": queryset.values('region').distinct().count(),
                "unique_categories": queryset.values('category').distinct().count(),
                "unique_communities": queryset.values('community').distinct().count(),
                "date_range": queryset.aggregate(
                    earliest=Min('recording_date'),
                    latest=Max('recording_date')
                )
            }
            
            return {
                "analysis_type": "summary",
                "system_summary": summary,
                "query": query,
                "summary": f"System contains {summary['total_datasets']} datasets across {summary['unique_regions']} regions"
            }
            
        except Exception as e:
            return {"error": f"Summary analysis failed: {str(e)}"}


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
                "mean_db": getattr(dataset.noise_analysis, "mean_db", None) if hasattr(dataset, 'noise_analysis') and dataset.noise_analysis else None,
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
    model=AGENT_CONFIG.get("MODEL", "gpt-4"),
    api_key=os.getenv("OPENAI_API_KEY")
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
                if hasattr(last_message, 'content'):
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
    data_summary: Optional[str] = Field(default=None, description="Summary of the data to be visualized")


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

    def _run(self, query: str, data_summary: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        try:
            # Analyze the query and data to determine the best visualization
            analysis_prompt = f"""
            Analyze the following audio data query and data to recommend the best visualization type:
            
            Query: {query}
            Data Summary: {data_summary or "No data summary provided"}
            
            Available chart types for audio data:
            - pie_chart: Best for showing proportions/percentages of audio categories, regions, or device types
            - bar_chart: Best for comparing audio metrics across categories (decibel levels, frequency ranges, etc.)
            - line_chart: Best for showing audio trends over time, frequency response curves, or decibel trends
            - heatmap: Best for showing frequency correlations, spectral patterns, or geographic audio patterns
            - scatter_plot: Best for showing relationships between audio variables (frequency vs amplitude, etc.)
            - box_plot: Best for showing distribution of audio metrics (decibel levels, frequency ranges, etc.)
            - area_chart: Best for showing cumulative audio energy or frequency spectrum analysis
            
            Audio-specific considerations:
            - Frequency analysis often benefits from line charts or area charts
            - Decibel level comparisons work well with bar charts or box plots
            - Geographic audio data distribution works well with pie charts or heatmaps
            - Spectral characteristics are best shown with line charts or heatmaps
            - Audio feature correlations work well with scatter plots
            
            Return a JSON response with:
            1. recommended_chart: The best chart type for this audio data
            2. reasoning: Why this chart type is best for the audio data and analysis
            3. chart_template: A template object for creating the visualization
            4. data_requirements: What audio data fields are needed for this visualization
            """
            
            # Use a simple LLM call to analyze and recommend
            llm = ChatOpenAI(
                model=AGENT_CONFIG.get("MODEL", "gpt-4o-mini"),
                api_key=os.getenv("OPENAI_API_KEY")
            )
            response = llm.invoke([HumanMessage(content=analysis_prompt)])
            
            # Parse the response and create chart template
            recommendation = self._parse_visualization_recommendation(response.content, query)
            
            chart_type = recommendation["recommended_chart"]
            chart_template = self._generate_chart_template(chart_type)
            
            return {
                "visualization_type": chart_type,
                "visualization_name": self._get_visualization_name(chart_type),
                "chart_template": chart_template,
                "recommendation": recommendation,
                "frontend_data": {
                    "type": chart_type,
                    "name": self._get_visualization_name(chart_type),
                    "config": chart_template["config"],
                    "data_structure": self._get_data_structure(chart_type),
                    "description": recommendation.get("reasoning", "Audio data visualization")
                },
                "message": f"Recommended {self._get_visualization_name(chart_type)} for this data analysis"
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
                    "data_requirements": ["category", "value"]
                },
                "frontend_data": {
                    "type": chart_type,
                    "name": self._get_visualization_name(chart_type),
                    "config": chart_template["config"],
                    "data_structure": self._get_data_structure(chart_type),
                    "description": "Default bar chart for audio data visualization"
                },
                "message": "Error in visualization analysis, using default bar chart"
            }
    
    def _parse_visualization_recommendation(self, llm_response: str, query: str) -> Dict[str, Any]:
        """Parse LLM response and extract visualization recommendation"""
        try:
            # Audio-specific keyword-based analysis if JSON parsing fails
            query_lower = query.lower()
            
            # Audio-specific keywords for pie charts
            if any(word in query_lower for word in ["proportion", "percentage", "share", "part of", "distribution by", "breakdown by", "region", "category", "device type"]):
                return {
                    "recommended_chart": "pie_chart",
                    "reasoning": "Query asks for proportions or distribution of audio categories/regions",
                    "data_requirements": ["audio_category", "count_or_percentage"]
                }
            # Audio-specific keywords for line charts
            elif any(word in query_lower for word in ["trend", "over time", "time series", "change over", "frequency response", "spectrum", "decibel trend", "audio level"]):
                return {
                    "recommended_chart": "line_chart",
                    "reasoning": "Query asks for audio trends over time or frequency analysis",
                    "data_requirements": ["time_or_frequency", "audio_value"]
                }
            # Audio-specific keywords for heatmaps
            elif any(word in query_lower for word in ["correlation", "relationship", "pattern", "heat", "spectral", "frequency correlation", "geographic pattern"]):
                return {
                    "recommended_chart": "heatmap",
                    "reasoning": "Query asks for audio correlations, spectral patterns, or geographic audio patterns",
                    "data_requirements": ["x_axis", "y_axis", "audio_intensity"]
                }
            # Audio-specific keywords for box plots
            elif any(word in query_lower for word in ["distribution", "outliers", "quartile", "median", "decibel level", "frequency range", "audio statistics"]):
                return {
                    "recommended_chart": "box_plot",
                    "reasoning": "Query asks for distribution analysis of audio metrics",
                    "data_requirements": ["audio_category", "numerical_audio_values"]
                }
            # Audio-specific keywords for scatter plots
            elif any(word in query_lower for word in ["scatter", "relationship between", "correlation between", "frequency vs", "amplitude vs", "audio feature"]):
                return {
                    "recommended_chart": "scatter_plot",
                    "reasoning": "Query asks for relationship between audio variables",
                    "data_requirements": ["audio_variable_x", "audio_variable_y"]
                }
            # Audio-specific keywords for area charts
            elif any(word in query_lower for word in ["cumulative", "total over", "area under", "energy", "spectrum analysis", "frequency spectrum"]):
                return {
                    "recommended_chart": "area_chart",
                    "reasoning": "Query asks for cumulative audio energy or spectrum analysis",
                    "data_requirements": ["frequency_or_time", "cumulative_audio_value"]
                }
            # Audio-specific keywords for bar charts
            elif any(word in query_lower for word in ["compare", "comparison", "decibel", "audio level", "frequency", "acoustic", "noise level"]):
                return {
                    "recommended_chart": "bar_chart",
                    "reasoning": "Query asks for comparison of audio metrics across categories",
                    "data_requirements": ["audio_category", "audio_metric_value"]
                }
            else:
                return {
                    "recommended_chart": "bar_chart",
                    "reasoning": "Default choice for general audio data comparisons",
                    "data_requirements": ["audio_category", "audio_value"]
                }
        except Exception as e:
            logger.error(f"Error parsing visualization recommendation: {e}")
            return {
                "recommended_chart": "bar_chart",
                "reasoning": "Default recommendation due to parsing error",
                "data_requirements": ["category", "value"]
            }
    
    def _generate_chart_template(self, chart_type: str) -> Dict[str, Any]:
        """Generate chart template based on chart type"""
        templates = {
            "pie_chart": {
                "type": "pie",
                "config": {
                    "data": {
                        "labels": [],
                        "datasets": [{
                            "data": [],
                            "backgroundColor": [
                                "#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0",
                                "#9966FF", "#FF9F40", "#FF6384", "#C9CBCF"
                            ]
                        }]
                    },
                    "options": {
                        "responsive": True,
                        "plugins": {
                            "legend": {"position": "bottom"},
                            "title": {"display": True, "text": "Data Distribution"}
                        }
                    }
                }
            },
            "bar_chart": {
                "type": "bar",
                "config": {
                    "data": {
                        "labels": [],
                        "datasets": [{
                            "label": "Values",
                            "data": [],
                            "backgroundColor": "#36A2EB",
                            "borderColor": "#36A2EB",
                            "borderWidth": 1
                        }]
                    },
                    "options": {
                        "responsive": True,
                        "scales": {
                            "y": {"beginAtZero": True}
                        },
                        "plugins": {
                            "title": {"display": True, "text": "Data Comparison"}
                        }
                    }
                }
            },
            "line_chart": {
                "type": "line",
                "config": {
                    "data": {
                        "labels": [],
                        "datasets": [{
                            "label": "Trend",
                            "data": [],
                            "borderColor": "#36A2EB",
                            "backgroundColor": "rgba(54, 162, 235, 0.1)",
                            "tension": 0.1
                        }]
                    },
                    "options": {
                        "responsive": True,
                        "scales": {
                            "y": {"beginAtZero": True}
                        },
                        "plugins": {
                            "title": {"display": True, "text": "Trend Analysis"}
                        }
                    }
                }
            },
            "heatmap": {
                "type": "heatmap",
                "config": {
                    "data": {
                        "datasets": [{
                            "label": "Heatmap",
                            "data": [],
                            "backgroundColor": "rgba(54, 162, 235, 0.8)"
                        }]
                    },
                    "options": {
                        "responsive": True,
                        "plugins": {
                            "title": {"display": True, "text": "Data Patterns"}
                        },
                        "scales": {
                            "x": {"type": "category"},
                            "y": {"type": "category"}
                        }
                    }
                }
            },
            "scatter_plot": {
                "type": "scatter",
                "config": {
                    "data": {
                        "datasets": [{
                            "label": "Data Points",
                            "data": [],
                            "backgroundColor": "#36A2EB",
                            "borderColor": "#36A2EB"
                        }]
                    },
                    "options": {
                        "responsive": True,
                        "scales": {
                            "x": {"type": "linear", "position": "bottom"},
                            "y": {"type": "linear"}
                        },
                        "plugins": {
                            "title": {"display": True, "text": "Relationship Analysis"}
                        }
                    }
                }
            },
            "box_plot": {
                "type": "boxplot",
                "config": {
                    "data": {
                        "labels": [],
                        "datasets": [{
                            "label": "Distribution",
                            "data": [],
                            "backgroundColor": "rgba(54, 162, 235, 0.5)",
                            "borderColor": "#36A2EB"
                        }]
                    },
                    "options": {
                        "responsive": True,
                        "plugins": {
                            "title": {"display": True, "text": "Data Distribution"}
                        }
                    }
                }
            },
            "area_chart": {
                "type": "line",
                "config": {
                    "data": {
                        "labels": [],
                        "datasets": [{
                            "label": "Cumulative",
                            "data": [],
                            "borderColor": "#36A2EB",
                            "backgroundColor": "rgba(54, 162, 235, 0.3)",
                            "fill": True,
                            "tension": 0.1
                        }]
                    },
                    "options": {
                        "responsive": True,
                        "scales": {
                            "y": {"beginAtZero": True}
                        },
                        "plugins": {
                            "title": {"display": True, "text": "Cumulative Analysis"}
                        }
                    }
                }
            }
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
            "area_chart": "Area Chart"
        }
        return names.get(chart_type, "Bar Chart")
    
    def _get_data_structure(self, chart_type: str) -> Dict[str, Any]:
        """Get expected data structure for each chart type"""
        structures = {
            "pie_chart": {
                "labels": "Array of category names",
                "data": "Array of values corresponding to labels",
                "description": "For showing proportions/percentages of audio categories"
            },
            "bar_chart": {
                "labels": "Array of category names",
                "data": "Array of values for each category",
                "description": "For comparing audio metrics across categories"
            },
            "line_chart": {
                "labels": "Array of time points or frequency values",
                "data": "Array of values over time/frequency",
                "description": "For showing audio trends over time or frequency analysis"
            },
            "heatmap": {
                "x_labels": "Array of x-axis categories",
                "y_labels": "Array of y-axis categories", 
                "data": "2D array of intensity values",
                "description": "For showing audio correlations and patterns"
            },
            "scatter_plot": {
                "x_data": "Array of x-axis values",
                "y_data": "Array of y-axis values",
                "description": "For showing relationships between audio variables"
            },
            "box_plot": {
                "labels": "Array of category names",
                "data": "Array of arrays containing numerical values for each category",
                "description": "For showing distribution of audio metrics"
            },
            "area_chart": {
                "labels": "Array of time points or frequency values",
                "data": "Array of cumulative values",
                "description": "For showing cumulative audio energy or spectrum analysis"
            }
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
                AudioDataAggregationTool(),
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
