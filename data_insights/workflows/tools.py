from pydantic import BaseModel ,Field,model_validator
from typing import Dict, List, Any, Literal, Optional
from typing import Optional, Dict, Any
from pydantic import BaseModel
from django.db.models import Q
import uuid
import logging
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
from data.models import NoiseDataset,AudioFeature,BulkAudioUpload,BulkReprocessingTask



class NoiseDatasetSearchInput(BaseModel):
    """input for noise dataset search"""
    
    filter_criteria:Optional[Dict[
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
        ]
    ]]= Field(default_factory=dict, description="Filter criteria for noise dataset")







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
        default=True, description="Include region, category, class, and related metadata"
    )






# Tool implementation for searching Noise Datasets
class NoiseDatasetSearchTool(BaseTool):
    name: str = "search_noise_datasets"
    description: str = """Search for noise datasets based on criteria. 
    Returns a query handle for large result sets. 
    Use this to find datasets by name, location, noise_level, date range, etc."""

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

            if "location" in filter_criteria:
                queryset = queryset.filter(location__icontains=filter_criteria["location"])

            if "noise_level_min" in filter_criteria:
                queryset = queryset.filter(noise_level__gte=filter_criteria["noise_level_min"])

            if "noise_level_max" in filter_criteria:
                queryset = queryset.filter(noise_level__lte=filter_criteria["noise_level_max"])

            if "date_from" in filter_criteria:
                queryset = queryset.filter(recorded_at__gte=filter_criteria["date_from"])

            if "date_to" in filter_criteria:
                queryset = queryset.filter(recorded_at__lte=filter_criteria["date_to"])

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
                        "location",
                        "noise_level",
                        "recorded_at",
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
                        "location": dataset.location,
                        "noise_level": dataset.noise_level,
                        "recorded_at": dataset.recorded_at,
                    }

                    if include_features and hasattr(dataset, "features"):
                        features = dataset.features
                        dataset_data["features"] = {
                            "rms": getattr(features, "rms", None),
                            "spectral_centroid": getattr(features, "spectral_centroid", None),
                            "zero_crossing_rate": getattr(features, "zero_crossing_rate", None),
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

            # Apply filters
            if "rms" in filter_criteria:
                queryset = queryset.filter(rms__gte=filter_criteria["rms"])

            if "zcr" in filter_criteria:
                queryset = queryset.filter(zcr__gte=filter_criteria["zcr"])

            if "spectral_centroid" in filter_criteria:
                queryset = queryset.filter(
                    spectral_centroid__gte=filter_criteria["spectral_centroid"]
                )

            if "spectral_bandwidth" in filter_criteria:
                queryset = queryset.filter(
                    spectral_bandwidth__gte=filter_criteria["spectral_bandwidth"]
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
                            "rms": feature.rms,
                            "zcr": feature.zcr,
                            "spectral_centroid": feature.spectral_centroid,
                            "spectral_bandwidth": feature.spectral_bandwidth,
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
                "location": getattr(dataset, "location", None),
                "noise_level": getattr(dataset, "noise_level", None),
                "recorded_at": getattr(dataset, "recorded_at", None),
                "duration": getattr(dataset, "duration", None),
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




llm = ChatOpenAI(model=AGENT_CONFIG.get("MODEL", "gpt-4"))

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
        from  .prompt import SQL_SYSTEM_TEMPLATE

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
            msg = response["messages"][-1].content

            if "no results found" in str(msg).lower():
                return {"message": "No results found"}
            if "error" in str(msg).lower():
                return {"message": "Error in data analysis tool"}

            return {"message": msg}

        except Exception as e:
            logger.error(f"Error in data analysis tool: {e}")
            return {"message": "Error in data analysis tool"}




