import logging
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from django.db import connection
from django.db.models import Sum
from django.conf import settings

from .intent_classifier import IntentClassifier
from .rag_service import RAGService

logger = logging.getLogger(__name__)


class DatasetService:
    """Hybrid service that routes questions to SQL or RAG based on intent"""

    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.rag_service = None  # Initialize lazily

    def query_dataset(
        self, question: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for dataset queries

        Args:
            question: User's natural language question
            context: Additional context (table name, filters, etc.)

        Returns:
            {
                'answer': str,
                'intent': str,
                'data_used': dict,
                'confidence': float,
                'processing_time': float
            }
        """
        import time

        start_time = time.time()

        # Classify intent
        routing_info = self.intent_classifier.get_routing_info(question)
        intent = routing_info["intent"]

        logger.info(f"Classified question as {intent}: {question}")

        # Route to appropriate handler
        if intent == "NUMERIC":
            result = self._handle_numeric_query(question, context)
        elif intent == "EXPLANATORY":
            result = self._handle_explanatory_query(question, context)
        elif intent == "MIXED":
            result = self._handle_mixed_query(question, context)
        else:
            result = self._handle_explanatory_query(question, context)

        # Add metadata
        result.update(
            {
                "intent": intent,
                "confidence": routing_info["confidence"],
                "processing_time": time.time() - start_time,
                "routing_reasoning": routing_info["reasoning"],
            }
        )

        return result

    def _handle_numeric_query(
        self, question: str, context: Dict = None
    ) -> Dict[str, Any]:
        """
        Handle numeric queries using SQL/database operations

        This is where you'd implement:
        - SQL query generation
        - Pandas operations
        - Database aggregations
        """
        from chatbot.models import Document

        question_lower = question.lower()

        # Handle specific dataset/document counting and statistical questions
        # More flexible matching for counting and statistical keywords
        counting_keywords = [
            "how many",
            "how much",
            "count",
            "number of",
            "total",
            "amount of",
        ]
        statistical_keywords = [
            "average",
            "mean",
            "median",
            "min",
            "max",
            "minimum",
            "maximum",
            "std",
            "standard deviation",
            "variance",
            "distribution",
            "statistics",
            "stats",
            "summary",
            "analysis",
        ]
        dataset_keywords = [
            "dataset",
            "datasets",
            "datssets",
            "datsset",
            "document",
            "documents",
            "doc",
            "docs",
            "file",
            "files",
            "collection",
            "collections",
            "recording",
            "recordings",
            "audio",
            "noise",
            "speech",
            "clean",
            "mixed",
            "animals",
            "sound",
            "sample",
            "samples",
        ]

        has_counting = any(keyword in question_lower for keyword in counting_keywords)
        has_statistics = any(
            keyword in question_lower for keyword in statistical_keywords
        )
        has_dataset = any(keyword in question_lower for keyword in dataset_keywords)

        # Check for statistical questions about audio features
        audio_features = [
            "mfcc",
            "spectral",
            "frequency",
            "duration",
            "sample rate",
            "rms",
            "energy",
            "chroma",
            "feature",
            "features",
        ]
        has_audio_features = any(
            feature in question_lower for feature in audio_features
        )

        if (
            (has_counting and has_dataset)
            or (has_statistics and has_dataset)
            or (has_audio_features)
        ):

            try:
                from data.models import (
                    Dataset as AudioDataset,
                    NoiseDataset,
                    AudioFeature,
                    NoiseAnalysis,
                )

                # First try LLM-powered SQL generation for complex queries
                # This allows the AI to understand the schema and generate appropriate queries
                if (
                    len(question_lower.split()) > 3
                ):  # Complex questions with more context
                    llm_result = self.handle_complex_database_query(question)
                    if (
                        llm_result
                        and llm_result["data_used"]["type"] == "llm_generated_sql"
                    ):
                        return llm_result

                # Fall back to rule-based statistical/feature analysis
                if has_audio_features or has_statistics:
                    # Statistical/feature analysis questions
                    return self._handle_statistical_questions(
                        question_lower, has_audio_features
                    )
                else:
                    # Basic counting questions
                    return self._handle_counting_questions()

            except Exception as e:
                return {
                    "answer": f"I encountered an error while checking your datasets: {str(e)}. Please try again or contact support.",
                    "data_used": {"type": "error", "error": str(e)},
                    "sources": [],
                }

        # Handle other numeric questions with a helpful response
        return {
            "answer": f"I detected this as a numeric query: '{question}'. For numeric questions, I would query the database directly to get accurate counts, averages, or filtered results. Currently, I'm set up to analyze documents and can tell you about your document collection, but database queries for other numeric data would be handled here.",
            "data_used": {"type": "numeric_query", "method": "database"},
            "sql_equivalent": self._generate_sql_placeholder(question),
        }

    def _handle_counting_questions(self):
        """Handle basic counting questions about datasets"""
        from data.models import Dataset as AudioDataset, NoiseDataset
        from chatbot.models import Document
        from django.db.models import Sum

        # 1. Document datasets (RAG chatbot documents)
        total_docs = Document.objects.count()
        processed_docs = Document.objects.filter(processed=True).count()
        unprocessed_docs = total_docs - processed_docs
        total_chunks = (
            Document.objects.filter(processed=True).aggregate(
                total_chunks=Sum("total_chunks")
            )["total_chunks"]
            or 0
        )

        # 2. Audio datasets (data collection datasets)
        total_audio_datasets = AudioDataset.objects.count()
        total_noise_records = NoiseDataset.objects.count()

        # Build comprehensive answer
        answer_parts = []

        # Document datasets (RAG)
        if total_docs > 0:
            doc_info = (
                f"You have {total_docs} document datasets uploaded for chatbot analysis"
            )
            if processed_docs > 0:
                doc_info += f" ({processed_docs} processed with {total_chunks} text chunks for querying"
            if unprocessed_docs > 0:
                doc_info += f", {unprocessed_docs} still processing"
            doc_info += ")."
            answer_parts.append(doc_info)

        # Audio datasets (data collection)
        if total_audio_datasets > 0 or total_noise_records > 0:
            audio_info = f"You also have {total_audio_datasets} audio dataset types "
            if total_noise_records > 0:
                audio_info += (
                    f"with {total_noise_records} individual noise recordings collected"
                )
            audio_info += "."
            answer_parts.append(audio_info)

        # If no datasets found
        if total_docs == 0 and total_audio_datasets == 0 and total_noise_records == 0:
            answer = "You don't have any datasets yet. You can upload documents for chatbot analysis or start collecting audio data through the data collection interface."
        else:
            answer = " ".join(answer_parts)

        return {
            "answer": answer,
            "data_used": {
                "type": "database_query",
                "method": "comprehensive_dataset_count",
                "document_datasets": {
                    "total": total_docs,
                    "processed": processed_docs,
                    "chunks": total_chunks,
                },
                "audio_datasets": {
                    "types": total_audio_datasets,
                    "records": total_noise_records,
                },
            },
            "sources": [],
        }

    def _handle_statistical_questions(self, question_lower, has_audio_features):
        """Handle statistical and feature analysis questions"""
        from data.models import (
            Dataset as AudioDataset,
            NoiseDataset,
            AudioFeature,
            NoiseAnalysis,
        )
        from django.db.models import Avg, Max, Min, Count, StdDev
        import json

        # Check what type of statistical question this is
        if "average" in question_lower or "mean" in question_lower:
            return self._get_average_statistics()
        elif "distribution" in question_lower or "breakdown" in question_lower:
            return self._get_distribution_statistics(question_lower)
        elif has_audio_features:
            return self._get_audio_feature_statistics(question_lower)
        elif (
            "summary" in question_lower
            or "stats" in question_lower
            or "statistics" in question_lower
        ):
            return self._get_dataset_summary()
        else:
            return self._get_basic_statistics()

    def _get_average_statistics(self):
        """Get average statistics for audio datasets"""
        from data.models import AudioFeature, NoiseDataset
        from django.db.models import Avg

        try:
            # Get average statistics
            avg_stats = AudioFeature.objects.aggregate(
                avg_duration=Avg("duration"),
                avg_sample_rate=Avg("sample_rate"),
                avg_rms_energy=Avg("rms_energy"),
                avg_spectral_centroid=Avg("spectral_centroid"),
            )

            answer = "Here are the average statistics for your audio recordings:\\n"
            answer += (
                f"• Average duration: {avg_stats['avg_duration']:.2f} seconds\\n"
                if avg_stats["avg_duration"]
                else "• Average duration: Not available\\n"
            )
            answer += (
                f"• Average sample rate: {avg_stats['avg_sample_rate']:.0f} Hz\\n"
                if avg_stats["avg_sample_rate"]
                else "• Average sample rate: Not available\\n"
            )
            answer += (
                f"• Average RMS energy: {avg_stats['avg_rms_energy']:.4f}\\n"
                if avg_stats["avg_rms_energy"]
                else "• Average RMS energy: Not available\\n"
            )
            answer += (
                f"• Average spectral centroid: {avg_stats['avg_spectral_centroid']:.2f} Hz\\n"
                if avg_stats["avg_spectral_centroid"]
                else "• Average spectral centroid: Not available\\n"
            )

            total_features = AudioFeature.objects.count()
            answer += f"\\nBased on {total_features} analyzed audio recordings."

            return {
                "answer": answer,
                "data_used": {
                    "type": "statistical_analysis",
                    "method": "averages",
                    "total_recordings": total_features,
                    "statistics": avg_stats,
                },
                "sources": [],
            }
        except Exception as e:
            return {
                "answer": f"I couldn't retrieve average statistics due to an error: {str(e)}",
                "data_used": {"type": "error", "error": str(e)},
                "sources": [],
            }

    def _get_distribution_statistics(self, question_lower):
        """Get distribution statistics by categories"""
        from data.models import NoiseDataset
        from django.db.models import Count

        try:
            answer = "Here are the distributions in your audio datasets:\\n\\n"

            # Region distribution
            if "region" in question_lower:
                regions = (
                    NoiseDataset.objects.values("region__name")
                    .annotate(count=Count("id"))
                    .order_by("-count")
                )
                if regions:
                    answer += "📍 **By Region:**\\n"
                    for region in regions:
                        if region["region__name"]:
                            answer += f"• {region['region__name']}: {region['count']} recordings\\n"
                    answer += "\\n"

            # Category distribution
            if "category" in question_lower or "breakdown" in question_lower:
                categories = (
                    NoiseDataset.objects.values("category__name")
                    .annotate(count=Count("id"))
                    .order_by("-count")
                )
                if categories:
                    answer += "📂 **By Category:**\\n"
                    for category in categories:
                        if category["category__name"]:
                            answer += f"• {category['category__name']}: {category['count']} recordings\\n"
                    answer += "\\n"

            # Time of day distribution
            if "time" in question_lower or "day" in question_lower:
                times = (
                    NoiseDataset.objects.values("time_of_day__name")
                    .annotate(count=Count("id"))
                    .order_by("-count")
                )
                if times:
                    answer += "🕐 **By Time of Day:**\\n"
                    for time in times:
                        if time["time_of_day__name"]:
                            answer += f"• {time['time_of_day__name']}: {time['count']} recordings\\n"
                    answer += "\\n"

            # Community distribution
            if "community" in question_lower:
                communities = (
                    NoiseDataset.objects.values("community__name")
                    .annotate(count=Count("id"))
                    .order_by("-count")
                )
                if communities:
                    answer += "🏘️ **By Community:**\\n"
                    for community in communities:
                        if community["community__name"]:
                            answer += f"• {community['community__name']}: {community['count']} recordings\\n"
                    answer += "\\n"

            total_recordings = NoiseDataset.objects.count()
            answer += f"**Total recordings analyzed:** {total_recordings}"

            return {
                "answer": answer,
                "data_used": {
                    "type": "distribution_analysis",
                    "method": "categorical_breakdown",
                    "total_recordings": total_recordings,
                },
                "sources": [],
            }
        except Exception as e:
            return {
                "answer": f"I couldn't retrieve distribution statistics due to an error: {str(e)}",
                "data_used": {"type": "error", "error": str(e)},
                "sources": [],
            }

    def _get_audio_feature_statistics(self, question_lower):
        """Get statistics about audio features"""
        from data.models import AudioFeature
        from django.db.models import Avg, Max, Min, StdDev

        try:
            answer = "Here are the audio feature statistics from your analyzed recordings:\\n\\n"

            # Basic audio properties
            basic_stats = AudioFeature.objects.aggregate(
                avg_duration=Avg("duration"),
                max_duration=Max("duration"),
                min_duration=Min("duration"),
                avg_sample_rate=Avg("sample_rate"),
                avg_rms=Avg("rms_energy"),
                std_rms=StdDev("rms_energy"),
            )

            if basic_stats["avg_duration"]:
                answer += f"**Duration Statistics:**\\n"
                answer += f"• Average: {basic_stats['avg_duration']:.2f} seconds\\n"
                if basic_stats["max_duration"]:
                    answer += f"• Longest: {basic_stats['max_duration']:.2f} seconds\\n"
                if basic_stats["min_duration"]:
                    answer += (
                        f"• Shortest: {basic_stats['min_duration']:.2f} seconds\\n"
                    )
                answer += "\\n"

            if basic_stats["avg_sample_rate"]:
                answer += f"**Sample Rate:** {basic_stats['avg_sample_rate']:.0f} Hz average\\n\\n"

            # Spectral features
            spectral_stats = AudioFeature.objects.aggregate(
                avg_centroid=Avg("spectral_centroid"),
                avg_bandwidth=Avg("spectral_bandwidth"),
                avg_rolloff=Avg("spectral_rolloff"),
            )

            if spectral_stats["avg_centroid"]:
                answer += f"**Spectral Features:**\\n"
                answer += f"• Average spectral centroid: {spectral_stats['avg_centroid']:.2f} Hz\\n"
                if spectral_stats["avg_bandwidth"]:
                    answer += f"• Average bandwidth: {spectral_stats['avg_bandwidth']:.2f} Hz\\n"
                if spectral_stats["avg_rolloff"]:
                    answer += (
                        f"• Average rolloff: {spectral_stats['avg_rolloff']:.2f} Hz\\n"
                    )
                answer += "\\n"

            # MFCC information
            mfcc_count = AudioFeature.objects.exclude(mfccs__isnull=True).count()
            if mfcc_count > 0:
                answer += f"**MFCC Analysis:** Available for {mfcc_count} recordings\\n"
                answer += "• 13 Mel-Frequency Cepstral Coefficients extracted per recording\\n\\n"

            total_analyzed = AudioFeature.objects.count()
            answer += f"**Analysis Summary:** {total_analyzed} recordings have been analyzed for audio features."

            return {
                "answer": answer,
                "data_used": {
                    "type": "feature_analysis",
                    "method": "audio_features",
                    "total_analyzed": total_analyzed,
                    "statistics": {**basic_stats, **spectral_stats},
                },
                "sources": [],
            }
        except Exception as e:
            return {
                "answer": f"I couldn't retrieve audio feature statistics due to an error: {str(e)}",
                "data_used": {"type": "error", "error": str(e)},
                "sources": [],
            }

    def _get_dataset_summary(self):
        """Get comprehensive dataset summary"""
        from data.models import (
            Dataset as AudioDataset,
            NoiseDataset,
            AudioFeature,
            NoiseAnalysis,
        )
        from django.db.models import Count

        try:
            total_datasets = AudioDataset.objects.count()
            total_recordings = NoiseDataset.objects.count()
            analyzed_recordings = AudioFeature.objects.count()
            analyzed_noise = NoiseAnalysis.objects.count()

            answer = "**Dataset Summary**\\n\\n"
            answer += f"**Dataset Types:** {total_datasets}\\n"
            answer += f"**Total Recordings:** {total_recordings}\\n"
            answer += f"**Analyzed for Features:** {analyzed_recordings}\\n"
            answer += f"**Noise Analysis:** {analyzed_noise}\\n\\n"

            # Top categories
            top_categories = (
                NoiseDataset.objects.values("category__name")
                .annotate(count=Count("id"))
                .order_by("-count")[:5]
            )

            if top_categories:
                answer += "**Top Categories:**\\n"
                for category in top_categories:
                    if category["category__name"]:
                        answer += (
                            f"• {category['category__name']}: {category['count']}\\n"
                        )

            return {
                "answer": answer,
                "data_used": {
                    "type": "summary_statistics",
                    "method": "comprehensive_overview",
                    "summary": {
                        "dataset_types": total_datasets,
                        "total_recordings": total_recordings,
                        "analyzed_features": analyzed_recordings,
                        "analyzed_noise": analyzed_noise,
                    },
                },
                "sources": [],
            }
        except Exception as e:
            return {
                "answer": f"I couldn't generate a dataset summary due to an error: {str(e)}",
                "data_used": {"type": "error", "error": str(e)},
                "sources": [],
            }

    def _get_basic_statistics(self):
        """Get basic statistical overview"""
        from data.models import NoiseDataset, AudioFeature
        from django.db.models import Count, Avg

        try:
            total_recordings = NoiseDataset.objects.count()
            analyzed_recordings = AudioFeature.objects.count()

            stats = AudioFeature.objects.aggregate(
                avg_duration=Avg("duration"), avg_sample_rate=Avg("sample_rate")
            )

            answer = f"Your audio dataset contains {total_recordings} recordings, "
            answer += f"with {analyzed_recordings} analyzed for audio features."

            if stats["avg_duration"]:
                answer += f" The average recording duration is {stats['avg_duration']:.2f} seconds."
            if stats["avg_sample_rate"]:
                answer += f" Recordings are sampled at an average rate of {stats['avg_sample_rate']:.0f} Hz."

            return {
                "answer": answer,
                "data_used": {
                    "type": "basic_statistics",
                    "method": "overview",
                    "total_recordings": total_recordings,
                    "analyzed_recordings": analyzed_recordings,
                    "averages": stats,
                },
                "sources": [],
            }
        except Exception as e:
            return {
                "answer": f"I couldn't retrieve basic statistics due to an error: {str(e)}",
                "data_used": {"type": "error", "error": str(e)},
                "sources": [],
            }

    def _get_database_schema(self):
        """Get comprehensive database schema information for LLM"""
        from django.apps import apps
        from django.db import models

        schema_info = {"tables": {}, "relationships": [], "enums": {}}

        # Get all models from data app
        data_models = apps.get_app_config("data").get_models()

        for model in data_models:
            table_name = model._meta.db_table
            fields_info = {}

            for field in model._meta.get_fields():
                if hasattr(field, "column"):
                    field_info = {
                        "type": field.__class__.__name__,
                        "nullable": field.null,
                        "blank": field.blank,
                    }

                    # Add specific field information
                    if hasattr(field, "max_length") and field.max_length:
                        field_info["max_length"] = field.max_length

                    if hasattr(field, "choices") and field.choices:
                        field_info["choices"] = [choice[0] for choice in field.choices]

                    if hasattr(field, "related_model") and field.related_model:
                        # Foreign key relationship
                        try:
                            related_table = field.related_model._meta.db_table
                            field_info["foreign_key_to"] = related_table
                            schema_info["relationships"].append(
                                {
                                    "from_table": table_name,
                                    "from_field": field.column,
                                    "to_table": related_table,
                                    "relationship": "foreign_key",
                                }
                            )
                        except AttributeError:
                            # Handle cases where related_model doesn't have _meta
                            pass

                    if field.__class__.__name__ == "ArrayField":
                        field_info["base_field_type"] = (
                            field.base_field.__class__.__name__
                        )
                        if hasattr(field.base_field, "size"):
                            field_info["array_size"] = field.base_field.size

                    fields_info[field.column] = field_info

            schema_info["tables"][table_name] = {
                "model_name": model.__name__,
                "fields": fields_info,
                "description": model.__doc__ or f"{model.__name__} model",
            }

        return schema_info

    def _validate_sql_security(self, sql_query):
        """
        Comprehensive SQL security validation to ensure only safe read-only queries.

        Args:
            sql_query (str): The SQL query to validate

        Returns:
            tuple: (is_safe: bool, reason: str)
        """
        import re

        if not sql_query or not isinstance(sql_query, str):
            return False, "Empty or invalid query"

        sql_upper = sql_query.upper().strip()

        # Must start with SELECT
        if not sql_upper.startswith("SELECT"):
            return False, "Query must start with SELECT"

        # Dangerous keywords that are NEVER allowed
        dangerous_keywords = [
            "INSERT",
            "UPDATE",
            "DELETE",
            "DROP",
            "CREATE",
            "ALTER",
            "TRUNCATE",
            "EXEC",
            "EXECUTE",
            "MERGE",
            "BULK",
            "BACKUP",
            "RESTORE",
            "GRANT",
            "REVOKE",
            "DENY",
            "COMMIT",
            "ROLLBACK",
            "SAVEPOINT",
            "SHUTDOWN",
            "KILL",
            "DBCC",
            "RECONFIGURE",
        ]

        for keyword in dangerous_keywords:
            if re.search(r"\b" + keyword + r"\b", sql_upper):
                return False, f"Dangerous keyword '{keyword}' not allowed"

        # Subqueries and complex operations that might be risky
        risky_patterns = [
            r";\s*(SELECT|INSERT|UPDATE|DELETE)",  # Multiple statements
            r"UNION\s+SELECT.*INTO",  # UNION with INTO
            r"SELECT.*INTO\s+\w+",  # SELECT INTO
            r"EXEC\s*\(",  # Dynamic execution
            r"SP_",  # Stored procedures (potential risk)
        ]

        for pattern in risky_patterns:
            if re.search(pattern, sql_upper, re.IGNORECASE):
                return False, f"Risky SQL pattern detected: {pattern}"

        # Check for potentially dangerous functions
        dangerous_functions = [
            "XP_CMDSHELL",
            "SP_EXECUTESQL",
            "OPENROWSET",
            "OPENDATASOURCE",
            "BULK INSERT",
            "FORMAT",
            "CONVERT",
            "CAST",
        ]

        for func in dangerous_functions:
            if func.upper() in sql_upper:
                return False, f"Potentially dangerous function '{func}' not allowed"

        # Allow only safe clauses and keywords
        allowed_keywords = [
            "SELECT",
            "FROM",
            "WHERE",
            "JOIN",
            "INNER",
            "LEFT",
            "RIGHT",
            "FULL",
            "OUTER",
            "ON",
            "GROUP",
            "BY",
            "HAVING",
            "ORDER",
            "ASC",
            "DESC",
            "LIMIT",
            "OFFSET",
            "DISTINCT",
            "AS",
            "AND",
            "OR",
            "NOT",
            "IN",
            "EXISTS",
            "BETWEEN",
            "LIKE",
            "COUNT",
            "SUM",
            "AVG",
            "MAX",
            "MIN",
            "STDDEV",
            "VARIANCE",
            "CASE",
            "WHEN",
            "THEN",
            "ELSE",
            "END",
            "COALESCE",
            "NULLIF",
            "IS",
            "NULL",
        ]

        # Extract all words from the query
        words = re.findall(r"\b\w+\b", sql_upper)

        # Check if all non-numeric words are in allowed list or are column/table names
        for word in words:
            if not word.isdigit() and word not in allowed_keywords:
                # Allow alphanumeric words (potential column/table names)
                if not re.match(r"^[A-Z_][A-Z0-9_]*$", word):
                    return False, f"Potentially unsafe keyword: '{word}'"

        # Additional length check (prevent extremely long queries)
        if len(sql_query) > 2000:
            return False, "Query too long (max 2000 characters)"

        return True, "Query is safe"

    def _generate_sql_query(self, question_lower, schema_info):
        """Generate SQL query using LLM with database schema awareness"""
        from langchain_openai import ChatOpenAI
        from langchain.prompts import PromptTemplate
        from django.conf import settings

        # Create schema description for the prompt
        schema_description = self._format_schema_for_prompt(schema_info)

        system_prompt = f"""You are a SQL expert that generates safe, accurate SQL queries based on natural language questions.

DATABASE SCHEMA:
{schema_description}

STRICT SECURITY REQUIREMENTS:
1. Generate ONLY SELECT queries - NEVER use INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, or any other data modification statements
2. Use ONLY tables and columns that exist in the schema above
3. Use proper JOINs for relationships when needed
4. Use appropriate aggregations (COUNT, AVG, SUM, etc.) when requested
5. Return only SELECT statements - no explanations, no comments
6. If the question cannot be answered with the available schema, return "UNABLE_TO_ANSWER"
7. Keep queries simple and safe

QUESTION: {question_lower}

SQL QUERY:"""

        try:
            llm = ChatOpenAI(
                model_name=settings.OPENAI_MODEL,
                openai_api_key=settings.OPENAI_API_KEY,
                temperature=0.1,  # Low temperature for consistent SQL generation
            )

            response = llm.invoke(system_prompt)
            sql_query = response.content.strip()

            # Clean the response - remove markdown code blocks if present
            if sql_query.startswith("```"):
                # Extract SQL from markdown code block
                lines = sql_query.split("\n")
                if (
                    len(lines) >= 3
                    and lines[0].startswith("```")
                    and lines[-1].startswith("```")
                ):
                    sql_query = "\n".join(lines[1:-1]).strip()
                else:
                    # Handle single line with backticks
                    sql_query = (
                        sql_query.replace("```sql", "").replace("```", "").strip()
                    )

            # Enhanced security validation
            if not sql_query or "UNABLE_TO_ANSWER" in sql_query:
                return None

            # Use comprehensive security validation
            is_safe, reason = self._validate_sql_security(sql_query)
            if not is_safe:
                logger.warning(f"Unsafe SQL query rejected: {reason}")
                logger.warning(f"Rejected query: {sql_query}")
                return None

            logger.info(f"Generated safe SQL query: {sql_query}")
            return sql_query

        except Exception as e:
            logger.error(f"Error generating SQL query: {e}")
            return None

    def _format_results_with_llm(self, question, sql_results):
        """Use LLM to format SQL results into natural language response"""
        from langchain_openai import ChatOpenAI
        from django.conf import settings
        import json

        # Format the results as JSON for the LLM
        results_json = json.dumps(
            {
                "columns": sql_results["columns"],
                "rows": sql_results["rows"][
                    :50
                ],  # Limit to first 50 rows to avoid token limits
                "total_rows": sql_results["row_count"],
            },
            indent=2,
            default=str,
        )

        prompt = f"""You are a helpful data analyst. A user asked: "{question}"

Here are the database query results:
{results_json}

Please provide a natural, conversational response that:
1. Directly answers the user's question using the data
2. Presents the information in a clear, readable format
3. Uses appropriate formatting (bold key values, lists where helpful)
4. Includes relevant context and insights if applicable
5. If there are many results, summarize and show key examples
6. Be conversational and helpful, not robotic

Response:"""

        try:
            llm = ChatOpenAI(
                model_name=settings.OPENAI_MODEL,
                openai_api_key=settings.OPENAI_API_KEY,
                temperature=0.3,  # Slightly creative for natural responses
            )

            response = llm.invoke(prompt)
            return response.content.strip()

        except Exception as e:
            logger.error(f"Error formatting results with LLM: {e}")
            # Fallback to simple formatting
            columns = sql_results["columns"]
            rows = sql_results["rows"]

            if len(columns) == 1 and len(rows) == 1:
                value = list(rows[0].values())[0]
                return f"The result is: **{value}**"
            else:
                return f"Found {len(rows)} results with {len(columns)} columns. The data includes: {', '.join(columns[:3])}{'...' if len(columns) > 3 else ''}"

    def _format_schema_for_prompt(self, schema_info):
        """Format schema information for LLM prompt"""
        formatted = []

        for table_name, table_info in schema_info["tables"].items():
            formatted.append(f"TABLE: {table_name} ({table_info['model_name']})")
            formatted.append(f"Description: {table_info['description']}")
            formatted.append("Columns:")

            for col_name, col_info in table_info["fields"].items():
                col_type = col_info["type"]
                nullable = "NULL" if col_info["nullable"] else "NOT NULL"

                if "max_length" in col_info:
                    col_type += f"({col_info['max_length']})"

                if "choices" in col_info:
                    choices_str = ", ".join([f"'{c}'" for c in col_info["choices"][:3]])
                    if len(col_info["choices"]) > 3:
                        choices_str += "..."
                    col_type += f" CHOICES: [{choices_str}]"

                if "foreign_key_to" in col_info:
                    col_type += f" → {col_info['foreign_key_to']}"

                if "array_size" in col_info:
                    col_type += f"[{col_info['array_size']}]"

                formatted.append(f"  - {col_name}: {col_type} {nullable}")

            formatted.append("")

        if schema_info["relationships"]:
            formatted.append("RELATIONSHIPS:")
            for rel in schema_info["relationships"]:
                formatted.append(
                    f"  - {rel['from_table']}.{rel['from_field']} → {rel['to_table']}"
                )

        return "\n".join(formatted)

    def _execute_safe_sql(self, sql_query):
        """Execute SQL query safely and return results with multiple security layers"""
        try:
            # Final security validation before execution
            is_safe, reason = self._validate_sql_security(sql_query)
            if not is_safe:
                logger.error(
                    f"SECURITY VIOLATION: Attempted to execute unsafe query: {reason}"
                )
                logger.error(f"BLOCKED QUERY: {sql_query}")
                return {
                    "error": "Query execution blocked for security reasons",
                    "columns": [],
                    "rows": [],
                    "row_count": 0,
                }

            # Log safe query execution
            logger.info(f"Executing validated SQL query: {sql_query[:100]}...")

            # Set read-only transaction if supported
            with connection.cursor() as cursor:
                # Execute in a transaction that's marked as read-only if possible
                try:
                    cursor.execute("SET TRANSACTION READ ONLY")
                except Exception:
                    # Not all databases support this, continue anyway
                    pass

                cursor.execute(sql_query)
                columns = (
                    [col[0] for col in cursor.description] if cursor.description else []
                )
                rows = cursor.fetchall()

                # Limit result size for safety and performance
                max_rows = 1000  # Reasonable limit for web responses
                if len(rows) > max_rows:
                    logger.warning(
                        f"Query returned {len(rows)} rows, truncating to {max_rows}"
                    )
                    rows = rows[:max_rows]

                # Convert to more readable format
                results = []
                for row in rows:
                    result_dict = {}
                    for i, value in enumerate(row):
                        if i < len(columns):
                            result_dict[columns[i]] = value
                        else:
                            result_dict[f"col_{i}"] = value
                    results.append(result_dict)

                logger.info(
                    f"Query executed successfully, returned {len(results)} rows"
                )
                return {"columns": columns, "rows": results, "row_count": len(results)}

        except Exception as e:
            logger.error(f"Error executing SQL query: {e}")
            logger.error(f"Failed query: {sql_query}")
            return {"error": str(e), "columns": [], "rows": [], "row_count": 0}

    def _analyze_sql_results(self, question_lower, sql_results):
        """Analyze SQL results and generate human-readable answer using LLM"""
        if "error" in sql_results:
            return f"I encountered an error while querying the database: {sql_results['error']}"

        if sql_results["row_count"] == 0:
            return "No data was found matching your query."

        # Use LLM to format the results naturally based on context
        return self._format_results_with_llm(question_lower, sql_results)

    def handle_complex_database_query(self, question):
        """Handle complex database queries using LLM-generated SQL"""
        question_lower = question.lower()

        try:
            # Get database schema
            schema_info = self._get_database_schema()

            # Generate SQL using LLM
            sql_query = self._generate_sql_query(question_lower, schema_info)

            if not sql_query:
                # Fall back to existing statistical methods
                return self._handle_numeric_query(question)

            logger.info(f"Generated SQL: {sql_query}")

            # Execute the query safely
            sql_results = self._execute_safe_sql(sql_query)

            # Analyze and format results
            answer = self._analyze_sql_results(question_lower, sql_results)

            return {
                "answer": answer,
                "data_used": {
                    "type": "llm_generated_sql",
                    "method": "dynamic_query",
                    "sql_query": sql_query,
                    "query_results": sql_results,
                },
                "sources": [],
            }

        except Exception as e:
            logger.error(f"Error in complex database query: {e}")
            # Fall back to existing methods
            return self._handle_numeric_query(question)

    def _handle_explanatory_query(
        self, question: str, context: Dict = None
    ) -> Dict[str, Any]:
        """
        Handle explanatory queries using RAG

        This routes to the existing RAG service for document-based questions
        """
        if not self.rag_service:
            self.rag_service = RAGService()

        # Convert dataset context to RAG format
        # This would need to be implemented based on your data structure
        dataset_chunks = self._convert_dataset_to_chunks(context)

        if dataset_chunks:
            # Add to RAG and query
            vector_ids = self.rag_service.add_documents(
                texts=[chunk["content"] for chunk in dataset_chunks],
                metadatas=[chunk["metadata"] for chunk in dataset_chunks],
            )

            # Query using RAG
            answer = self.rag_service.query(question)

            return {
                "answer": answer,
                "data_used": {"type": "rag_query", "chunks_used": len(dataset_chunks)},
                "vector_ids": vector_ids,
            }
        else:
            return {
                "answer": f"I detected this as an explanatory query: '{question}'. For explanatory questions, I would analyze patterns and provide insights from the data.",
                "data_used": {
                    "type": "explanatory_query",
                    "method": "pattern_analysis",
                },
            }

    def _handle_mixed_query(
        self, question: str, context: Dict = None
    ) -> Dict[str, Any]:
        """
        Handle mixed queries that need both numeric data and explanation

        This combines SQL results with RAG analysis
        """
        # First, get numeric data
        numeric_result = self._handle_numeric_query(question, context)

        # Then, get explanatory analysis
        explanatory_result = self._handle_explanatory_query(
            f"Analyze and explain: {question}. Numeric data: {numeric_result.get('answer', '')}",
            context,
        )

        # Combine results
        combined_answer = (
            f"{numeric_result['answer']}\n\n{explanatory_result['answer']}"
        )

        return {
            "answer": combined_answer,
            "data_used": {
                "numeric_part": numeric_result["data_used"],
                "explanatory_part": explanatory_result["data_used"],
            },
            "combined_analysis": True,
        }

    def _convert_dataset_to_chunks(self, context: Dict = None) -> List[Dict[str, Any]]:
        """
        Convert dataset rows to RAG-compatible chunks

        This is a placeholder - implement based on your data structure
        """
        # Example implementation for audio data
        # You'd query your actual database tables here

        chunks = []

        # Placeholder for demonstration
        if context and "table" in context:
            table_name = context["table"]

            # This would be actual database queries
            if table_name == "audio_recordings":
                chunks.append(
                    {
                        "content": "Sample audio recording data with metadata",
                        "metadata": {"source": "database", "table": table_name},
                    }
                )

        return chunks

    def _generate_sql_placeholder(self, question: str) -> str:
        """Generate placeholder SQL for numeric queries"""
        # This would use LangChain SQL agent in production
        return f"-- SQL query for: {question}\nSELECT COUNT(*) FROM your_table WHERE condition;"

    def get_available_tables(self) -> List[str]:
        """Get list of available database tables"""
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name NOT LIKE 'django_%'
                AND table_name NOT LIKE 'auth_%'
            """
            )
            tables = [row[0] for row in cursor.fetchall()]
        return tables

    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get schema information for a table"""
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = %s
                AND table_schema = 'public'
                ORDER BY ordinal_position
            """,
                [table_name],
            )

            columns = []
            for row in cursor.fetchall():
                columns.append(
                    {"name": row[0], "type": row[1], "nullable": row[2] == "YES"}
                )

            return {
                "table_name": table_name,
                "columns": columns,
                "row_count": self._get_row_count(table_name),
            }

    def _get_row_count(self, table_name: str) -> int:
        """Get approximate row count for a table"""
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
        return count

    def test_sql_security(self):
        """Test the SQL security validation with various queries"""
        test_cases = [
            # Safe queries
            ("SELECT COUNT(*) FROM users", True, "Basic SELECT"),
            (
                "SELECT name, email FROM users WHERE active = 1",
                True,
                "SELECT with WHERE",
            ),
            (
                "SELECT COUNT(*) as total FROM recordings GROUP BY category",
                True,
                "SELECT with GROUP BY",
            ),
            (
                "SELECT u.name, COUNT(r.id) FROM users u LEFT JOIN recordings r ON u.id = r.user_id GROUP BY u.id",
                True,
                "Complex JOIN",
            ),
            # Dangerous queries
            ("DELETE FROM users WHERE id = 1", False, "DELETE statement"),
            ("UPDATE users SET active = 0", False, "UPDATE statement"),
            ("INSERT INTO users (name) VALUES ('test')", False, "INSERT statement"),
            ("DROP TABLE users", False, "DROP statement"),
            ("CREATE TABLE test (id INT)", False, "CREATE statement"),
            ("ALTER TABLE users ADD COLUMN test INT", False, "ALTER statement"),
            ("SELECT * FROM users; DELETE FROM logs", False, "Multiple statements"),
            ("SELECT * FROM users INTO new_table", False, "SELECT INTO"),
            ("EXEC sp_help", False, "Stored procedure"),
            (
                "SELECT * FROM users; SELECT * FROM xp_cmdshell('dir')",
                False,
                "System command",
            ),
            ("SELECT FORMAT(GETDATE(), 'yyyy-MM-dd')", False, "Dangerous function"),
            (
                "SELECT * FROM OPENROWSET('SQLNCLI', 'Server=server;Trusted_Connection=yes', 'SELECT * FROM table')",
                False,
                "OPENROWSET",
            ),
        ]

        results = []
        for query, expected_safe, description in test_cases:
            is_safe, reason = self._validate_sql_security(query)
            passed = is_safe == expected_safe
            results.append(
                {
                    "query": query[:50] + "..." if len(query) > 50 else query,
                    "expected_safe": expected_safe,
                    "actual_safe": is_safe,
                    "reason": reason,
                    "passed": passed,
                    "description": description,
                }
            )

        return results


# Convenience functions
def query_dataset(question: str, context: Dict = None) -> Dict[str, Any]:
    """Quick function to query datasets"""
    service = DatasetService()
    return service.query_dataset(question, context)


def classify_intent(question: str) -> str:
    """Quick function to classify question intent"""
    classifier = IntentClassifier()
    return classifier.classify_intent(question)
