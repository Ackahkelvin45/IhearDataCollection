import logging
import hashlib
import json
import re
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from django.db import connection
from django.db.models import Sum
from django.conf import settings
from django.core.cache import cache

from .intent_classifier import IntentClassifier
from .rag_service import RAGService

logger = logging.getLogger(__name__)

DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 200
NUMERIC_CACHE_TTL_SECONDS = 60 * 5


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

        context = context or {}
        pagination = self._parse_pagination(question, context)
        context.update(pagination)

        # Classify intent
        routing_info = self.intent_classifier.get_routing_info(question)
        intent = routing_info.get("intent", "EXPLANATORY")
        if intent not in ("NUMERIC", "EXPLANATORY", "MIXED"):
            intent = "EXPLANATORY"

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

        # Add metadata (safe access for confidence/reasoning)
        try:
            conf = float(routing_info.get("confidence", 0.8))
            conf = max(0.0, min(conf, 1.0))
        except (TypeError, ValueError):
            conf = 0.8
        result.update(
            {
                "intent": intent,
                "confidence": conf,
                "processing_time": time.time() - start_time,
                "routing_reasoning": routing_info.get("reasoning", "") or "",
                "pagination": pagination,
            }
        )

        return result

    def _get_allowed_tables(self) -> List[str]:
        """Return database tables for data-related Django models."""
        from django.apps import apps

        allowed_tables: List[str] = []
        for app_label in ("data", "core"):
            try:
                app_config = apps.get_app_config(app_label)
            except LookupError:
                continue
            for model in app_config.get_models():
                allowed_tables.append(model._meta.db_table)

        # De-dupe and stable sort
        return sorted(set(allowed_tables))

    def _parse_pagination(self, question: str, context: Dict[str, Any]) -> Dict[str, int]:
        """Parse pagination from context or question text."""
        question_lower = (question or "").lower()

        def _clamp_page_size(value: int) -> int:
            return max(1, min(int(value), MAX_PAGE_SIZE))

        page_size = context.get("page_size") or context.get("limit") or DEFAULT_PAGE_SIZE
        try:
            page_size = _clamp_page_size(int(page_size))
        except (TypeError, ValueError):
            page_size = DEFAULT_PAGE_SIZE

        page = context.get("page") or 1
        try:
            page = max(1, int(page))
        except (TypeError, ValueError):
            page = 1

        offset = context.get("offset") or 0
        try:
            offset = max(0, int(offset))
        except (TypeError, ValueError):
            offset = 0

        # Parse hints from question text
        m = re.search(r"(?:per page|page size|pagesize|limit)\s*(\d{1,4})", question_lower)
        if m:
            page_size = _clamp_page_size(int(m.group(1)))

        m = re.search(r"\bpage\s+(\d{1,4})\b", question_lower)
        if m:
            page = max(1, int(m.group(1)))

        m = re.search(r"\boffset\s+(\d{1,6})\b", question_lower)
        if m:
            offset = max(0, int(m.group(1)))

        if offset == 0 and page > 1:
            offset = (page - 1) * page_size

        return {"page": page, "page_size": page_size, "offset": offset}

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
        context = context or {}
        page_size = int(context.get("page_size", DEFAULT_PAGE_SIZE))
        offset = int(context.get("offset", 0))

        cache_key = f"chatbot:numeric:{hashlib.md5(f'{question_lower}|{page_size}|{offset}'.encode()).hexdigest()}"
        cached = cache.get(cache_key)
        if cached:
            return cached

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
            "datsets",  # Common typo
            "datset",   # Common typo
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
        # More flexible matching - check if any dataset keyword appears
        has_dataset = any(keyword in question_lower for keyword in dataset_keywords) or "dataset" in question_lower

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

        # Fast path for common aggregates (no LLM/tool call)
        if has_counting and (has_dataset or "dataset" in question_lower or "datasets" in question_lower or "datsets" in question_lower):
            try:
                result = self._handle_counting_questions(question_lower)
                cache.set(cache_key, result, NUMERIC_CACHE_TTL_SECONDS)
                return result
            except Exception as e:
                logger.error(f"Error in counting query: {e}", exc_info=True)

        if (has_statistics and has_dataset) or has_audio_features:
            try:
                result = self._handle_statistical_questions(
                    question_lower, has_audio_features
                )
                cache.set(cache_key, result, NUMERIC_CACHE_TTL_SECONDS)
                return result
            except Exception as e:
                logger.error(f"Error in statistical query: {e}", exc_info=True)

        # Prefer tool-based SQL agent for numeric/DB questions
        if context.get("use_sql_tool", True):
            tool_result = self._handle_numeric_query_with_tool(
                question, {"page_size": page_size, "offset": offset}
            )
            if tool_result:
                cache.set(cache_key, tool_result, NUMERIC_CACHE_TTL_SECONDS)
                return tool_result

        # Check if this is a counting question about datasets (most common case)
        if has_counting and (has_dataset or "dataset" in question_lower or "datasets" in question_lower or "datsets" in question_lower):
            try:
                from data.models import (
                    Dataset as AudioDataset,
                    NoiseDataset,
                    AudioFeature,
                    NoiseAnalysis,
                )

                # For "how many datasets" questions, always query the database
                result = self._handle_counting_questions(question_lower)
                cache.set(cache_key, result, NUMERIC_CACHE_TTL_SECONDS)
                return result

            except Exception as e:
                logger.error(f"Error in counting query: {e}", exc_info=True)
                return {
                    "answer": f"I encountered an error while checking your datasets: {str(e)}. Please try again or contact support.",
                    "data_used": {"type": "error", "error": str(e)},
                    "sources": [],
                }

        # Handle statistical/feature questions
        if (has_statistics and has_dataset) or has_audio_features:
            try:
                from data.models import (
                    Dataset as AudioDataset,
                    NoiseDataset,
                    AudioFeature,
                    NoiseAnalysis,
                )

                # First try LLM-powered SQL generation for complex queries
                # Check if this is a complex computational question
                is_complex_query = self._is_complex_computational_query(question_lower)
                
                if is_complex_query or len(question_lower.split()) > 3:
                    llm_result = self.handle_complex_database_query(question)
                    if (
                        llm_result
                        and llm_result.get("data_used", {}).get("type") == "llm_generated_sql"
                    ):
                        return llm_result

                # Fall back to rule-based statistical/feature analysis
                result = self._handle_statistical_questions(
                    question_lower, has_audio_features
                )
                cache.set(cache_key, result, NUMERIC_CACHE_TTL_SECONDS)
                return result

            except Exception as e:
                logger.error(f"Error in statistical query: {e}", exc_info=True)
                return {
                    "answer": f"I encountered an error while analyzing your data: {str(e)}. Please try again.",
                    "data_used": {"type": "error", "error": str(e)},
                    "sources": [],
                }

        # For other numeric questions, try to query anyway if it's a counting question
        if has_counting:
            try:
                result = self._handle_counting_questions(question_lower)
                cache.set(cache_key, result, NUMERIC_CACHE_TTL_SECONDS)
                return result
            except Exception as e:
                logger.debug(f"Could not handle counting question: {e}")

        # Try complex query handler as last resort for any numeric/computational question
        try:
            llm_result = self.handle_complex_database_query(question)
            if (
                llm_result
                and llm_result.get("data_used", {}).get("type") == "llm_generated_sql"
            ):
                cache.set(cache_key, llm_result, NUMERIC_CACHE_TTL_SECONDS)
                return llm_result
        except Exception as e:
            logger.debug(f"Complex query handler failed: {e}")

        # Last resort: generic response
        result = {
            "answer": f"I detected this as a numeric/computational query: '{question}'. I'm attempting to query the database to get accurate results for you.",
            "data_used": {"type": "numeric_query", "method": "database"},
            "sql_equivalent": self._generate_sql_placeholder(question),
        }
        cache.set(cache_key, result, NUMERIC_CACHE_TTL_SECONDS)
        return result

    def _handle_numeric_query_with_tool(self, question: str, pagination: Dict[str, int]) -> Optional[Dict[str, Any]]:
        """
        Use the tool-calling SQL agent for numeric/database questions.
        Returns None on failure so caller can fall back to heuristic handling.
        """
        try:
            from data_insights.workflows.sql_agent import TextToSQLAgent
            from data_insights.workflows.prompt import SQL_SYSTEM_TEMPLATE
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage
            from django.conf import settings

            allowed_tables = self._get_allowed_tables()
            if not allowed_tables:
                logger.warning("No allowed tables found for SQL tool agent")
                return None

            llm = ChatOpenAI(
                model_name=settings.OPENAI_MODEL,
                openai_api_key=settings.OPENAI_API_KEY,
                temperature=0.1,
                max_tokens=1500,
            )

            agent = TextToSQLAgent(
                llm=llm,
                system_prompt=SQL_SYSTEM_TEMPLATE,
                include_tables=allowed_tables,
                ai_answer=True,
                top_k=pagination.get("page_size", DEFAULT_PAGE_SIZE),
                default_offset=pagination.get("offset", 0),
            )

            workflow = agent.compile_workflow()
            result = workflow.invoke(
                {
                    "messages": [HumanMessage(content=question)],
                    "n_trials": 0,
                }
            )

            if result and "messages" in result:
                table = None
                try:
                    from langchain_core.messages import ToolMessage
                    for msg in reversed(result["messages"]):
                        if isinstance(msg, ToolMessage):
                            table = self._parse_tool_table(msg.content, pagination)
                            break
                except Exception:
                    table = None

                last_message = result["messages"][-1]
                if hasattr(last_message, "content"):
                    answer = last_message.content
                    return {
                        "answer": answer,
                        "data_used": {
                            "type": "tool_sql_agent",
                            "method": "text_to_sql_agent",
                            "allowed_tables": allowed_tables,
                        },
                        "table": table,
                        "sources": [],
                    }

            return None

        except Exception as e:
            logger.warning(f"SQL tool agent failed, falling back: {e}")
            return None

    def _parse_tool_table(self, content: Any, pagination: Dict[str, int]) -> Optional[Dict[str, Any]]:
        """Parse tool output into table data for UI."""
        if content is None:
            return None

        data = None
        if isinstance(content, list):
            data = content
        elif isinstance(content, dict):
            data = [content]
        elif isinstance(content, str):
            if content.lower().startswith("error"):
                return None
            try:
                data = json.loads(content)
            except Exception:
                return None

        if not isinstance(data, list):
            return None

        rows = [row for row in data if isinstance(row, dict)]
        columns = list(rows[0].keys()) if rows else []
        page_size = int(pagination.get("page_size", DEFAULT_PAGE_SIZE))
        page = int(pagination.get("page", 1))
        offset = int(pagination.get("offset", 0))
        row_count = len(rows)
        has_more = row_count >= page_size

        return {
            "columns": columns,
            "rows": rows,
            "page": page,
            "page_size": page_size,
            "offset": offset,
            "row_count": row_count,
            "has_more": has_more,
        }
    
    def _is_complex_computational_query(self, question_lower):
        """Detect if a question requires complex computational analysis"""
        import re
        
        # Keywords that indicate complex computations
        complex_keywords = [
            # Aggregations and statistics
            "average", "mean", "median", "sum", "total", "aggregate", "statistics",
            "distribution", "percentile", "quartile", "standard deviation", "variance",
            # Comparisons
            "compare", "comparison", "versus", "vs", "difference", "ratio", "percentage",
            # Grouping and categorization
            "group by", "grouped", "by category", "by region", "by type", "breakdown",
            "distribution", "split", "categorized",
            # Trends and patterns
            "trend", "pattern", "over time", "per month", "per year", "per day",
            "growth", "increase", "decrease", "change",
            # Complex filters
            "where", "filter", "filtered", "excluding", "including", "between",
            "greater than", "less than", "above", "below",
            # Joins and relationships
            "with", "including", "related", "associated", "linked",
            # Ranking and sorting
            "top", "bottom", "highest", "lowest", "rank", "ranking", "sorted",
            # Calculations
            "calculate", "computation", "compute", "formula", "equation",
            # Complex questions
            "how many", "what is", "which", "show me", "list", "find",
            # Multiple conditions
            "and", "or", "both", "either", "neither",
        ]
        
        # Check for multiple complex keywords (indicates complexity)
        keyword_count = sum(1 for keyword in complex_keywords if keyword in question_lower)
        
        # Check for mathematical operations or comparisons
        has_math = bool(re.search(r'\d+\s*[+\-*/]\s*\d+', question_lower))
        has_comparison = bool(re.search(r'(greater|less|more|fewer|higher|lower|above|below)\s+than', question_lower))
        
        # Check for aggregation functions
        has_aggregation = bool(re.search(r'\b(count|sum|avg|average|max|min|total)\b', question_lower))
        
        # Check for grouping indicators
        has_grouping = bool(re.search(r'\b(by|group|per|each|for each)\b', question_lower))
        
        # Complex if:
        # - Has 2+ complex keywords, OR
        # - Has math/comparison + aggregation, OR
        # - Has grouping + aggregation, OR
        # - Question is longer than 8 words (likely complex)
        is_complex = (
            keyword_count >= 2 or
            (has_math or has_comparison) and has_aggregation or
            has_grouping and has_aggregation or
            len(question_lower.split()) > 8
        )
        
        return is_complex

    def _handle_counting_questions(self, question: str = ""):
        """Handle basic counting questions about datasets"""
        from data.models import Dataset as AudioDataset, NoiseDataset
        from chatbot.models import Document
        from core.models import Category
        from django.db.models import Sum

        question_lower = question.lower() if question else ""
        
        # Determine what the user is asking about
        asking_about_noise = any(word in question_lower for word in ["noise", "recording", "audio", "sound"])
        asking_about_docs = any(word in question_lower for word in ["document", "doc", "file", "upload"])
        asking_about_categories = any(word in question_lower for word in ["category", "categories"])
        asking_about_all = not asking_about_noise and not asking_about_docs

        if asking_about_categories:
            total_categories = Category.objects.count()
            used_categories = NoiseDataset.objects.values("category_id").distinct().count()

            answer = (
                f"You have **{total_categories}** total categories, "
                f"with **{used_categories}** categories currently used in noise datasets."
            )
            table = {
                "columns": ["metric", "value"],
                "rows": [
                    {"metric": "total_categories", "value": total_categories},
                    {"metric": "used_categories_in_noise", "value": used_categories},
                ],
                "page": 1,
                "page_size": 2,
                "offset": 0,
                "row_count": 2,
                "has_more": False,
            }
            return {
                "answer": answer,
                "table": table,
                "data_used": {
                    "type": "database_query",
                    "method": "category_count",
                    "total_categories": total_categories,
                    "used_categories": used_categories,
                },
                "sources": [],
            }

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

        # 2. Audio/noise datasets (data collection datasets)
        total_audio_datasets = AudioDataset.objects.count()
        total_noise_records = NoiseDataset.objects.count()

        # Build answer based on what user is asking about
        answer_parts = []

        if asking_about_noise or asking_about_all:
            # User is asking about noise/audio datasets or all datasets
            if total_noise_records > 0:
                answer_parts.append(f"You have **{total_noise_records:,}** noise datasets available.")
                if total_audio_datasets > 0:
                    answer_parts.append(f"These are organized into {total_audio_datasets} dataset types.")
            elif asking_about_noise:
                answer_parts.append("You don't have any noise datasets yet. You can start collecting audio data through the data collection interface.")

        if asking_about_docs or asking_about_all:
            # User is asking about documents or all datasets
            if total_docs > 0:
                doc_info = f"You have **{total_docs}** document(s) uploaded for chatbot analysis"
                if processed_docs > 0:
                    doc_info += f" ({processed_docs} processed with {total_chunks:,} text chunks"
                if unprocessed_docs > 0:
                    doc_info += f", {unprocessed_docs} still processing"
                doc_info += ")."
                answer_parts.append(doc_info)
            elif asking_about_docs:
                answer_parts.append("You don't have any documents uploaded yet. You can upload documents through the chatbot interface.")

        # If asking about all and no datasets found
        if asking_about_all and total_docs == 0 and total_audio_datasets == 0 and total_noise_records == 0:
            answer = "You don't have any datasets yet. You can upload documents for chatbot analysis or start collecting audio data through the data collection interface."
        elif answer_parts:
            answer = " ".join(answer_parts)
        else:
            # Fallback
            total_all = total_noise_records + total_docs
            if total_all > 0:
                answer = f"You have **{total_all:,}** total datasets ({total_noise_records:,} noise datasets and {total_docs} documents)."
            else:
                answer = "You don't have any datasets yet."

        table_rows = [
            {"metric": "noise_records", "value": total_noise_records},
            {"metric": "audio_dataset_types", "value": total_audio_datasets},
            {"metric": "uploaded_documents", "value": total_docs},
            {"metric": "processed_documents", "value": processed_docs},
            {"metric": "total_text_chunks", "value": total_chunks},
        ]

        table = {
            "columns": ["metric", "value"],
            "rows": table_rows,
            "page": 1,
            "page_size": len(table_rows),
            "offset": 0,
            "row_count": len(table_rows),
            "has_more": False,
        }

        return {
            "answer": answer,
            "table": table,
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

        # Allow only safe clauses and keywords (expanded for complex queries)
        allowed_keywords = [
            # Basic SQL
            "SELECT", "FROM", "WHERE", "AS", "AND", "OR", "NOT", "IN", "EXISTS", "BETWEEN", "LIKE", "ILIKE",
            # Joins
            "JOIN", "INNER", "LEFT", "RIGHT", "FULL", "OUTER", "ON", "USING",
            # Grouping and ordering
            "GROUP", "BY", "HAVING", "ORDER", "ASC", "DESC", "NULLS", "FIRST", "LAST",
            # Aggregations
            "COUNT", "SUM", "AVG", "MAX", "MIN", "STDDEV", "STDDEV_POP", "STDDEV_SAMP", 
            "VARIANCE", "VAR_POP", "VAR_SAMP", "ARRAY_AGG", "STRING_AGG",
            # Conditional logic
            "CASE", "WHEN", "THEN", "ELSE", "END", "COALESCE", "NULLIF", "IS", "NULL",
            # Set operations
            "UNION", "INTERSECT", "EXCEPT", "ALL",
            # CTEs and subqueries
            "WITH", "RECURSIVE",
            # Window functions
            "OVER", "PARTITION", "ROWS", "RANGE", "UNBOUNDED", "PRECEDING", "FOLLOWING", "CURRENT", "ROW",
            "ROW_NUMBER", "RANK", "DENSE_RANK", "LAG", "LEAD", "FIRST_VALUE", "LAST_VALUE",
            # Limits and distinct
            "LIMIT", "OFFSET", "DISTINCT", "ON",
            # String functions
            "CONCAT", "SUBSTRING", "UPPER", "LOWER", "TRIM", "LTRIM", "RTRIM", "LENGTH", "POSITION",
            # Date/time functions
            "DATE_TRUNC", "EXTRACT", "AGE", "NOW", "CURRENT_DATE", "CURRENT_TIME", "CURRENT_TIMESTAMP",
            # Math functions
            "ABS", "ROUND", "FLOOR", "CEIL", "CEILING", "TRUNC", "MOD", "POWER", "SQRT",
            # Type casting
            "CAST", "::",
            # Comparison
            "=", "!=", "<>", "<", ">", "<=", ">=",
        ]

        # Extract all words from the query (but be more lenient for complex queries)
        words = re.findall(r"\b\w+\b", sql_upper)

        # Check if all non-numeric words are in allowed list or are column/table names
        # Allow common table/column naming patterns
        for word in words:
            if not word.isdigit() and word not in allowed_keywords:
                # Allow alphanumeric words (potential column/table names)
                # Also allow common SQL function names that might be in lowercase
                if not re.match(r"^[A-Z_][A-Z0-9_]*$", word):
                    # Check if it's a common PostgreSQL function (case-insensitive)
                    common_functions = [
                        "DATE_TRUNC", "EXTRACT", "AGE", "NOW", "CURRENT_DATE", 
                        "CURRENT_TIME", "CURRENT_TIMESTAMP", "CONCAT", "SUBSTRING",
                        "UPPER", "LOWER", "TRIM", "LENGTH", "POSITION", "ABS",
                        "ROUND", "FLOOR", "CEIL", "CEILING", "TRUNC", "MOD", 
                        "POWER", "SQRT", "ROW_NUMBER", "RANK", "DENSE_RANK",
                        "LAG", "LEAD", "FIRST_VALUE", "LAST_VALUE", "ARRAY_AGG",
                        "STRING_AGG", "STDDEV_POP", "STDDEV_SAMP", "VAR_POP", "VAR_SAMP"
                    ]
                    if word.upper() not in common_functions:
                        # Still allow if it looks like a valid identifier (table/column name)
                        if not re.match(r"^[A-Z0-9_]+$", word):
                            return False, f"Potentially unsafe keyword: '{word}'"

        # Additional length check (allow longer queries for complex computations)
        if len(sql_query) > 5000:
            return False, "Query too long (max 5000 characters)"

        return True, "Query is safe"

    def _generate_sql_query_enhanced(self, question, schema_info):
        """Generate complex SQL queries using LLM with enhanced capabilities"""
        from langchain_openai import ChatOpenAI
        from django.conf import settings

        question_lower = question.lower()
        schema_description = self._format_schema_for_prompt(schema_info)

        # Enhanced prompt for complex queries
        system_prompt = f"""You are an expert SQL query generator specializing in complex data analysis queries for audio/noise datasets.

DATABASE SCHEMA:
{schema_description}

CAPABILITIES - You can generate:
1. **Aggregations**: COUNT, SUM, AVG, MIN, MAX, STDDEV, VARIANCE
2. **Grouping**: GROUP BY with multiple columns, HAVING clauses
3. **Joins**: INNER, LEFT, RIGHT joins across related tables
4. **Subqueries**: Correlated and non-correlated subqueries
5. **CTEs (Common Table Expressions)**: WITH clauses for complex multi-step queries
6. **Window Functions**: ROW_NUMBER(), RANK(), DENSE_RANK(), LAG(), LEAD(), SUM() OVER()
7. **Filtering**: WHERE with multiple conditions, IN, EXISTS, BETWEEN, LIKE, ILIKE
8. **Sorting**: ORDER BY with multiple columns, NULLS FIRST/LAST
9. **Set Operations**: UNION, INTERSECT, EXCEPT (when safe)
10. **Date/Time Functions**: DATE_TRUNC, EXTRACT, AGE, etc.
11. **String Functions**: CONCAT, SUBSTRING, UPPER, LOWER, TRIM
12. **Case Statements**: CASE WHEN for conditional logic

STRICT SECURITY REQUIREMENTS:
1. Generate ONLY SELECT queries - NEVER use INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, TRUNCATE, or any data modification
2. Use ONLY tables and columns that exist in the schema above
3. Use proper JOINs for relationships (data_noisedataset.id = data_audiofeature.noise_dataset_id, etc.)
4. For complex questions, use CTEs (WITH clauses) to break down the logic
5. Use appropriate aggregations and groupings based on the question
6. Return ONLY the SQL query - no explanations, no markdown, no comments
7. If the question cannot be answered with the available schema, return "UNABLE_TO_ANSWER"
8. Always use table aliases for readability (nd, af, na, cat, reg, etc.)
9. Limit results to reasonable sizes (use LIMIT when appropriate)

QUESTION: {question}

SQL QUERY:"""

        try:
            llm = ChatOpenAI(
                model_name=settings.OPENAI_MODEL,
                openai_api_key=settings.OPENAI_API_KEY,
                temperature=0.1,  # Low temperature for consistent SQL generation
                max_tokens=2000,  # Allow longer queries for complex computations
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

            # Remove any leading/trailing whitespace and newlines
            sql_query = sql_query.strip()

            # Enhanced security validation
            if not sql_query or "UNABLE_TO_ANSWER" in sql_query.upper():
                logger.warning("LLM returned UNABLE_TO_ANSWER or empty query")
                return None

            # Use comprehensive security validation
            is_safe, reason = self._validate_sql_security(sql_query)
            if not is_safe:
                logger.warning(f"Unsafe SQL query rejected: {reason}")
                logger.warning(f"Rejected query: {sql_query[:200]}...")
                return None

            logger.info(f"Generated safe SQL query: {sql_query[:100]}...")
            return sql_query

        except Exception as e:
            logger.error(f"Error generating SQL query: {e}", exc_info=True)
            return None

    def _format_results_with_llm_enhanced(self, question, sql_results):
        """Enhanced LLM formatting with better analysis and insights"""
        from langchain_openai import ChatOpenAI
        from django.conf import settings
        import json

        # Format the results as JSON for the LLM
        # Include more rows for complex analysis (up to 100)
        max_rows_to_show = min(100, sql_results["row_count"])
        
        results_json = json.dumps(
            {
                "columns": sql_results["columns"],
                "rows": sql_results["rows"][:max_rows_to_show],
                "total_rows": sql_results["row_count"],
                "showing_rows": max_rows_to_show,
            },
            indent=2,
            default=str,
        )

        prompt = f"""You are an expert data analyst helping users understand their audio/noise dataset analysis results.

USER QUESTION: "{question}"

QUERY RESULTS:
{results_json}

Please provide a comprehensive, natural response that:
1. **Directly answers** the user's question using the actual data from the results
2. **Highlights key findings** - identify patterns, trends, outliers, or interesting insights
3. **Uses clear formatting**:
   - Use **bold** for important numbers and key values
   - Use bullet points (•) for lists
   - Use tables or structured formatting when showing multiple values
   - Use markdown formatting appropriately
4. **Provides context**: Explain what the numbers mean in practical terms
5. **Handles large datasets**: If there are many results, summarize key patterns and show representative examples
6. **Statistical insights**: If appropriate, mention averages, ranges, distributions, or comparisons
7. **Be conversational**: Write naturally, as if explaining to a colleague, not robotically

IMPORTANT:
- Use the actual data values from the results
- Don't make up numbers or statistics
- If the question asks for computation/analysis, show the computed results clearly
- For complex queries, break down the answer into clear sections if helpful

Response:"""

        try:
            llm = ChatOpenAI(
                model_name=settings.OPENAI_MODEL,
                openai_api_key=settings.OPENAI_API_KEY,
                temperature=0.3,  # Slightly creative for natural responses
                max_tokens=2000,  # Allow longer responses for complex analysis
            )

            response = llm.invoke(prompt)
            formatted_answer = response.content.strip()

            # Add a note if results were truncated
            if sql_results["row_count"] > max_rows_to_show:
                formatted_answer += f"\n\n*Note: Showing {max_rows_to_show} of {sql_results['row_count']} total results.*"

            return formatted_answer

        except Exception as e:
            logger.error(f"Error formatting results with LLM: {e}")
            # Fallback to basic formatting
            return self._format_results_basic(question, sql_results)
    
    def _format_results_basic(self, question, sql_results):
        """Basic fallback formatting without LLM"""
        if sql_results["row_count"] == 0:
            return "No results found."
        
        answer = f"Found {sql_results['row_count']} result(s):\n\n"
        
        # Show column headers
        if sql_results["columns"]:
            answer += " | ".join(sql_results["columns"]) + "\n"
            answer += "-" * (len(" | ".join(sql_results["columns"]))) + "\n"
        
        # Show first 10 rows
        for i, row in enumerate(sql_results["rows"][:10]):
            if isinstance(row, dict):
                values = [str(row.get(col, "")) for col in sql_results["columns"]]
            else:
                values = [str(v) for v in row]
            answer += " | ".join(values) + "\n"
        
        if sql_results["row_count"] > 10:
            answer += f"\n... and {sql_results['row_count'] - 10} more results."
        
        return answer

    def _format_results_with_llm(self, question, sql_results):
        """Use LLM to format SQL results into natural language response"""
        return self._format_results_with_llm_enhanced(question, sql_results)

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
        """Format schema information for LLM prompt with enhanced context"""
        formatted = []
        
        # Add header with key information
        formatted.append("=" * 80)
        formatted.append("DATABASE SCHEMA FOR AUDIO/NOISE DATASET ANALYSIS")
        formatted.append("=" * 80)
        formatted.append("")

        # Group tables by domain for better understanding
        core_tables = []
        data_tables = []
        chatbot_tables = []
        
        for table_name in schema_info["tables"].keys():
            if table_name.startswith("core_"):
                core_tables.append(table_name)
            elif table_name.startswith("data_"):
                data_tables.append(table_name)
            elif table_name.startswith("chatbot_"):
                chatbot_tables.append(table_name)
            else:
                data_tables.append(table_name)

        # Format core reference tables first
        if core_tables:
            formatted.append("📋 CORE REFERENCE TABLES (Categories, Regions, Communities, etc.):")
            formatted.append("")
            for table_name in sorted(core_tables):
                table_info = schema_info["tables"][table_name]
                formatted.extend(self._format_table_info(table_name, table_info))
                formatted.append("")

        # Format main data tables
        if data_tables:
            formatted.append("📊 MAIN DATA TABLES (Noise Datasets, Audio Features, Analysis):")
            formatted.append("")
            for table_name in sorted(data_tables):
                table_info = schema_info["tables"][table_name]
                formatted.extend(self._format_table_info(table_name, table_info))
                formatted.append("")

        # Format chatbot tables
        if chatbot_tables:
            formatted.append("💬 CHATBOT TABLES (Documents, Messages):")
            formatted.append("")
            for table_name in sorted(chatbot_tables):
                table_info = schema_info["tables"][table_name]
                formatted.extend(self._format_table_info(table_name, table_info))
                formatted.append("")

        # Add relationship information with more context
        if schema_info["relationships"]:
            formatted.append("=" * 80)
            formatted.append("KEY RELATIONSHIPS (for JOINs):")
            formatted.append("=" * 80)
            formatted.append("")
            
            # Group relationships by table
            rels_by_table = {}
            for rel in schema_info["relationships"]:
                from_table = rel['from_table']
                if from_table not in rels_by_table:
                    rels_by_table[from_table] = []
                rels_by_table[from_table].append(rel)
            
            for table_name in sorted(rels_by_table.keys()):
                formatted.append(f"Table: {table_name}")
                for rel in rels_by_table[table_name]:
                    formatted.append(
                        f"  • {rel['from_field']} → {rel['to_table']} (Foreign Key)"
                    )
                formatted.append("")

        # Add common query patterns
        formatted.append("=" * 80)
        formatted.append("COMMON QUERY PATTERNS:")
        formatted.append("=" * 80)
        formatted.append("")
        formatted.append("1. Join noise datasets with audio features:")
        formatted.append("   data_noisedataset nd JOIN data_audiofeature af ON nd.id = af.noise_dataset_id")
        formatted.append("")
        formatted.append("2. Join with categories:")
        formatted.append("   data_noisedataset nd JOIN core_category cat ON nd.category_id = cat.id")
        formatted.append("")
        formatted.append("3. Join with regions:")
        formatted.append("   data_noisedataset nd JOIN core_region reg ON nd.region_id = reg.id")
        formatted.append("")
        formatted.append("4. Aggregate by category:")
        formatted.append("   SELECT cat.name, COUNT(*) FROM data_noisedataset nd")
        formatted.append("   JOIN core_category cat ON nd.category_id = cat.id GROUP BY cat.name")
        formatted.append("")

        return "\n".join(formatted)
    
    def _format_table_info(self, table_name, table_info):
        """Format individual table information"""
        formatted = []
        formatted.append(f"TABLE: {table_name}")
        formatted.append(f"Model: {table_info['model_name']}")
        if table_info.get('description'):
            formatted.append(f"Description: {table_info['description']}")
        formatted.append("Columns:")
        
        # Sort columns: primary keys first, then foreign keys, then others
        pk_cols = []
        fk_cols = []
        other_cols = []
        
        for col_name, col_info in table_info["fields"].items():
            if "id" in col_name.lower() and col_name.lower().endswith("id"):
                if col_name.lower() == "id":
                    pk_cols.append((col_name, col_info))
                else:
                    fk_cols.append((col_name, col_info))
            else:
                other_cols.append((col_name, col_info))
        
        all_cols = pk_cols + fk_cols + other_cols
        
        for col_name, col_info in all_cols:
            col_type = col_info["type"]
            nullable = "NULL" if col_info["nullable"] else "NOT NULL"
            
            # Add type details
            if "max_length" in col_info:
                col_type += f"({col_info['max_length']})"
            
            if "choices" in col_info:
                choices_str = ", ".join([f"'{c}'" for c in col_info["choices"][:5]])
                if len(col_info["choices"]) > 5:
                    choices_str += f" ... ({len(col_info['choices'])} total)"
                col_type += f" CHOICES: [{choices_str}]"
            
            # Mark foreign keys clearly
            if "foreign_key_to" in col_info:
                col_type += f" → FK to {col_info['foreign_key_to']}"
            
            if "array_size" in col_info:
                col_type += f"[{col_info['array_size']}]"
            
            # Mark primary keys
            if col_name.lower() == "id":
                formatted.append(f"  - {col_name}: {col_type} {nullable} [PRIMARY KEY]")
            else:
                formatted.append(f"  - {col_name}: {col_type} {nullable}")
        
        return formatted

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

    def _analyze_sql_results_enhanced(self, question, sql_results):
        """Enhanced analysis of SQL results with better formatting and insights"""
        if "error" in sql_results:
            return f"I encountered an error while querying the database: {sql_results['error']}. Please try rephrasing your question."

        if sql_results["row_count"] == 0:
            return "No data was found matching your query. You might want to check your filters or try a different question."

        # Use enhanced LLM formatting with better context
        return self._format_results_with_llm_enhanced(question, sql_results)
    
    def _analyze_sql_results(self, question_lower, sql_results):
        """Analyze SQL results and generate human-readable answer using LLM"""
        return self._analyze_sql_results_enhanced(question_lower, sql_results)

    def handle_complex_database_query(self, question):
        """Handle complex database queries using LLM-generated SQL with advanced capabilities"""
        question_lower = question.lower()

        try:
            # First, try using the sophisticated TextToSQLAgent from data_insights if available
            try:
                from data_insights.workflows.sql_agent import TextToSQLAgent
                from data_insights.workflows.prompt import SQL_SYSTEM_TEMPLATE
                from langchain_openai import ChatOpenAI
                from django.conf import settings
                
                # Define allowed tables for complex queries (data + core models)
                allowed_tables = self._get_allowed_tables()
                
                # Create TextToSQLAgent for sophisticated SQL generation
                llm = ChatOpenAI(
                    model_name=settings.OPENAI_MODEL,
                    openai_api_key=settings.OPENAI_API_KEY,
                    temperature=0.1,
                )
                
                agent = TextToSQLAgent(
                    llm=llm,
                    system_prompt=SQL_SYSTEM_TEMPLATE,
                    include_tables=allowed_tables,
                    ai_answer=True,  # Let LLM format the answer
                    top_k=100,
                )
                
                # Compile workflow
                workflow = agent.compile_workflow()
                
                # Execute query using the agent
                from data_insights.workflows.schema import PostgresSQLInput
                from langchain_core.messages import HumanMessage
                
                initial_state = {
                    "messages": [HumanMessage(content=question)],
                    "n_trials": 0,
                }
                
                result = workflow.invoke(initial_state)
                
                # Extract answer from agent response
                if result and "messages" in result:
                    last_message = result["messages"][-1]
                    if hasattr(last_message, "content"):
                        answer = last_message.content
                        
                        return {
                            "answer": answer,
                            "data_used": {
                                "type": "llm_generated_sql",
                                "method": "text_to_sql_agent",
                                "agent_used": True,
                            },
                            "sources": [],
                        }
            except ImportError:
                logger.debug("TextToSQLAgent not available, using basic SQL generation")
            except Exception as agent_error:
                logger.warning(f"TextToSQLAgent failed, falling back to basic SQL: {agent_error}")

            # Fallback to enhanced basic SQL generation
            schema_info = self._get_database_schema()

            # Generate SQL using enhanced LLM prompt
            sql_query = self._generate_sql_query_enhanced(question, schema_info)

            if not sql_query:
                # Fall back to existing statistical methods
                return self._handle_numeric_query(question)

            logger.info(f"Generated SQL: {sql_query}")

            # Execute the query safely
            sql_results = self._execute_safe_sql(sql_query)

            # Analyze and format results with enhanced analysis
            answer = self._analyze_sql_results_enhanced(question, sql_results)

            return {
                "answer": answer,
                "data_used": {
                    "type": "llm_generated_sql",
                    "method": "enhanced_dynamic_query",
                    "sql_query": sql_query,
                    "query_results": sql_results,
                },
                "sources": [],
            }

        except Exception as e:
            logger.error(f"Error in complex database query: {e}", exc_info=True)
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
            self.rag_service.add_documents(
                texts=[chunk["content"] for chunk in dataset_chunks],
                metadatas=[chunk["metadata"] for chunk in dataset_chunks],
            )

            # Query using RAG
            rag_result = self.rag_service.query(question)

            return {
                "answer": rag_result.get("answer", ""),
                "sources": rag_result.get("sources", []),
                "data_used": {"type": "rag_query", "chunks_used": len(dataset_chunks)},
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
