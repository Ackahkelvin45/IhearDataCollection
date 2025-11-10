from django.shortcuts import render, redirect
from .forms import NoiseDatasetForm, BulkAudioUploadForm
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from core.models import Class, SubClass, Community, Category, Region
from django.shortcuts import get_object_or_404
import plotly.graph_objects as go
import numpy as np
import json
from django.db.models import Count, Avg, Sum
from django.utils import timezone
from django.contrib.auth.mixins import LoginRequiredMixin
import hashlib
from django.views.generic import DeleteView, ListView
from django.urls import reverse_lazy
from django.contrib import messages
from .models import NoiseDataset, AudioFeature, NoiseAnalysis
from datetime import timedelta
from django.db.models import Q
from .models import BulkAudioUpload
from .tasks import process_bulk_upload
import logging
import os
from .utils import generate_dataset_name, generate_noise_id
import glob
from django.conf import settings
from plotly.utils import PlotlyJSONEncoder
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from rest_framework.pagination import PageNumberPagination
from .serializers import NoiseDatasetSerializer
import math
import csv
from io import BytesIO, StringIO
from django.http import HttpResponse, StreamingHttpResponse
from django.utils.encoding import smart_str

try:
    import openpyxl
except Exception:
    openpyxl = None

logger = logging.getLogger(__name__)


class RenamedFile:
    """Wrapper class to rename an uploaded file without changing its content"""

    def __init__(self, file, new_name):
        self.file = file
        self.name = new_name
        self._name = new_name

    def __getattr__(self, attr):
        return getattr(self.file, attr)


@login_required
def view_dashboard(request):
    return render(
        request,
        "data/dashboard.html",
    )


def clean_json(obj):
    """
    Recursively clean NaN / Infinity / None values from dicts & lists for JSON compliance.
    """
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json(v) for v in obj]
    elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return 0
    elif obj is None:
        return 0
    return obj


class DashboardView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        try:
            # Get filter type and duration unit from request
            filter_type = request.GET.get("filter_type", "category")
            duration_unit = request.GET.get("duration_unit", "seconds").lower()
            if duration_unit not in ["seconds", "hours"]:
                duration_unit = "seconds"

            # Basic stats
            total_recordings = NoiseDataset.objects.count()
            user_recordings = (
                NoiseDataset.objects.filter(collector=request.user).count()
                if request.user.is_authenticated
                else 0
            )
            categories_count = Category.objects.count()
            regions_count = Region.objects.count()
            classes_count = Class.objects.count()
            sub_classes_count = SubClass.objects.count()

            # Calculate total duration in hours
            total_duration_seconds = (
                AudioFeature.objects.aggregate(total_duration=Sum("duration"))[
                    "total_duration"
                ]
                or 0
            )
            total_duration_hours = round(total_duration_seconds / 3600, 2)

            # Calculate average duration per recording
            avg_duration_seconds = (
                AudioFeature.objects.aggregate(avg_duration=Avg("duration"))[
                    "avg_duration"
                ]
                or 0
            )
            avg_duration_seconds = round(avg_duration_seconds, 2)

            # Duration by class in hours
            duration_by_class = (
                AudioFeature.objects.select_related("noise_dataset__class_name")
                .values("noise_dataset__class_name__name")
                .annotate(total_duration=Sum("duration"))
                .order_by("-total_duration")
            )
            class_hours = [
                {
                    "label": item.get("noise_dataset__class_name__name") or "Unknown",
                    "hours": round((item.get("total_duration") or 0) / 3600, 2),
                }
                for item in duration_by_class
            ]

            # Data for category pie chart
            category_data = (
                NoiseDataset.objects.values("category__name")
                .annotate(count=Count("id"))
                .order_by("-count")
            )
            category_labels = [
                item["category__name"] or "Unknown" for item in category_data
            ]
            category_counts = [item["count"] for item in category_data]

            # Data for region bar chart
            region_data = (
                NoiseDataset.objects.values("region__name")
                .annotate(count=Count("id"))
                .order_by("-count")
            )
            region_labels = [item["region__name"] or "Unknown" for item in region_data]
            region_counts = [item["count"] for item in region_data]

            # Data for duration by region chart
            duration_by_region = (
                AudioFeature.objects.select_related("noise_dataset__region")
                .values("noise_dataset__region__name")
                .annotate(total_duration=Sum("duration"))
                .order_by("-total_duration")
            )
            duration_region_labels = [
                item.get("noise_dataset__region__name") or "Unknown"
                for item in duration_by_region
            ]
            duration_region_hours = [
                round(item["total_duration"] / 3600, 2) if item["total_duration"] else 0
                for item in duration_by_region
            ]

            # Data for time line chart (last 12 months)
            time_labels = []
            time_counts = []
            now = timezone.now()
            for i in range(11, -1, -1):
                month = now - timedelta(days=30 * i)
                time_labels.append(month.strftime("%b %Y"))
                start_date = month.replace(
                    day=1, hour=0, minute=0, second=0, microsecond=0
                )
                if month.month == 12:
                    end_date = month.replace(year=month.year + 1, month=1, day=1)
                else:
                    end_date = month.replace(month=month.month + 1, day=1)
                count = NoiseDataset.objects.filter(
                    recording_date__gte=start_date, recording_date__lt=end_date
                ).count()
                time_counts.append(count)

            # Data for audio features radar chart (averages)
            audio_features = AudioFeature.objects.aggregate(
                avg_rms=Avg("rms_energy"),
                avg_centroid=Avg("spectral_centroid"),
                avg_bandwidth=Avg("spectral_bandwidth"),
                avg_zcr=Avg("zero_crossing_rate"),
                avg_harmonic=Avg("harmonic_ratio"),
                avg_percussive=Avg("percussive_ratio"),
            )
            audio_features_data = [
                audio_features["avg_rms"],
                audio_features["avg_centroid"],
                audio_features["avg_bandwidth"],
                audio_features["avg_zcr"],
                audio_features["avg_harmonic"],
                audio_features["avg_percussive"],
            ]

            # Average parameters data for different filters
            def get_average_parameters(filter_type, duration_unit):
                if filter_type == "category":
                    queryset = (
                        NoiseDataset.objects.values("category__name")
                        .annotate(
                            avg_count=Count("id"),
                            avg_duration=Avg("audio_features__duration"),
                            total_duration=Sum("audio_features__duration"),
                        )
                        .order_by("-avg_count")
                    )
                elif filter_type == "class":
                    queryset = (
                        NoiseDataset.objects.values("class_name__name")
                        .annotate(
                            avg_count=Count("id"),
                            avg_duration=Avg("audio_features__duration"),
                            total_duration=Sum("audio_features__duration"),
                        )
                        .order_by("-avg_count")
                    )
                elif filter_type == "subclass":
                    queryset = (
                        NoiseDataset.objects.values("subclass__name")
                        .annotate(
                            avg_count=Count("id"),
                            avg_duration=Avg("audio_features__duration"),
                            total_duration=Sum("audio_features__duration"),
                        )
                        .order_by("-avg_count")
                    )
                elif filter_type == "region":
                    queryset = (
                        NoiseDataset.objects.values("region__name")
                        .annotate(
                            avg_count=Count("id"),
                            avg_duration=Avg("audio_features__duration"),
                            total_duration=Sum("audio_features__duration"),
                        )
                        .order_by("-avg_count")
                    )
                elif filter_type == "community":
                    queryset = (
                        NoiseDataset.objects.values("community__name")
                        .annotate(
                            avg_count=Count("id"),
                            avg_duration=Avg("audio_features__duration"),
                            total_duration=Sum("audio_features__duration"),
                        )
                        .order_by("-avg_count")
                    )
                else:
                    queryset = (
                        NoiseDataset.objects.values("category__name")
                        .annotate(
                            avg_count=Count("id"),
                            avg_duration=Avg("audio_features__duration"),
                            total_duration=Sum("audio_features__duration"),
                        )
                        .order_by("-avg_count")
                    )

                return [
                    {
                        "name": item.get(list(item.keys())[0]) or "Unknown",
                        "avg_count": item["avg_count"],
                        "avg_duration": (
                            round(((item["avg_duration"] or 0) / 3600), 2)
                            if duration_unit == "hours"
                            else round(item["avg_duration"] or 0, 2)
                        ),
                        "total_duration": (
                            round(((item.get("total_duration") or 0) / 3600), 2)
                            if duration_unit == "hours"
                            else round(item.get("total_duration") or 0, 2)
                        ),
                        "duration_unit": duration_unit,
                    }
                    for item in queryset
                ]

            # Get average parameters data for different filters and selected unit
            average_parameters = get_average_parameters(filter_type, duration_unit)

            response_data = {
                "basic_stats": {
                    "total_recordings": total_recordings,
                    "user_recordings": user_recordings,
                    "categories_count": categories_count,
                    "classes_count": classes_count,
                    "sub_classes_count": sub_classes_count,
                    "regions_count": regions_count,
                    "total_duration_hours": total_duration_hours,
                    "avg_duration_seconds": avg_duration_seconds,
                },
                "class_hours": class_hours,
                "average_parameters": average_parameters,
                "current_filter": filter_type,
                "current_unit": duration_unit,
                "charts": {
                    "category": {
                        "labels": category_labels,
                        "data": category_counts,
                    },
                    "region": {
                        "labels": region_labels,
                        "data": region_counts,
                    },
                    "duration_region": {
                        "labels": duration_region_labels,
                        "data": duration_region_hours,
                    },
                    "time_series": {
                        "labels": time_labels,
                        "data": time_counts,
                    },
                    "audio_features": audio_features_data,
                },
            }

            # Clean NaN / Infinity before sending response
            return Response(clean_json(response_data))

        except Exception as e:
            return Response(
                {"error": "Failed to fetch dashboard data", "detail": str(e)},
                status=500,
            )


class NoiseDatasetDeleteView(LoginRequiredMixin, DeleteView):
    model = NoiseDataset
    success_url = reverse_lazy("data:datasetlist")

    def delete(self, request, *args, **kwargs):
        response = super().delete(request, *args, **kwargs)
        messages.success(
            request, f'Dataset "{self.object.noise_id}" was deleted successfully.'
        )
        return response


@login_required
def view_datasetlist(request):
    datasets = NoiseDataset.objects.filter().order_by("-updated_at")
    context = {
        "datasets": datasets,
    }
    return render(request, "data/datasetlist.html", context)


class NoiseDatasetListView(ListView, LoginRequiredMixin):
    model = NoiseDataset
    template_name = "data/datasetlist.html"
    context_object_name = "datasets"
    ordering = ["-created_at"]
    paginate_by = 200  # default, can be overridden by query param

    def get_paginate_by(self, queryset):
        try:
            page_size = int(self.request.GET.get("page_size", self.paginate_by))
        except (TypeError, ValueError):
            page_size = self.paginate_by
        allowed = {50, 100, 200, 300}
        return page_size if page_size in allowed else self.paginate_by

    def get_queryset(self):
        queryset = super().get_queryset()

        # Search functionality
        search = self.request.GET.get("search")
        if search:
            queryset = queryset.filter(
                Q(noise_id__icontains=search) | Q(name__icontains=search)
            )

        # Category filter
        category = self.request.GET.get("category")
        if category:
            queryset = queryset.filter(category__name=category)

        # Class filter
        class_name = self.request.GET.get("class")
        if class_name:
            queryset = queryset.filter(class_name__name=class_name)

        # Subclass filter
        subclass = self.request.GET.get("subclass")
        if subclass:
            queryset = queryset.filter(subclass__name=subclass)

        # Region filter
        region = self.request.GET.get("region")
        if region:
            queryset = queryset.filter(region__name=region)

        # Community filter
        community = self.request.GET.get("community")
        if community:
            queryset = queryset.filter(community__name=community)

        # Date range filter
        date_range = self.request.GET.get("date_range")
        if date_range:
            today = timezone.now().date()
            if date_range == "today":
                queryset = queryset.filter(recording_date__date=today)
            elif date_range == "week":
                start_date = today - timedelta(days=today.weekday())
                queryset = queryset.filter(recording_date__date__gte=start_date)
            elif date_range == "month":
                queryset = queryset.filter(
                    recording_date__month=today.month, recording_date__year=today.year
                )
            elif date_range == "year":
                queryset = queryset.filter(recording_date__year=today.year)

        # Sorting
        sort = self.request.GET.get("sort")
        if sort:
            order = self.request.GET.get("order", "asc")
            if order == "desc":
                sort = f"-{sort}"
            queryset = queryset.order_by(sort)

        return queryset

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        # Add filter options to context for the template
        context["categories"] = Category.objects.all()
        context["classes"] = Class.objects.all()
        context["subclasses"] = SubClass.objects.all()
        context["regions"] = Region.objects.all()
        context["communities"] = Community.objects.all()

        # Add current filters to context
        context["current_filters"] = {
            "search": self.request.GET.get("search"),
            "category": self.request.GET.get("category"),
            "class": self.request.GET.get("class"),
            "subclass": self.request.GET.get("subclass"),
            "region": self.request.GET.get("region"),
            "community": self.request.GET.get("community"),
            "date_range": self.request.GET.get("date_range"),
            "page_size": self.request.GET.get("page_size"),
        }

        return context


def _filtered_noise_queryset(request):
    queryset = NoiseDataset.objects.all().order_by("-created_at")

    # Search
    search = request.GET.get("search")
    if search:
        queryset = queryset.filter(
            Q(noise_id__icontains=search) | Q(name__icontains=search)
        )

    # Category
    category = request.GET.get("category")
    if category:
        queryset = queryset.filter(category__name=category)

    # Class
    class_name = request.GET.get("class")
    if class_name:
        queryset = queryset.filter(class_name__name=class_name)

    # Subclass
    subclass = request.GET.get("subclass")
    if subclass:
        queryset = queryset.filter(subclass__name=subclass)

    # Region
    region = request.GET.get("region")
    if region:
        queryset = queryset.filter(region__name=region)

    # Community
    community = request.GET.get("community")
    if community:
        queryset = queryset.filter(community__name=community)

    # Date range
    date_range = request.GET.get("date_range")
    if date_range:
        today = timezone.now().date()
        if date_range == "today":
            queryset = queryset.filter(recording_date__date=today)
        elif date_range == "week":
            start_date = today - timedelta(days=today.weekday())
            queryset = queryset.filter(recording_date__date__gte=start_date)
        elif date_range == "month":
            queryset = queryset.filter(
                recording_date__month=today.month, recording_date__year=today.year
            )
        elif date_range == "year":
            queryset = queryset.filter(recording_date__year=today.year)

    return queryset


class ExportDataAPIView(APIView):
    """API endpoint to fetch datasets for export with pagination"""

    permission_classes = [IsAuthenticated]

    def get(self, request):
        import time
        from django.db import connection
        
        start_time = time.time()
        
        # Get batch parameters first
        batch_size = int(request.GET.get("batch_size", 500))
        offset = int(request.GET.get("offset", 0))
        batch_size = min(batch_size, 1000)
        
        # Build WHERE clause based on filters
        where_clauses = []
        params = []
        
        # Apply search filter
        search_query = request.GET.get("search", "").strip()
        if search_query:
            where_clauses.append("(nd.name ILIKE %s OR nd.noise_id ILIKE %s OR nd.description ILIKE %s)")
            search_pattern = f"%{search_query}%"
            params.extend([search_pattern, search_pattern, search_pattern])
        
        # Apply other filters from request
        if request.GET.get("category"):
            where_clauses.append("nd.category_id = %s")
            params.append(request.GET.get("category"))
        if request.GET.get("region"):
            where_clauses.append("nd.region_id = %s")
            params.append(request.GET.get("region"))
        if request.GET.get("community"):
            where_clauses.append("nd.community_id = %s")
            params.append(request.GET.get("community"))
        if request.GET.get("dataset_type"):
            where_clauses.append("nd.dataset_type_id = %s")
            params.append(request.GET.get("dataset_type"))
        if request.GET.get("collector"):
            where_clauses.append("nd.collector_id = %s")
            params.append(request.GET.get("collector"))
        
        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        # For first request, get total count
        if offset == 0:
            count_sql = f"""
                SELECT COUNT(*) 
                FROM data_noisedataset nd
                WHERE {where_sql}
            """
            with connection.cursor() as cursor:
                cursor.execute(count_sql, params)
                total_count = cursor.fetchone()[0]
        else:
            total_count = int(request.GET.get("total", 0))

        logger.info(f"Export API: offset={offset}, batch_size={batch_size}, total={total_count}")
        
        # Raw SQL query for maximum performance - join all tables in one query
        sql = f"""
            SELECT 
                nd.id, nd.noise_id, nd.name, nd.description, nd.recording_device,
                nd.recording_date, nd.created_at, nd.updated_at, nd.audio,
                cat.name as category_name,
                cls.name as class_name,
                sub.name as subclass_name,
                reg.name as region_name,
                com.name as community_name,
                usr.username as collector_username,
                dt.name as dataset_type_name,
                mt.name as microphone_type_name,
                tod.name as time_of_day_name,
                af.duration, af.sample_rate, af.num_samples, af.rms_energy,
                af.zero_crossing_rate, af.spectral_centroid, af.spectral_bandwidth,
                af.spectral_rolloff, af.spectral_flatness, af.harmonic_ratio, af.percussive_ratio,
                na.mean_db, na.max_db, na.min_db, na.std_db,
                na.peak_count, na.peak_interval_mean, na.dominant_frequency,
                na.frequency_range, na.event_count
            FROM data_noisedataset nd
            LEFT JOIN core_category cat ON nd.category_id = cat.id
            LEFT JOIN core_class cls ON nd.class_name_id = cls.id
            LEFT JOIN core_subclass sub ON nd.subclass_id = sub.id
            LEFT JOIN core_region reg ON nd.region_id = reg.id
            LEFT JOIN core_community com ON nd.community_id = com.id
            LEFT JOIN authentication_customuser usr ON nd.collector_id = usr.id
            LEFT JOIN core_specific_mix_setting dt ON nd.dataset_type_id = dt.id
            LEFT JOIN core_microphone_type mt ON nd.microphone_type_id = mt.id
            LEFT JOIN core_time_of_day tod ON nd.time_of_day_id = tod.id
            LEFT JOIN data_audiofeature af ON nd.id = af.noise_dataset_id
            LEFT JOIN data_noiseanalysis na ON nd.id = na.noise_dataset_id
            WHERE {where_sql}
            ORDER BY nd.id
            LIMIT %s OFFSET %s
        """
        
        query_params = params + [batch_size, offset]
        
        with connection.cursor() as cursor:
            cursor.execute(sql, query_params)
            columns = [col[0] for col in cursor.description]
            rows = cursor.fetchall()
        
        logger.info(f"Raw SQL query completed in {time.time() - start_time:.2f}s, fetched {len(rows)} rows")

        # Build export-ready rows from raw SQL results (ultra-fast)
        export_data = []
        col_map = {col: idx for idx, col in enumerate(columns)}
        
        for row in rows:
            def safe_val(key, default=""):
                val = row[col_map.get(key)]
                if val is None:
                    return default
                # Handle datetime objects
                if hasattr(val, 'isoformat'):
                    return val.isoformat()
                return val
            
            export_data.append({
                "noise_id": safe_val('noise_id'),
                "name": safe_val('name'),
                "dataset_type": safe_val('dataset_type_name'),
                "collector": safe_val('collector_username'),
                "description": safe_val('description'),
                "recording_device": safe_val('recording_device'),
                "recording_date": safe_val('recording_date'),
                "created_at": safe_val('created_at'),
                "updated_at": safe_val('updated_at'),
                "region": safe_val('region_name'),
                "community": safe_val('community_name'),
                "category": safe_val('category_name'),
                "class": safe_val('class_name'),
                "subclass": safe_val('subclass_name'),
                "time_of_day": safe_val('time_of_day_name'),
                "microphone_type": safe_val('microphone_type_name'),
                "audio_file": safe_val('audio'),
                "audio_size": "",
                "audio_duration": safe_val('duration'),
                "sample_rate": safe_val('sample_rate'),
                "num_samples": safe_val('num_samples'),
                "rms_energy": safe_val('rms_energy'),
                "zero_crossing_rate": safe_val('zero_crossing_rate'),
                "spectral_centroid": safe_val('spectral_centroid'),
                "spectral_bandwidth": safe_val('spectral_bandwidth'),
                "spectral_rolloff": safe_val('spectral_rolloff'),
                "spectral_flatness": safe_val('spectral_flatness'),
                "harmonic_ratio": safe_val('harmonic_ratio'),
                "percussive_ratio": safe_val('percussive_ratio'),
                "mean_db": safe_val('mean_db'),
                "max_db": safe_val('max_db'),
                "min_db": safe_val('min_db'),
                "std_db": safe_val('std_db'),
                "peak_count": safe_val('peak_count'),
                "peak_interval_mean": safe_val('peak_interval_mean'),
                "dominant_frequency": safe_val('dominant_frequency'),
                "frequency_range": safe_val('frequency_range'),
                "event_count": safe_val('event_count'),
            })
        
        logger.info(f"Total API time: {time.time() - start_time:.2f}s for {len(export_data)} records")

        return Response({
            "results": export_data,
            "total": total_count,
            "offset": offset,
            "batch_size": batch_size,
            "has_more": (offset + batch_size) < total_count,
        })


@login_required
def export_noise_datasets(request):
    fmt = (request.GET.get("format") or "csv").lower()
    queryset = _filtered_noise_queryset(request)

    # Check total count before processing
    total_count = queryset.count()
    MAX_EXPORT_LIMIT = 50000  # Maximum records to export in one go

    if total_count > MAX_EXPORT_LIMIT:
        return HttpResponse(
            f"Export limit exceeded. You are trying to export {total_count} records, "
            f"but the maximum allowed is {MAX_EXPORT_LIMIT}. Please use filters to reduce the dataset size.",
            status=400,
            content_type="text/plain",
        )

    # Preload related to minimize queries - use iterator() for memory efficiency
    queryset = queryset.select_related(
        "category",
        "class_name",
        "subclass",
        "region",
        "community",
        "collector",
        "dataset_type",
        "microphone_type",
        "time_of_day",
        "audio_features",
        "noise_analysis",
    )

    # Define full set of columns
    columns = [
        ("Noise ID", None),
        ("Name", None),
        ("Dataset Type", None),
        ("Collector", None),
        ("Description", None),
        ("Recording Device", None),
        ("Recording Date", None),
        ("Created At", None),
        ("Updated At", None),
        ("Region", None),
        ("Community", None),
        ("Category", None),
        ("Class", None),
        ("Subclass", None),
        ("Time Of Day", None),
        ("Microphone Type", None),
        ("Audio File", None),
        ("Audio Size (bytes)", None),
        ("Audio Duration (s)", None),
        ("Sample Rate", None),
        ("Num Samples", None),
        ("RMS Energy", None),
        ("Zero Crossing Rate", None),
        ("Spectral Centroid", None),
        ("Spectral Bandwidth", None),
        ("Spectral Rolloff", None),
        ("Spectral Flatness", None),
        ("Harmonic Ratio", None),
        ("Percussive Ratio", None),
        ("Mean dB", None),
        ("Max dB", None),
        ("Min dB", None),
        ("Std dB", None),
        ("Peak Count", None),
        ("Peak Interval Mean", None),
        ("Dominant Frequency (Hz)", None),
        ("Frequency Range", None),
        ("Event Count", None),
    ]

    def build_row(obj):
        af = getattr(obj, "audio_features", None)
        na = getattr(obj, "noise_analysis", None)
        try:
            audio_name = obj.audio.name if obj.audio else ""
        except Exception:
            audio_name = ""
        try:
            audio_size = obj.audio.size if obj.audio else None
        except Exception:
            audio_size = None
        return [
            smart_str(getattr(obj, "noise_id", "")),
            smart_str(getattr(obj, "name", "")),
            smart_str(
                getattr(
                    getattr(obj, "dataset_type", None),
                    "get_name_display",
                    lambda: getattr(getattr(obj, "dataset_type", None), "name", ""),
                )()
            ),
            smart_str(getattr(getattr(obj, "collector", None), "username", "")),
            smart_str(getattr(obj, "description", "")),
            smart_str(getattr(obj, "recording_device", "")),
            smart_str(
                obj.recording_date.isoformat()
                if getattr(obj, "recording_date", None)
                else ""
            ),
            smart_str(
                obj.created_at.isoformat() if getattr(obj, "created_at", None) else ""
            ),
            smart_str(
                obj.updated_at.isoformat() if getattr(obj, "updated_at", None) else ""
            ),
            smart_str(getattr(getattr(obj, "region", None), "name", "")),
            smart_str(getattr(getattr(obj, "community", None), "name", "")),
            smart_str(getattr(getattr(obj, "category", None), "name", "")),
            smart_str(getattr(getattr(obj, "class_name", None), "name", "")),
            smart_str(getattr(getattr(obj, "subclass", None), "name", "")),
            smart_str(getattr(getattr(obj, "time_of_day", None), "name", "")),
            smart_str(getattr(getattr(obj, "microphone_type", None), "name", "")),
            smart_str(audio_name),
            audio_size if audio_size is not None else "",
            af.duration if af and af.duration is not None else "",
            af.sample_rate if af and af.sample_rate is not None else "",
            af.num_samples if af and af.num_samples is not None else "",
            af.rms_energy if af and af.rms_energy is not None else "",
            af.zero_crossing_rate if af and af.zero_crossing_rate is not None else "",
            af.spectral_centroid if af and af.spectral_centroid is not None else "",
            af.spectral_bandwidth if af and af.spectral_bandwidth is not None else "",
            af.spectral_rolloff if af and af.spectral_rolloff is not None else "",
            af.spectral_flatness if af and af.spectral_flatness is not None else "",
            af.harmonic_ratio if af and af.harmonic_ratio is not None else "",
            af.percussive_ratio if af and af.percussive_ratio is not None else "",
            na.mean_db if na and na.mean_db is not None else "",
            na.max_db if na and na.max_db is not None else "",
            na.min_db if na and na.min_db is not None else "",
            na.std_db if na and na.std_db is not None else "",
            na.peak_count if na and na.peak_count is not None else "",
            na.peak_interval_mean if na and na.peak_interval_mean is not None else "",
            na.dominant_frequency if na and na.dominant_frequency is not None else "",
            smart_str(na.frequency_range) if na and na.frequency_range else "",
            na.event_count if na and na.event_count is not None else "",
        ]

    if fmt == "xlsx":
        if openpyxl is None:
            return HttpResponse(
                "Excel export requires openpyxl. Please install it.", status=500
            )

        # For XLSX, use write-only mode for better memory efficiency with large datasets
        # Increased limit to 50,000 to match CSV limit - Excel can handle much more
        MAX_XLSX_LIMIT = 50000
        if total_count > MAX_XLSX_LIMIT:
            return HttpResponse(
                f"Excel export limit exceeded. You are trying to export {total_count} records, "
                f"but Excel export maximum is {MAX_XLSX_LIMIT}. Please use CSV export or apply filters to reduce the dataset size.",
                status=400,
                content_type="text/plain",
            )

        # Use write-only workbook for better memory efficiency with large datasets
        output = BytesIO()
        wb = openpyxl.Workbook(write_only=True)
        ws = wb.create_sheet(title="Noise Datasets")

        # Write header
        ws.append([c[0] for c in columns])

        # Process in chunks to avoid memory issues
        CHUNK_SIZE = 1000
        offset = 0
        processed = 0

        while processed < total_count:
            chunk = queryset[offset : offset + CHUNK_SIZE]
            for obj in chunk:
                try:
                    ws.append(build_row(obj))
                except Exception as e:
                    logger.error(f"Error building row for dataset {obj.id}: {e}")
                    # Continue with next row
                    continue
            offset += CHUNK_SIZE
            processed += len(chunk)
            # Clear memory
            del chunk

        # Save the workbook to the output stream
        wb.save(output)
        output.seek(0)
        response = HttpResponse(
            output.getvalue(),
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        response["Content-Disposition"] = 'attachment; filename="noise_datasets.xlsx"'
        return response

    # CSV: Use streaming response for large datasets
    def csv_generator():
        # Create a buffer for CSV writing
        buffer = StringIO()
        writer = csv.writer(buffer)

        # Write header first
        writer.writerow([c[0] for c in columns])
        header_data = buffer.getvalue()
        buffer.seek(0)
        buffer.truncate(0)
        yield header_data

        # Process queryset in chunks using iterator() for memory efficiency
        CHUNK_SIZE = 500
        offset = 0
        processed = 0

        while processed < total_count:
            chunk = queryset[offset : offset + CHUNK_SIZE]
            for obj in chunk:
                try:
                    row = build_row(obj)
                    writer.writerow(row)
                    row_data = buffer.getvalue()
                    buffer.seek(0)
                    buffer.truncate(0)
                    yield row_data
                except Exception as e:
                    logger.error(f"Error building row for dataset {obj.id}: {e}")
                    # Continue with next row
                    continue
            offset += CHUNK_SIZE
            processed += len(chunk)
            # Clear memory
            del chunk

        buffer.close()

    response = StreamingHttpResponse(csv_generator(), content_type="text/csv")
    response["Content-Disposition"] = 'attachment; filename="noise_datasets.csv"'
    return response


@login_required
def noise_dataset_create(request):
    if request.method == "POST":
        form = NoiseDatasetForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                noise_dataset = form.save(commit=False)
                noise_dataset.collector = request.user

                # Generate the new noise ID format
                noise_dataset.noise_id = generate_noise_id(request.user)

                # Generate dataset name
                noise_dataset.name = generate_dataset_name(noise_dataset)

                # Handle audio file
                if "audio" in request.FILES:
                    audio_file = request.FILES["audio"]

                    # Generate hash of the new audio file for duplicate checking
                    hash_md5 = hashlib.md5()
                    for chunk in audio_file.chunks():
                        hash_md5.update(chunk)

                    # Check against existing records
                    # duplicates = NoiseDataset.objects.filter(collector=request.user)
                    # for dataset in duplicates:
                    # if dataset.audio:
                    # Reset file pointer after reading chunks for hash
                    #  audio_file.seek(0)

                    # Compare hashes
                    # if dataset.get_audio_hash() == new_audio_hash:
                    #  messages.error(request, "This audio file has already been uploaded!")
                    #  return redirect('data:noise_dataset_create')

                    # Reset file pointer again before processing
                    audio_file.seek(0)

                    # Get file extension
                    file_extension = audio_file.name.split(".")[-1].lower()

                    # Generate new filename using noise_id
                    new_filename = f"{noise_dataset.noise_id}.{file_extension}"

                    # Create a renamed file object
                    renamed_file = RenamedFile(audio_file, new_filename)

                    # Replace the file in the form data
                    request.FILES["audio"] = renamed_file

                    # Calculate duration

                noise_dataset.save()
                form.save_m2m()

                messages.success(
                    request,
                    f'Noise dataset "{noise_dataset.name}" created successfully!',
                )
                return redirect("data:noise_dataset_create")

            except Exception as e:
                messages.error(request, f"Error creating dataset: {str(e)}")
                print(f"Error creating noise dataset: {e}")
    else:
        form = NoiseDatasetForm()

    context = {"form": form}
    return render(request, "data/AddNewData.html", context)


@login_required
def bulk_upload_view(request):
    if request.method == "POST":
        form = BulkAudioUploadForm(request.POST)
        if form.is_valid():
            try:
                # Prepare metadata
                metadata = {
                    "description": form.cleaned_data["description"],
                    "region_id": form.cleaned_data["region"].id,
                    "category_id": form.cleaned_data["category"].id,
                    "time_of_day_id": form.cleaned_data["time_of_day"].id,
                    "community_id": form.cleaned_data["community"].id,
                    "class_name_id": form.cleaned_data["class_name"].id,
                    "subclass_id": (
                        form.cleaned_data["subclass"].id
                        if form.cleaned_data["subclass"]
                        else None
                    ),
                    "microphone_type_id": (
                        form.cleaned_data["microphone_type"].id
                        if form.cleaned_data["microphone_type"]
                        else None
                    ),
                    "recording_date": form.cleaned_data["recording_date"].isoformat(),
                    "recording_device": form.cleaned_data["recording_device"],
                }

                upload_dir = os.path.join(
                    settings.SHARED_UPLOADS_DIR, f"user_{request.user.id}"
                )
                os.makedirs(upload_dir, exist_ok=True)
                file_pattern = os.path.join(upload_dir, "*")
                file_paths = [p for p in glob.glob(file_pattern) if os.path.isfile(p)]

                if not file_paths:
                    return JsonResponse(
                        {"status": "error", "error": "No files found for processing"},
                        status=400,
                    )

                # Create bulk upload record
                bulk_upload = BulkAudioUpload.objects.create(
                    user=request.user,
                    metadata=metadata,
                    total_files=len(file_paths),
                )

                # Start async processing with user ID
                process_bulk_upload.delay(
                    bulk_upload.id,
                    file_paths,
                    request.user.id,
                )

                return JsonResponse(
                    {
                        "status": "success",
                        "bulk_upload_id": bulk_upload.id,
                        "message": "Bulk upload started successfully",
                        "total_files": bulk_upload.total_files,
                    }
                )

            except Exception as e:
                logger.error(
                    f"Bulk upload initialization failed: {str(e)}", exc_info=True
                )
                return JsonResponse({"status": "error", "error": str(e)}, status=500)
        else:
            return JsonResponse({"status": "error", "errors": form.errors}, status=400)
    else:
        form = BulkAudioUploadForm()

    return render(request, "data/bulk_upload.html", {"form": form})


@login_required
def upload_chunk(request):
    if request.method == "POST":
        try:
            chunk = request.FILES.get("file")
            chunk_number = int(request.POST.get("chunkNumber", 0))
            total_chunks = int(request.POST.get("totalChunks", 1))
            file_name = request.POST.get("fileName", "")
            file_uid = request.POST.get("fileUid", "")

            if not chunk or not file_name:
                return JsonResponse(
                    {"status": "error", "error": "Missing file data"}, status=400
                )

            # Create user-specific upload directory
            upload_dir = os.path.join(
                settings.SHARED_UPLOADS_DIR, f"user_{request.user.id}"
            )
            os.makedirs(upload_dir, exist_ok=True)

            # Create temporary directory for this file's chunks
            temp_dir = os.path.join(upload_dir, f"temp_{file_uid}")
            os.makedirs(temp_dir, exist_ok=True)

            # Save chunk
            chunk_path = os.path.join(temp_dir, f"{chunk_number}.part")
            with open(chunk_path, "wb+") as f:
                for piece in chunk.chunks():
                    f.write(piece)

            # Check if all chunks received
            received_chunks = len(
                [name for name in os.listdir(temp_dir) if name.endswith(".part")]
            )
            if received_chunks == total_chunks:
                # Reassemble file with unique prefix to avoid collisions
                safe_uid = "".join(
                    c for c in file_uid if c.isalnum() or c in ("-", "_")
                )
                final_name = f"{safe_uid}_{file_name}" if safe_uid else file_name
                final_path = os.path.join(upload_dir, final_name)
                with open(final_path, "wb") as outfile:
                    for i in range(total_chunks):
                        chunk_path = os.path.join(temp_dir, f"{i}.part")
                        with open(chunk_path, "rb") as infile:
                            outfile.write(infile.read())
                        os.remove(chunk_path)
                os.rmdir(temp_dir)

                return JsonResponse(
                    {"status": "success", "completed": True, "file_path": final_path}
                )

            return JsonResponse({"status": "success", "completed": False})

        except Exception as e:
            logger.error(f"Chunk upload failed: {str(e)}", exc_info=True)
            return JsonResponse({"status": "error", "error": str(e)}, status=500)

    return JsonResponse(
        {"status": "error", "error": "Invalid request method"}, status=400
    )


@login_required
def bulk_upload_progress(request, bulk_upload_id):
    try:
        bulk_upload = BulkAudioUpload.objects.get(id=bulk_upload_id, user=request.user)
        return JsonResponse(
            {
                "status": bulk_upload.status,
                "processed": bulk_upload.processed_files,
                "total": bulk_upload.total_files,
                "failed": bulk_upload.failed_files,
            }
        )
    except BulkAudioUpload.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)


@login_required
def cancel_upload(request, bulk_upload_id):
    try:
        bulk_upload = BulkAudioUpload.objects.get(id=bulk_upload_id, user=request.user)
        if bulk_upload.status in ["pending", "processing"]:
            bulk_upload.status = "cancelled"
            bulk_upload.save()

            # Clean up files
            upload_dir = os.path.join(
                settings.SHARED_UPLOADS_DIR, f"user_{request.user.id}"
            )
            file_pattern = os.path.join(upload_dir, "*")
            for file_path in glob.glob(file_pattern):
                try:
                    os.remove(file_path)
                except:
                    pass

            return JsonResponse({"status": "success", "message": "Upload cancelled"})
        else:
            return JsonResponse(
                {"status": "error", "error": "Cannot cancel completed upload"},
                status=400,
            )
    except BulkAudioUpload.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)


@login_required
def noise_dataset_edit(request, pk):

    noise_dataset = NoiseDataset.objects.get(pk=pk, collector=request.user)

    if request.method == "POST":
        form = NoiseDatasetForm(request.POST, request.FILES, instance=noise_dataset)
        if form.is_valid():
            try:
                updated_dataset = form.save(commit=False)

                # Handle audio file update
                if "audio" in request.FILES:
                    audio_file = request.FILES["audio"]

                    # Generate hash of the new audio file for duplicate checking
                    hash_md5 = hashlib.md5()
                    for chunk in audio_file.chunks():
                        hash_md5.update(chunk)
                    new_audio_hash = hash_md5.hexdigest()

                    # Check against existing records (excluding current record)
                    duplicates = NoiseDataset.objects.filter(
                        collector=request.user
                    ).exclude(pk=pk)
                    for dataset in duplicates:
                        if dataset.audio:
                            # Reset file pointer after reading chunks for hash
                            audio_file.seek(0)

                            # Compare hashes
                            if dataset.get_audio_hash() == new_audio_hash:
                                messages.error(
                                    request,
                                    "This audio file has already been uploaded!",
                                )
                                return redirect("data:noise_dataset_edit", pk=pk)

                    # Reset file pointer again before processing
                    audio_file.seek(0)

                    # Get file extension
                    file_extension = audio_file.name.split(".")[-1].lower()

                    # Generate new filename using existing noise_id
                    new_filename = f"{noise_dataset.noise_id}.{file_extension}"

                    # Create a renamed file object
                    renamed_file = RenamedFile(audio_file, new_filename)

                    # Replace the file in the form data
                    request.FILES["audio"] = renamed_file

                    # Calculate duration

                # Update dataset name based on new values
                updated_dataset.name = generate_dataset_name(updated_dataset)
                updated_dataset.save()
                form.save_m2m()

                messages.success(
                    request,
                    f'Noise dataset "{updated_dataset.name}" updated successfully!',
                )
                return redirect("data:noise_dataset_detail", dataset_id=pk)

            except Exception as e:
                messages.error(request, f"Error updating dataset: {str(e)}")
                print(f"Error updating noise dataset: {e}")
    else:
        form = NoiseDatasetForm(instance=noise_dataset)

    context = {
        "form": form,
        "noise_dataset": noise_dataset,
    }
    return render(request, "data/AddNewData.html", context)


def load_classes(request):
    category_id = request.GET.get("category_id")
    classes = Class.objects.filter(category_id=category_id).order_by("name")
    return JsonResponse(list(classes.values("id", "name")), safe=False)


def load_subclasses(request):
    class_id = request.GET.get("class_id")
    subclasses = SubClass.objects.filter(parent_class_id=class_id).order_by("name")
    return JsonResponse(list(subclasses.values("id", "name")), safe=False)


def load_communities(request):
    region_id = request.GET.get("region_id")
    communities = Community.objects.filter(region_id=region_id).order_by("name")
    data = [{"id": c.id, "name": c.name} for c in communities]
    return JsonResponse(data, safe=False)


def show_pages(request):
    return render(request, "data/datasetlist.html")


@login_required
def noise_detail(request, dataset_id):
    dataset = get_object_or_404(NoiseDataset, pk=dataset_id)

    # Get related data
    audio_features = getattr(dataset, "audio_features", None)
    noise_analysis = getattr(dataset, "noise_analysis", None)

    # Precompute safe audio fields to avoid storage errors in templates
    safe_audio_url = None
    safe_audio_size = None
    safe_audio_ext = None
    audio_exists = False

    try:
        if dataset.audio:
            # Check url
            try:
                safe_audio_url = dataset.audio.url
                audio_exists = True
            except Exception as url_exc:
                print(
                    f"[noise_detail] Failed to resolve audio URL for {dataset.pk}: {url_exc}"
                )
                safe_audio_url = None
                audio_exists = False

            # Size and extension are optional in UI; compute only if URL exists
            if audio_exists:
                try:
                    safe_audio_size = dataset.audio.size
                except Exception as size_exc:
                    print(
                        f"[noise_detail] Failed to resolve audio size for {dataset.pk}: {size_exc}"
                    )
                    safe_audio_size = None

                try:
                    audio_name = dataset.audio.name or ""
                    safe_audio_ext = (
                        audio_name[-4:].upper()
                        if len(audio_name) >= 4
                        else audio_name.upper()
                    ) or None
                except Exception as ext_exc:
                    print(
                        f"[noise_detail] Failed to resolve audio extension for {dataset.pk}: {ext_exc}"
                    )
                    safe_audio_ext = None
    except Exception as audio_exc:
        print(
            f"[noise_detail] Unexpected audio resolution error for {dataset.pk}: {audio_exc}"
        )
        safe_audio_url = None
        safe_audio_size = None
        safe_audio_ext = None
        audio_exists = False

    # Prepare visualization data (plots are now fetched via API for faster load)
    context = {
        "noise_dataset": dataset,
        "audio_features": audio_features,
        "noise_analysis": noise_analysis,
        # Safe audio fields for template
        "audio_url": safe_audio_url,
        "audio_exists": audio_exists,
        "audio_size": safe_audio_size,
        "audio_ext": safe_audio_ext,
    }

    return render(request, "data/Noise_detail.html", context)


@login_required
def api_waveform(request, dataset_id):
    dataset = get_object_or_404(NoiseDataset, pk=dataset_id)
    audio_features = getattr(dataset, "audio_features", None)
    if not audio_features or not audio_features.waveform_data:
        return JsonResponse({"success": False, "reason": "no_waveform"})
    fig = create_waveform_plot(audio_features.waveform_data)
    fig_dict = json.loads(json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder))
    return JsonResponse({"success": True, "figure": fig_dict})


@login_required
def api_spectrogram(request, dataset_id):
    dataset = get_object_or_404(NoiseDataset, pk=dataset_id)
    audio_features = getattr(dataset, "audio_features", None)
    if not audio_features or not audio_features.mel_spectrogram:
        return JsonResponse({"success": False, "reason": "no_spectrogram"})
    fig = create_spectrogram_plot(audio_features.mel_spectrogram)
    fig_dict = json.loads(json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder))
    return JsonResponse({"success": True, "figure": fig_dict})


@login_required
def api_mfcc(request, dataset_id):
    dataset = get_object_or_404(NoiseDataset, pk=dataset_id)
    audio_features = getattr(dataset, "audio_features", None)
    if not audio_features or not audio_features.mfccs:
        return JsonResponse({"success": False, "reason": "no_mfcc"})
    fig = create_mfcc_plot(audio_features.mfccs)
    fig_dict = json.loads(json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder))
    return JsonResponse({"success": True, "figure": fig_dict})


@login_required
def api_freq_features(request, dataset_id):
    dataset = get_object_or_404(NoiseDataset, pk=dataset_id)
    audio_features = getattr(dataset, "audio_features", None)
    if not audio_features:
        return JsonResponse({"success": False, "reason": "no_features"})
    fig = create_frequency_features_plot(audio_features)
    fig_dict = json.loads(json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder))
    return JsonResponse({"success": True, "figure": fig_dict})


def create_waveform_plot(waveform_data):
    fig = go.Figure()

    if isinstance(waveform_data, dict):
        x = waveform_data.get("time", [])
        y = waveform_data.get("amplitude", [])

    elif isinstance(waveform_data, list):
        x = list(range(len(waveform_data)))
        y = waveform_data
    else:
        x = []
        y = []

    fig.add_trace(
        go.Scatter(x=x, y=y, mode="lines", name="Waveform", line=dict(color="#1f77b4"))
    )
    fig.update_layout(
        title="Audio Waveform",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        height=300,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


def create_spectrogram_plot(spectrogram_data):
    fig = go.Figure()

    if isinstance(spectrogram_data, dict):

        z = spectrogram_data.get("values", [])
        x = spectrogram_data.get("time", [])
        y = spectrogram_data.get("freq", [])
    elif isinstance(spectrogram_data, list):

        z = spectrogram_data
        x = list(range(len(spectrogram_data[0]))) if spectrogram_data else []
        y = list(range(len(spectrogram_data))) if spectrogram_data else []
    else:
        z, x, y = [], [], []

    if z:  # Only plot if we have data
        fig.add_trace(go.Heatmap(z=z, x=x, y=y, colorscale="Jet"))

    fig.update_layout(
        title="Mel Spectrogram",
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        height=300,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


def create_mfcc_plot(mfccs):
    # Debug: Print the input to see what we're working with
    print(f"Raw MFCCs data type: {type(mfccs)}")
    if hasattr(mfccs, "shape"):
        print(f"MFCCs shape: {mfccs.shape}")
    elif isinstance(mfccs, list):
        print(f"MFCCs list length: {len(mfccs)}")

    # Convert to numpy array if not already
    try:
        if not isinstance(mfccs, np.ndarray):
            mfccs = np.array(mfccs)
    except Exception as e:
        print(f"Error converting MFCCs to numpy array: {e}")
        mfccs = np.array([])

    # Validate the array
    if mfccs.size == 0:
        print("Warning: Empty MFCC data received")
        # Return empty figure with message
        fig = go.Figure()
        fig.update_layout(
            title="MFCC Coefficients (No Data)",
            height=300,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        return fig

    # Ensure 2D array
    if len(mfccs.shape) == 1:
        mfccs = np.expand_dims(mfccs, axis=0)

    print(f"Final MFCCs shape: {mfccs.shape}")

    # Create the plot
    fig = go.Figure(
        data=go.Heatmap(z=mfccs, colorscale="Viridis", colorbar=dict(title="Magnitude"))
    )

    fig.update_layout(
        title="MFCC Coefficients",
        xaxis_title="Coefficient Index",
        yaxis_title="Frame",
        height=300,
        margin=dict(l=40, r=40, t=40, b=40),
    )

    return fig


def create_frequency_features_plot(audio_features):
    fig = go.Figure()

    features = [
        ("spectral_centroid", "Spectral Centroid", "red"),
        ("spectral_bandwidth", "Spectral Bandwidth", "blue"),
        ("spectral_rolloff", "Spectral Rolloff", "green"),
    ]

    for i, (attr, name, color) in enumerate(features, start=1):
        value = getattr(audio_features, attr, None)
        if value is not None:
            fig.add_trace(
                go.Scatter(
                    x=[i],
                    y=[value],
                    mode="markers+text",
                    name=name,
                    text=[name],
                    textposition="top center",
                    marker=dict(size=12, color=color),
                )
            )

    fig.update_layout(
        title="Spectral Features Comparison",
        xaxis=dict(
            title="Feature Type",
            tickvals=[1, 2, 3],
            ticktext=["Centroid", "Bandwidth", "Rolloff"],
            range=[0.5, 3.5],  # Add some padding
        ),
        yaxis_title="Frequency (Hz)",
        height=300,
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=False,  # Since we're showing labels on markers
    )
    return fig
