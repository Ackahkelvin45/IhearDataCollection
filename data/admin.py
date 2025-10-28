from django.contrib import admin
from .models import (
    NoiseDataset,
    AudioFeature,
    NoiseAnalysis,
    VisualizationPreset,
    BulkReprocessingTask,
    Dataset,
)
from unfold.admin import ModelAdmin
from django.shortcuts import render, redirect
from django.contrib import messages
from django.urls import path
from django.http import HttpResponseRedirect, JsonResponse
from django.db.models import Q
from .tasks import process_audio_task, bulk_reprocess_audio_analysis
from celery import group
import logging


class MissingClassificationFilter(admin.SimpleListFilter):
    title = 'Missing Classification'
    parameter_name = 'missing_classification'

    def lookups(self, request, model_admin):
        return (
            ('missing_category', 'Missing Category'),
            ('missing_class', 'Missing Class'),
            ('missing_subclass', 'Missing Subclass'),
            ('missing_all', 'Missing All (Category, Class, Subclass)'),
            ('missing_any', 'Missing Any Classification'),
        )

    def queryset(self, request, queryset):
        if self.value() == 'missing_category':
            return queryset.filter(category__isnull=True)
        elif self.value() == 'missing_class':
            return queryset.filter(class_name__isnull=True)
        elif self.value() == 'missing_subclass':
            return queryset.filter(subclass__isnull=True)
        elif self.value() == 'missing_all':
            return queryset.filter(
                category__isnull=True,
                class_name__isnull=True,
                subclass__isnull=True
            )
        elif self.value() == 'missing_any':
            return queryset.filter(
                Q(category__isnull=True) |
                Q(class_name__isnull=True) |
                Q(subclass__isnull=True)
            )
        return queryset

logger = logging.getLogger(__name__)


@admin.register(NoiseDataset)
class NoiseDatasetAdmin(ModelAdmin):
    list_display = (
        "noise_id",
        "name",
        "collector",
        "subclass",
        "category",
        "region",
        "community",
        "recording_date",
        "updated_at",
        "processing_status",
        "dataset_type",
    )
    list_filter = (
        "category",
        "region",
        "community",
        "time_of_day",
        "class_name",
        "subclass",
        "recording_date",
        MissingClassificationFilter,
    )
    search_fields = (
        "noise_id",
        "name",
        "description",
        "community__name",
        "collector__username",
    )
    list_editable = ("name",)
    list_select_related = ("collector", "category", "region", "community")
    list_per_page = 25
    actions = ["reprocess_audio_analysis"]

    fieldsets = (
        (
            "Basic Information",
            {
                "fields": (
                    "noise_id",
                    "name",
                    "collector",
                    "description",
                    "audio",
                    "dataset_type",
                )
            },
        ),
        ("Location Details", {"fields": ("region", "community")}),
        ("Classification", {"fields": ("category", "class_name", "subclass")}),
        (
            "Recording Details",
            {
                "fields": (
                    "time_of_day",
                    "recording_device",
                    "microphone_type",
                    "recording_date",
                )
            },
        ),
    )

    prepopulated_fields = {"name": ("category", "community")}
    readonly_fields = ("noise_id", "updated_at")
    date_hierarchy = "recording_date"

    def has_audio_features(self, obj):
        """Check if the dataset has audio features"""
        return hasattr(obj, "audio_features") and obj.audio_features is not None

    has_audio_features.boolean = True
    has_audio_features.short_description = "Audio Features"

    def has_noise_analysis(self, obj):
        """Check if the dataset has noise analysis"""
        return hasattr(obj, "noise_analysis") and obj.noise_analysis is not None

    has_noise_analysis.boolean = True
    has_noise_analysis.short_description = "Noise Analysis"

    def processing_status(self, obj):
        """Show the processing status of the dataset"""
        if not obj.audio:
            return "❌ No Audio File"
        elif not hasattr(obj, "audio_features") or obj.audio_features is None:
            return "⚠️ Missing Audio Features"
        elif not hasattr(obj, "noise_analysis") or obj.noise_analysis is None:
            return "⚠️ Missing Noise Analysis"
        else:
            return "✅ Complete"

    processing_status.short_description = "Processing Status"

    def reprocess_audio_analysis(self, request, queryset):
        """Admin action to reprocess audio analysis for selected datasets"""
        # Filter to only include datasets with audio files
        datasets_with_audio = queryset.filter(audio__isnull=False)
        dataset_ids = list(datasets_with_audio.values_list("id", flat=True))

        # Count datasets without audio files
        datasets_without_audio = queryset.filter(audio__isnull=True).count()

        if dataset_ids:
            # Create a group of tasks for parallel processing
            task_group = group(
                [process_audio_task.s(dataset_id) for dataset_id in dataset_ids]
            )

            # Execute the group
            result = task_group.apply_async()

            message = f"Started reprocessing audio analysis for {len(dataset_ids)} selected datasets. Task group ID: {result.id}"
            if datasets_without_audio > 0:
                message += (
                    f" ({datasets_without_audio} datasets skipped - no audio files)"
                )

            self.message_user(request, message, messages.SUCCESS)
        else:
            if datasets_without_audio > 0:
                self.message_user(
                    request,
                    f"No datasets with audio files selected for reprocessing. ({datasets_without_audio} datasets have no audio files)",
                    messages.WARNING,
                )
            else:
                self.message_user(
                    request, "No datasets selected for reprocessing.", messages.WARNING
                )

    reprocess_audio_analysis.short_description = (
        "🔄 Reprocess audio analysis for selected datasets"
    )

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path(
                "missing-analysis/",
                self.admin_site.admin_view(self.missing_analysis_view),
                name="data_noisedataset_missing_analysis",
            ),
            path(
                "redo-analysis/",
                self.admin_site.admin_view(self.redo_analysis_view),
                name="data_noisedataset_redo_analysis",
            ),
            path(
                "task-progress/<str:task_id>/",
                self.admin_site.admin_view(self.task_progress_view),
                name="data_noisedataset_task_progress",
            ),
            path(
                "api/task-status/<str:task_id>/",
                self.admin_site.admin_view(self.task_status_api),
                name="data_noisedataset_task_status",
            ),
        ]
        return custom_urls + urls

    def missing_analysis_view(self, request):
        """Admin view to show datasets with missing audio features or noise analysis"""

        # Get datasets missing audio features (only those with audio files)
        datasets_missing_audio_features = NoiseDataset.objects.filter(
            ~Q(audio_features__isnull=False), audio__isnull=False
        ).select_related("collector", "category", "region", "community")

        # Get datasets missing noise analysis (only those with audio files)
        datasets_missing_noise_analysis = NoiseDataset.objects.filter(
            ~Q(noise_analysis__isnull=False), audio__isnull=False
        ).select_related("collector", "category", "region", "community")

        # Get datasets missing both (only those with audio files)
        datasets_missing_both = NoiseDataset.objects.filter(
            ~Q(audio_features__isnull=False),
            ~Q(noise_analysis__isnull=False),
            audio__isnull=False,
        ).select_related("collector", "category", "region", "community")

        # Get datasets missing either (only those with audio files)
        datasets_missing_either = (
            NoiseDataset.objects.filter(
                Q(audio_features__isnull=True) | Q(noise_analysis__isnull=True),
                audio__isnull=False,
            )
            .select_related("collector", "category", "region", "community")
            .distinct()
        )

        # Get datasets without audio files for information
        datasets_without_audio = NoiseDataset.objects.filter(
            Q(audio_features__isnull=True) | Q(noise_analysis__isnull=True),
            audio__isnull=True,
        ).select_related("collector", "category", "region", "community")

        context = {
            "title": "Datasets with Missing Audio Analysis",
            "datasets_missing_audio_features": datasets_missing_audio_features,
            "datasets_missing_noise_analysis": datasets_missing_noise_analysis,
            "datasets_missing_both": datasets_missing_both,
            "datasets_missing_either": datasets_missing_either,
            "datasets_without_audio": datasets_without_audio,
            "total_missing_audio_features": datasets_missing_audio_features.count(),
            "total_missing_noise_analysis": datasets_missing_noise_analysis.count(),
            "total_missing_both": datasets_missing_both.count(),
            "total_missing_either": datasets_missing_either.count(),
            "total_without_audio": datasets_without_audio.count(),
            "opts": self.model._meta,
        }

        return render(request, "admin/data/noisedataset/missing_analysis.html", context)

    def redo_analysis_view(self, request):
        """Admin view to reprocess all datasets with missing analysis"""

        if request.method == "POST":
            # Get all datasets missing either audio features or noise analysis AND have audio files
            datasets_to_process = NoiseDataset.objects.filter(
                Q(audio_features__isnull=True) | Q(noise_analysis__isnull=True),
                audio__isnull=False,  # Only process datasets with audio files
            ).values_list("id", flat=True)

            if datasets_to_process:
                dataset_list = list(datasets_to_process)
                total_count = len(dataset_list)

                # Warn user if processing large number of datasets
                if total_count > 100:
                    messages.warning(
                        request,
                        f"⚠️ Large batch detected: {total_count} datasets. "
                        f"This may take a significant amount of time. "
                        f"Estimated time: {total_count * 2} seconds ({total_count * 2 / 60:.1f} minutes).",
                    )

                # Start the bulk reprocessing task with timeout
                task = bulk_reprocess_audio_analysis.apply_async(
                    args=[dataset_list, request.user.id],
                    countdown=1,  # 1 second delay to allow page redirect
                    time_limit=3600 * 6,  # 6 hours timeout
                    soft_time_limit=3600 * 5,  # 5 hours soft timeout
                )

                # Create tracking record
                BulkReprocessingTask.objects.create(
                    task_id=task.id,
                    user=request.user,
                    total_datasets=total_count,
                    status="pending",
                )

                messages.success(
                    request,
                    f"✅ Started bulk reprocessing for {total_count} datasets. "
                    f"Task ID: {task.id}. "
                    f"You can monitor progress on the task tracking page.",
                )

                # Redirect to task progress page
                return HttpResponseRedirect(f"../task-progress/{task.id}/")
            else:
                messages.info(request, "No datasets found that need reprocessing.")
                return HttpResponseRedirect("../missing-analysis/")

        # GET request - show confirmation page
        datasets_to_process = NoiseDataset.objects.filter(
            Q(audio_features__isnull=True) | Q(noise_analysis__isnull=True),
            audio__isnull=False,  # Only show datasets with audio files
        ).select_related("collector", "category", "region", "community")

        # Also get datasets without audio files for information
        datasets_without_audio = NoiseDataset.objects.filter(
            Q(audio_features__isnull=True) | Q(noise_analysis__isnull=True),
            audio__isnull=True,
        ).count()

        total_count = datasets_to_process.count()

        context = {
            "title": "Confirm Reprocessing Audio Analysis",
            "datasets_to_process": datasets_to_process,
            "total_datasets": total_count,
            "datasets_without_audio": datasets_without_audio,
            "estimated_time_minutes": total_count
            * 2
            / 60,  # Rough estimate: 2 seconds per dataset
            "opts": self.model._meta,
        }

        return render(
            request, "admin/data/noisedataset/redo_analysis_confirm.html", context
        )

    def changelist_view(self, request, extra_context=None):
        """Override changelist view to add custom links"""
        extra_context = extra_context or {}

        # Add counts for missing analysis (only for datasets with audio files)
        missing_audio_features_count = NoiseDataset.objects.filter(
            ~Q(audio_features__isnull=False), audio__isnull=False
        ).count()

        missing_noise_analysis_count = NoiseDataset.objects.filter(
            ~Q(noise_analysis__isnull=False), audio__isnull=False
        ).count()

        missing_either_count = (
            NoiseDataset.objects.filter(
                Q(audio_features__isnull=True) | Q(noise_analysis__isnull=True),
                audio__isnull=False,
            )
            .distinct()
            .count()
        )

        extra_context.update(
            {
                "missing_audio_features_count": missing_audio_features_count,
                "missing_noise_analysis_count": missing_noise_analysis_count,
                "missing_either_count": missing_either_count,
            }
        )

        return super().changelist_view(request, extra_context)

    def task_progress_view(self, request, task_id):
        """Admin view to show task progress"""
        try:
            task_record = BulkReprocessingTask.objects.get(task_id=task_id)
        except BulkReprocessingTask.DoesNotExist:
            messages.error(request, f"Task {task_id} not found.")
            return HttpResponseRedirect("../missing-analysis/")

        context = {
            "title": f"Task Progress - {task_id}",
            "task": task_record,
            "opts": self.model._meta,
        }

        return render(request, "admin/data/noisedataset/task_progress.html", context)

    def task_status_api(self, request, task_id):
        """API endpoint to get task status for AJAX updates"""
        try:
            task_record = BulkReprocessingTask.objects.get(task_id=task_id)

            # Try to get current task status from Celery
            from celery.result import AsyncResult

            celery_result = AsyncResult(task_id)

            # Update task record with current progress if available
            if celery_result.state == "PROGRESS" and celery_result.info:
                info = celery_result.info
                task_record.current_progress = info.get("current", 0)
                task_record.progress_percentage = info.get("progress_percentage", 0)
                task_record.current_status_message = info.get("status", "")
                task_record.processed_datasets = info.get("processed", 0)
                task_record.failed_datasets = info.get("failed", 0)
                task_record.failed_dataset_details = info.get("failed_datasets", [])
                task_record.save()

            # Update status based on Celery result
            if celery_result.ready():
                if celery_result.successful():
                    result = celery_result.result
                    if result.get("final_status") == "completed":
                        task_record.status = "completed"
                    elif result.get("final_status") == "completed_with_errors":
                        task_record.status = "completed_with_errors"
                    else:
                        task_record.status = "failed"

                    task_record.result_summary = result.get("status", "")
                    task_record.save()
                else:
                    task_record.status = "failed"
                    task_record.result_summary = str(celery_result.info)
                    task_record.save()

            return JsonResponse(
                {
                    "task_id": task_id,
                    "status": task_record.status,
                    "current_progress": task_record.current_progress,
                    "total_datasets": task_record.total_datasets,
                    "progress_percentage": task_record.progress_percentage,
                    "current_status_message": task_record.current_status_message,
                    "processed_datasets": task_record.processed_datasets,
                    "failed_datasets": task_record.failed_datasets,
                    "failed_dataset_details": task_record.failed_dataset_details,
                    "is_completed": task_record.is_completed,
                    "celery_state": celery_result.state,
                }
            )

        except BulkReprocessingTask.DoesNotExist:
            return JsonResponse({"error": "Task not found"}, status=404)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)


@admin.register(Dataset)
class DatasetAdmin(ModelAdmin):
    pass


@admin.register(AudioFeature)
class AudioFeatureAdmin(ModelAdmin):
    list_display = (
        "id",
        "noise_dataset",
        "duration",
        "sample_rate",
        "spectral_centroid",
    )
    list_filter = ("noise_dataset__category", "noise_dataset__class_name")
    search_fields = ("noise_dataset__name", "noise_dataset__noise_id")
    readonly_fields = ("noise_dataset",)
    list_select_related = ("noise_dataset",)


@admin.register(NoiseAnalysis)
class NoiseAnalysisAdmin(ModelAdmin):
    list_display = (
        "id",
        "noise_dataset",
        "mean_db",
        "max_db",
        "min_db",
        "dominant_frequency",
    )
    list_filter = ("noise_dataset__category", "noise_dataset__class_name")
    search_fields = ("noise_dataset__name", "noise_dataset__noise_id")
    readonly_fields = ("noise_dataset",)
    list_select_related = ("noise_dataset",)


@admin.register(VisualizationPreset)
class VisualizationPresetAdmin(ModelAdmin):
    list_display = ("name", "chart_type", "high_contrast")
    list_filter = ("chart_type", "high_contrast")
    search_fields = ("name", "description")
    filter_horizontal = ()


@admin.register(BulkReprocessingTask)
class BulkReprocessingTaskAdmin(ModelAdmin):
    list_display = (
        "task_id",
        "user",
        "status",
        "total_datasets",
        "processed_datasets",
        "failed_datasets",
        "progress_percentage",
        "created_at",
    )
    list_filter = ("status", "created_at")
    search_fields = ("task_id", "user__username")
    readonly_fields = (
        "task_id",
        "created_at",
        "updated_at",
        "current_progress",
        "progress_percentage",
        "current_status_message",
        "failed_dataset_details",
        "result_summary",
    )

    fieldsets = (
        (
            "Task Information",
            {"fields": ("task_id", "user", "status", "created_at", "updated_at")},
        ),
        (
            "Progress",
            {
                "fields": (
                    "total_datasets",
                    "processed_datasets",
                    "failed_datasets",
                    "current_progress",
                    "progress_percentage",
                    "current_status_message",
                )
            },
        ),
        ("Results", {"fields": ("failed_dataset_details", "result_summary")}),
    )

    def has_add_permission(self, request):
        return False  # Tasks should only be created by the system
