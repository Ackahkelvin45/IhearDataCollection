from django.contrib import admin
from .models import (
    NoiseDataset, 
    AudioFeature, 
    NoiseAnalysis, 
    BulkReprocessingTask,
    VisualizationPreset
)
from unfold.admin import ModelAdmin
from django.shortcuts import render, redirect
from django.contrib import messages
from django.urls import path
from django.http import HttpResponseRedirect
from django.db.models import Q
from .tasks import process_audio_task
from celery import group
import logging


logger = logging.getLogger(__name__)

@admin.register(NoiseDataset)
class NoiseDatasetAdmin(ModelAdmin):
    list_display = ('noise_id', 'name', 'collector', "subclass", 'category', 'region', 'community', 'recording_date', 'updated_at', 'processing_status')
    list_filter = ('category', 'region', 'community', 'time_of_day', 'class_name', "subclass", 'recording_date')
    search_fields = ('noise_id', 'name', 'description', 'community__name', 'collector__username')
    list_editable = ('name',)
    list_select_related = ('collector', 'category', 'region', 'community')
    list_per_page = 25
    actions = ['reprocess_audio_analysis']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('noise_id', 'name', 'collector', 'description', 'audio')
        }),
        ('Location Details', {
            'fields': ('region', 'community')
        }),
        ('Classification', {
            'fields': ('category', 'class_name', 'subclass')
        }),
        ('Recording Details', {
            'fields': ('time_of_day', 'recording_device', 'microphone_type', 'recording_date')
        }),
    )
    
    prepopulated_fields = {'name': ('category', 'community')}
    readonly_fields = ('noise_id', 'updated_at')
    date_hierarchy = 'recording_date'

    def has_audio_features(self, obj):
        """Check if the dataset has audio features"""
        return hasattr(obj, 'audio_features') and obj.audio_features is not None
    has_audio_features.boolean = True
    has_audio_features.short_description = 'Audio Features'

    def has_noise_analysis(self, obj):
        """Check if the dataset has noise analysis"""
        return hasattr(obj, 'noise_analysis') and obj.noise_analysis is not None
    has_noise_analysis.boolean = True
    has_noise_analysis.short_description = 'Noise Analysis'

    def processing_status(self, obj):
        """Show the processing status of the dataset"""
        if not obj.audio:
            return "❌ No Audio File"
        elif not hasattr(obj, 'audio_features') or obj.audio_features is None:
            return "⚠️ Missing Audio Features"
        elif not hasattr(obj, 'noise_analysis') or obj.noise_analysis is None:
            return "⚠️ Missing Noise Analysis"
        else:
            return "✅ Complete"
    processing_status.short_description = 'Processing Status'

    def reprocess_audio_analysis(self, request, queryset):
        """Admin action to reprocess audio analysis for selected datasets"""
        # Filter to only include datasets with audio files
        datasets_with_audio = queryset.filter(audio__isnull=False)
        dataset_ids = list(datasets_with_audio.values_list('id', flat=True))
        
        # Count datasets without audio files
        datasets_without_audio = queryset.filter(audio__isnull=True).count()
        
        if dataset_ids:
            # Create a group of tasks for parallel processing
            task_group = group([
                process_audio_task.s(dataset_id) 
                for dataset_id in dataset_ids
            ])
            
            # Execute the group
            result = task_group.apply_async()
            
            message = f'Started reprocessing audio analysis for {len(dataset_ids)} selected datasets. Task group ID: {result.id}'
            if datasets_without_audio > 0:
                message += f' ({datasets_without_audio} datasets skipped - no audio files)'
            
            self.message_user(
                request,
                message,
                messages.SUCCESS
            )
        else:
            if datasets_without_audio > 0:
                self.message_user(
                    request,
                    f'No datasets with audio files selected for reprocessing. ({datasets_without_audio} datasets have no audio files)',
                    messages.WARNING
                )
            else:
                self.message_user(
                    request,
                    'No datasets selected for reprocessing.',
                    messages.WARNING
                )
    
    reprocess_audio_analysis.short_description = "🔄 Reprocess audio analysis for selected datasets"

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path(
                'missing-analysis/',
                self.admin_site.admin_view(self.missing_analysis_view),
                name='data_noisedataset_missing_analysis',
            ),
            path(
                'redo-analysis/',
                self.admin_site.admin_view(self.redo_analysis_view),
                name='data_noisedataset_redo_analysis',
            ),
            path(
                'progress/<str:task_id>/',
                self.admin_site.admin_view(self.progress_view),
                name='data_noisedataset_progress',
            ),
        ]
        return custom_urls + urls

    def missing_analysis_view(self, request):
        """Admin view to show datasets with missing audio features or noise analysis"""
        
        # Get datasets missing audio features (only those with audio files)
        datasets_missing_audio_features = NoiseDataset.objects.filter(
            ~Q(audio_features__isnull=False),
            audio__isnull=False
        ).select_related('collector', 'category', 'region', 'community')
        
        # Get datasets missing noise analysis (only those with audio files)
        datasets_missing_noise_analysis = NoiseDataset.objects.filter(
            ~Q(noise_analysis__isnull=False),
            audio__isnull=False
        ).select_related('collector', 'category', 'region', 'community')
        
        # Get datasets missing both (only those with audio files)
        datasets_missing_both = NoiseDataset.objects.filter(
            ~Q(audio_features__isnull=False),
            ~Q(noise_analysis__isnull=False),
            audio__isnull=False
        ).select_related('collector', 'category', 'region', 'community')
        
        # Get datasets missing either (only those with audio files)
        datasets_missing_either = NoiseDataset.objects.filter(
            Q(audio_features__isnull=True) | Q(noise_analysis__isnull=True),
            audio__isnull=False
        ).select_related('collector', 'category', 'region', 'community').distinct()
        
        # Get datasets without audio files for information
        datasets_without_audio = NoiseDataset.objects.filter(
            Q(audio_features__isnull=True) | Q(noise_analysis__isnull=True),
            audio__isnull=True
        ).select_related('collector', 'category', 'region', 'community')
        
        # Get active bulk reprocessing tasks
        from .models import BulkReprocessingTask
        active_tasks = BulkReprocessingTask.objects.filter(
            status__in=['pending', 'processing']
        ).order_by('-created_at')[:5]
        
        context = {
            'title': 'Datasets with Missing Audio Analysis',
            'datasets_missing_audio_features': datasets_missing_audio_features,
            'datasets_missing_noise_analysis': datasets_missing_noise_analysis,
            'datasets_missing_both': datasets_missing_both,
            'datasets_missing_either': datasets_missing_either,
            'datasets_without_audio': datasets_without_audio,
            'total_missing_audio_features': datasets_missing_audio_features.count(),
            'total_missing_noise_analysis': datasets_missing_noise_analysis.count(),
            'total_missing_both': datasets_missing_both.count(),
            'total_missing_either': datasets_missing_either.count(),
            'total_without_audio': datasets_without_audio.count(),
            'active_tasks': active_tasks,
            'opts': self.model._meta,
            'site_title': self.admin_site.site_title,
            'site_header': self.admin_site.site_header,
            'has_permission': self.has_view_permission(request),
            'is_popup': False,
            'is_nav_sidebar_enabled': True,
            'available_apps': self.admin_site.get_app_list(request),
        }
        
        # Use the admin site's template response
        from django.template.response import TemplateResponse
        return TemplateResponse(
            request,
            'admin/data/noisedataset/missing_analysis.html',
            context
        )

    def redo_analysis_view(self, request):
        """Admin view to reprocess all datasets with missing analysis"""
        
        if request.method == 'POST':
            # Get all datasets missing either audio features or noise analysis AND have audio files
            datasets_to_process = NoiseDataset.objects.filter(
                Q(audio_features__isnull=True) | Q(noise_analysis__isnull=True),
                audio__isnull=False  # Only process datasets with audio files
            ).values_list('id', flat=True)
            
            if datasets_to_process:
                # Create a unique task ID
                import uuid
                task_id = f"bulk_reprocess_{uuid.uuid4().hex[:8]}"
                
                # Create task record
                from .models import BulkReprocessingTask
                task_record = BulkReprocessingTask.objects.create(
                    task_id=task_id,
                    created_by=request.user,
                    total_datasets=len(datasets_to_process),
                    status='pending'
                )
                
                # Start the bulk reprocessing task
                from .tasks import bulk_reprocess_audio_analysis
                bulk_reprocess_audio_analysis.delay(
                    task_id=task_id,
                    dataset_ids=list(datasets_to_process),
                    user_id=request.user.id
                )
                
                messages.success(
                    request, 
                    f'Started bulk reprocessing for {len(datasets_to_process)} datasets. '
                    f'Task ID: {task_id}. You can track progress in the admin.'
                )
            else:
                messages.info(request, 'No datasets found that need reprocessing.')
            
            return HttpResponseRedirect('../missing-analysis/')
        
        # GET request - show confirmation page
        datasets_to_process = NoiseDataset.objects.filter(
            Q(audio_features__isnull=True) | Q(noise_analysis__isnull=True),
            audio__isnull=False  # Only show datasets with audio files
        ).select_related('collector', 'category', 'region', 'community')
        
        # Also get datasets without audio files for information
        datasets_without_audio = NoiseDataset.objects.filter(
            Q(audio_features__isnull=True) | Q(noise_analysis__isnull=True),
            audio__isnull=True
        ).count()
        
        context = {
            'title': 'Confirm Reprocessing Audio Analysis',
            'datasets_to_process': datasets_to_process,
            'total_datasets': datasets_to_process.count(),
            'datasets_without_audio': datasets_without_audio,
            'opts': self.model._meta,
            'site_title': self.admin_site.site_title,
            'site_header': self.admin_site.site_header,
            'has_permission': self.has_view_permission(request),
            'is_popup': False,
            'is_nav_sidebar_enabled': True,
            'available_apps': self.admin_site.get_app_list(request),
        }
        
        # Use the admin site's template response
        from django.template.response import TemplateResponse
        return TemplateResponse(
            request,
            'admin/data/noisedataset/redo_analysis_confirm.html',
            context
        )

    def progress_view(self, request, task_id):
        """Admin view to show progress of a bulk reprocessing task"""
        from .models import BulkReprocessingTask
        
        try:
            task = BulkReprocessingTask.objects.get(task_id=task_id)
        except BulkReprocessingTask.DoesNotExist:
            messages.error(request, f'Task {task_id} not found.')
            return HttpResponseRedirect('../missing-analysis/')
        
        context = {
            'title': f'Progress - Task {task_id}',
            'task': task,
            'opts': self.model._meta,
            'site_title': self.admin_site.site_title,
            'site_header': self.admin_site.site_header,
            'has_permission': self.has_view_permission(request),
            'is_popup': False,
            'is_nav_sidebar_enabled': True,
            'available_apps': self.admin_site.get_app_list(request),
        }
        
        from django.template.response import TemplateResponse
        return TemplateResponse(
            request,
            'admin/data/noisedataset/progress.html',
            context
        )

    def changelist_view(self, request, extra_context=None):
        """Override changelist view to add custom links"""
        extra_context = extra_context or {}
        
        # Add counts for missing analysis (only for datasets with audio files)
        missing_audio_features_count = NoiseDataset.objects.filter(
            ~Q(audio_features__isnull=False),
            audio__isnull=False
        ).count()
        
        missing_noise_analysis_count = NoiseDataset.objects.filter(
            ~Q(noise_analysis__isnull=False),
            audio__isnull=False
        ).count()
        
        missing_either_count = NoiseDataset.objects.filter(
            Q(audio_features__isnull=True) | Q(noise_analysis__isnull=True),
            audio__isnull=False
        ).distinct().count()
        
        extra_context.update({
            'missing_audio_features_count': missing_audio_features_count,
            'missing_noise_analysis_count': missing_noise_analysis_count,
            'missing_either_count': missing_either_count,
        })
        
        return super().changelist_view(request, extra_context)

@admin.register(AudioFeature)
class AudioFeatureAdmin(ModelAdmin):
    list_display = ('id', 'noise_dataset', 'duration', 'sample_rate', 'spectral_centroid')
    list_filter = ('noise_dataset__category', 'noise_dataset__class_name')
    search_fields = ('noise_dataset__name', 'noise_dataset__noise_id')
    readonly_fields = ('noise_dataset',)
    list_select_related = ('noise_dataset',)

@admin.register(NoiseAnalysis)
class NoiseAnalysisAdmin(ModelAdmin):
    list_display = ('id', 'noise_dataset', 'mean_db', 'max_db', 'min_db', 'dominant_frequency')
    list_filter = ('noise_dataset__category', 'noise_dataset__class_name')
    search_fields = ('noise_dataset__name', 'noise_dataset__noise_id')
    readonly_fields = ('noise_dataset',)
    list_select_related = ('noise_dataset',)

@admin.register(BulkReprocessingTask)
class BulkReprocessingTaskAdmin(ModelAdmin):
    list_display = ('task_id', 'created_by', 'status', 'progress_percentage', 'total_datasets', 'processed_datasets', 'created_at')
    list_filter = ('status', 'created_at')
    search_fields = ('task_id', 'created_by__username')
    readonly_fields = ('task_id', 'created_by', 'created_at', 'updated_at', 'started_at', 'completed_at', 'progress_percentage')
    list_select_related = ('created_by',)
    
    fieldsets = (
        ('Task Information', {
            'fields': ('task_id', 'created_by', 'status', 'created_at', 'updated_at')
        }),
        ('Progress Tracking', {
            'fields': ('total_datasets', 'processed_datasets', 'successful_datasets', 'failed_datasets', 'progress_percentage')
        }),
        ('Timing', {
            'fields': ('started_at', 'completed_at')
        }),
        ('Error Information', {
            'fields': ('error_message', 'failed_dataset_ids'),
            'classes': ('collapse',)
        }),
    )
    
    def has_add_permission(self, request):
        return False  # Tasks should only be created by the system
    
    def has_change_permission(self, request, obj=None):
        return False  # Tasks should not be manually edited
    
    def has_delete_permission(self, request, obj=None):
        return request.user.is_superuser  # Only superusers can delete tasks

@admin.register(VisualizationPreset)
class VisualizationPresetAdmin(ModelAdmin):
    list_display = ('name', 'chart_type', 'high_contrast')
    list_filter = ('chart_type', 'high_contrast')
    search_fields = ('name', 'description')
    filter_horizontal = ()