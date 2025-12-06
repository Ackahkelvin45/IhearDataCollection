from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import JsonResponse
from django.core.exceptions import ValidationError
from django.conf import settings
from django.utils import timezone
from django.views.generic import ListView
from celery.result import AsyncResult
import json
import logging

from .models import ExportHistory
from data.tasks import export_with_audio_task
from data.models import Category

logger = logging.getLogger(__name__)


@login_required
def export_with_audio_view(request):
    """View to initiate export with audio files"""
    if request.method == 'POST':
        try:
            # Get export configuration
            folder_structure_json = request.POST.get('folder_structure', '{}')
            category_ids = request.POST.getlist('categories')
            export_name = request.POST.get('export_name', f'export_{timezone.now().strftime("%Y%m%d_%H%M%S")}')
            audio_structure_template = request.POST.get('audio_structure_template', '')

            # Validate folder structure JSON
            try:
                folder_structure = json.loads(folder_structure_json)
            except:
                return JsonResponse({'status': 'error', 'error': 'Invalid folder structure'}, status=400)

            excel_path = folder_structure.get('excel_path', '')
            audio_path = folder_structure.get('audio_path', '')

            if not excel_path or not audio_path:
                return JsonResponse({'status': 'error', 'error': 'Excel and audio paths are required'}, status=400)

            # Create export history record
            export_history = ExportHistory.objects.create(
                user=request.user,
                export_name=export_name,
                folder_structure=folder_structure,
                audio_structure_template=audio_structure_template,
                category_ids=category_ids if category_ids else None,
                applied_filters=dict(request.GET),  # Store current filters
                status='pending'
            )

            # Create Celery task
            task = export_with_audio_task.delay(
                export_history_id=export_history.id,
                folder_structure=folder_structure,
                category_ids=category_ids,
                export_name=export_name,
                filters=dict(request.GET)
            )

            # Update task_id in history
            export_history.task_id = task.id
            export_history.save()

            return JsonResponse({
                'status': 'success',
                'task_id': task.id,
                'message': 'Export started successfully'
            })
        except Exception as e:
            logger.error(f"Export initiation failed: {str(e)}", exc_info=True)
            return JsonResponse({'status': 'error', 'error': str(e)}, status=500)

    # GET request - return configuration form

    # Only show categories that have at least one dataset
    # Prefetch the dataset count to avoid N+1 queries
    categories_with_data = Category.objects.filter(
        noisedataset__isnull=False
    ).distinct().order_by('name').prefetch_related('noisedataset_set')

    return render(request, 'export/export_with_audio.html', {
        'categories': categories_with_data,
        'now': timezone.now()
    })


@login_required
def export_progress(request, task_id):
    """Check export progress"""
    try:
        task = AsyncResult(task_id)

        if task.ready():
            if task.successful():
                result = task.result
                return JsonResponse({
                    'status': 'completed',
                    'download_url': result.get('download_url'),
                    'file_size': result.get('file_size'),
                    'total_files': result.get('total_files')
                })
            else:
                # Task failed
                return JsonResponse({
                    'status': 'failed',
                    'error': str(task.info)
                }, status=500)
        else:
            # Task is still processing
            progress = task.info.get('progress', 0) if isinstance(task.info, dict) else 0
            return JsonResponse({
                'status': 'processing',
                'progress': progress,
                'current': task.info.get('current', 0) if isinstance(task.info, dict) else 0,
                'total': task.info.get('total', 0) if isinstance(task.info, dict) else 0
            })
    except Exception as e:
        logger.error(f"Progress check failed: {str(e)}")
        return JsonResponse({'status': 'error', 'error': str(e)}, status=500)


class ExportHistoryView(LoginRequiredMixin, ListView):
    model = ExportHistory
    template_name = 'export/export_history.html'
    context_object_name = 'exports'
    ordering = ['-created_at']
    paginate_by = 200

    def get_paginate_by(self, queryset):
        try:
            page_size = int(self.request.GET.get("page_size", self.paginate_by))
        except (TypeError, ValueError):
            page_size = self.paginate_by
        allowed = {50, 100, 200, 300}
        return page_size if page_size in allowed else self.paginate_by

    def get_queryset(self):
        queryset = super().get_queryset()
        queryset = queryset.filter(user=self.request.user)

        # Search functionality
        search = self.request.GET.get("search")
        if search:
            queryset = queryset.filter(
                Q(export_name__icontains=search) |
                Q(task_id__icontains=search)
            )

        # Status filter
        status = self.request.GET.get("status")
        if status:
            queryset = queryset.filter(status=status)

        return queryset

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        # Add current filters to context
        context["current_filters"] = {
            "search": self.request.GET.get("search"),
            "status": self.request.GET.get("status"),
            "page_size": self.request.GET.get("page_size"),
        }

        return context


@login_required
def download_export(request, export_id):
    """Serve the export file for download"""
    try:
        export_record = ExportHistory.objects.get(id=export_id, user=request.user)
        if export_record.status != 'completed' or not export_record.download_url:
            return JsonResponse({'status': 'error', 'error': 'Export not ready'}, status=404)

        # Return the download URL
        return JsonResponse({
            'status': 'success',
            'download_url': export_record.download_url,
            'file_size': export_record.file_size_mb
        })
    except ExportHistory.DoesNotExist:
        return JsonResponse({'status': 'error', 'error': 'Export not found'}, status=404)
