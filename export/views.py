from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import JsonResponse
from django.core.exceptions import ValidationError
from django.conf import settings
from django.utils import timezone
from django.views.generic import ListView
from django.db.models import Q
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
        # First check if we have a completed export in the database
        try:
            export_record = ExportHistory.objects.get(task_id=task_id, user=request.user)
            if export_record.status == 'completed':
                return JsonResponse({
                    'status': 'completed',
                    'download_url': f"/export/download/{export_record.id}/",
                    'file_size': export_record.file_size,
                    'total_files': export_record.total_files
                })
            elif export_record.status == 'failed':
                return JsonResponse({
                    'status': 'failed',
                    'error': export_record.error_message or 'Export failed'
                }, status=500)
        except ExportHistory.DoesNotExist:
            pass  # Continue with Celery task check

        # Fallback to Celery task status
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
    from django.http import HttpResponse, Http404
    from django.shortcuts import redirect
    from django.contrib import messages
    import os
    
    try:
        export_record = ExportHistory.objects.get(id=export_id, user=request.user)
        
        # Log the attempt
        logger.info(f"Download request for export {export_id}: status={export_record.status}, export_name={export_record.export_name}, user={request.user.id}")
        
        if export_record.status != 'completed':
            error_msg = f'Export "{export_record.export_name}" is not ready yet. Status: {export_record.status}'
            logger.warning(error_msg)
            if request.headers.get('Accept') == 'application/json':
                return JsonResponse({'status': 'error', 'error': error_msg}, status=404)
            messages.error(request, error_msg)
            return redirect('export:export_history')

        # For API requests, return the download URL
        if request.headers.get('Accept') == 'application/json' or request.GET.get('format') == 'json':
            return JsonResponse({
                'status': 'success',
                'download_url': f"/export/download/{export_record.id}/",
                'file_size': export_record.file_size_mb
            })

        # For direct browser requests, serve the file
        file_path = os.path.join(settings.MEDIA_ROOT, 'exports', f'user_{request.user.id}', f'{export_record.export_name}.zip')
        
        # Log the path being checked for debugging
        logger.info(f"Attempting to download export {export_id}: checking path {file_path}")
        logger.info(f"MEDIA_ROOT: {settings.MEDIA_ROOT}")
        
        # Check if directory exists
        export_dir = os.path.join(settings.MEDIA_ROOT, 'exports', f'user_{request.user.id}')
        files_in_dir = []
        
        if not os.path.exists(export_dir):
            error_msg = f'Export directory not found. Please contact support.'
            logger.error(f"Export directory does not exist: {export_dir}")
            if request.headers.get('Accept') == 'application/json':
                return JsonResponse({'status': 'error', 'error': error_msg}, status=404)
            messages.error(request, error_msg)
            return redirect('export:export_history')
        else:
            # List files in directory for debugging
            try:
                files_in_dir = os.listdir(export_dir)
                logger.info(f"Files in export directory ({export_dir}): {files_in_dir}")
            except Exception as e:
                logger.error(f"Error listing files in export directory: {str(e)}")

        if not os.path.exists(file_path):
            logger.error(f"Export file not found at path: {file_path}")
            logger.error(f"Available files in directory: {files_in_dir}")
            
            # Try alternative paths (in case export_name was modified)
            alternative_paths = [
                os.path.join(export_dir, f'{export_record.export_name}.zip'),
                os.path.join(settings.MEDIA_ROOT, 'exports', f'{export_record.export_name}.zip'),
            ]
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    logger.info(f"Found file at alternative path: {alt_path}")
                    file_path = alt_path
                    break
            else:
                # File not found anywhere
                error_msg = f'Export file "{export_record.export_name}.zip" not found. The file may have been deleted or the export may have failed.'
                logger.error(f"Export file not found. Expected: {file_path}, Available files: {files_in_dir}")
                
                # Update export status to failed if it was marked as completed but file doesn't exist
                if export_record.status == 'completed':
                    export_record.status = 'failed'
                    export_record.error_message = 'File not found on server'
                    export_record.save()
                    logger.warning(f"Updated export {export_id} status to 'failed' because file was not found")
                
                if request.headers.get('Accept') == 'application/json':
                    return JsonResponse({
                        'status': 'error', 
                        'error': error_msg,
                        'debug_info': {
                            'export_name': export_record.export_name,
                            'user_id': request.user.id,
                            'files_in_dir': files_in_dir
                        }
                    }, status=404)
                messages.error(request, error_msg)
                return redirect('export:export_history')

        # Verify file is readable
        if not os.access(file_path, os.R_OK):
            error_msg = f'Export file exists but cannot be read. Please contact support.'
            logger.error(f"Export file exists but is not readable: {file_path}")
            if request.headers.get('Accept') == 'application/json':
                return JsonResponse({'status': 'error', 'error': error_msg}, status=403)
            messages.error(request, error_msg)
            return redirect('export:export_history')

        # Serve the file
        try:
            with open(file_path, 'rb') as f:
                file_content = f.read()
                response = HttpResponse(file_content, content_type='application/zip')
                response['Content-Disposition'] = f'attachment; filename="{export_record.export_name}.zip"'
                response['Content-Length'] = len(file_content)
                logger.info(f"Successfully serving file: {file_path} ({len(file_content)} bytes)")
                return response
        except IOError as e:
            error_msg = f'Error reading export file: {str(e)}'
            logger.error(f"IOError reading file {file_path}: {str(e)}", exc_info=True)
            if request.headers.get('Accept') == 'application/json':
                return JsonResponse({'status': 'error', 'error': error_msg}, status=500)
            messages.error(request, error_msg)
            return redirect('export:export_history')

    except ExportHistory.DoesNotExist:
        error_msg = f'Export with ID {export_id} not found or you do not have permission to access it.'
        logger.error(f"ExportHistory not found for export_id: {export_id}, user: {request.user.id}")
        if request.headers.get('Accept') == 'application/json':
            return JsonResponse({'status': 'error', 'error': error_msg}, status=404)
        messages.error(request, error_msg)
        return redirect('export:export_history')
    except Exception as e:
        error_msg = f'An unexpected error occurred while downloading the export: {str(e)}'
        logger.error(f"Error downloading export {export_id}: {str(e)}", exc_info=True)
        if request.headers.get('Accept') == 'application/json':
            return JsonResponse({'status': 'error', 'error': error_msg}, status=500)
        messages.error(request, error_msg)
        return redirect('export:export_history')
