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
    if request.method == "POST":
        try:
            # Get export configuration
            folder_structure_json = request.POST.get("folder_structure", "{}")
            raw_category_ids = request.POST.getlist("categories")
            export_name = request.POST.get(
                "export_name", f'export_{timezone.now().strftime("%Y%m%d_%H%M%S")}'
            )
            audio_structure_template = request.POST.get("audio_structure_template", "")

            # Parse category IDs - handle split category parts (e.g., "123_part1")
            category_ids = []
            category_parts = {}  # {category_id: part_number}
            
            for cat_value in raw_category_ids:
                if cat_value == "all":
                    category_ids = []  # Empty means all categories
                    category_parts = {}
                    break
                elif "_part" in cat_value:
                    # This is a split category part, e.g., "123_part1"
                    try:
                        cat_id_str, part_str = cat_value.split("_part")
                        cat_id = int(cat_id_str)
                        part_num = int(part_str)
                        category_ids.append(str(cat_id))
                        category_parts[str(cat_id)] = part_num
                    except (ValueError, IndexError):
                        continue
                else:
                    # Regular category ID
                    category_ids.append(cat_value)

            # Validate folder structure JSON
            try:
                folder_structure = json.loads(folder_structure_json)
            except:
                return JsonResponse(
                    {"status": "error", "error": "Invalid folder structure"}, status=400
                )

            excel_path = folder_structure.get("excel_path", "")
            audio_path = folder_structure.get("audio_path", "")

            if not excel_path or not audio_path:
                return JsonResponse(
                    {"status": "error", "error": "Excel and audio paths are required"},
                    status=400,
                )

            # Get split count (default to 1 for single file export)
            try:
                split_count = int(request.POST.get("split_count", 1))
                if split_count not in [1, 2, 3]:
                    split_count = 1
            except (ValueError, TypeError):
                split_count = 1

            # Create export history record
            export_history = ExportHistory.objects.create(
                user=request.user,
                export_name=export_name,
                folder_structure=folder_structure,
                audio_structure_template=audio_structure_template,
                category_ids=category_ids if category_ids else None,
                applied_filters={
                    **dict(request.GET),
                    "category_parts": category_parts,  # Store which parts are selected
                },
                status="pending",
                split_count=split_count,
            )

            # Create Celery task
            task = export_with_audio_task.delay(
                export_history_id=export_history.id,
                folder_structure=folder_structure,
                category_ids=category_ids,
                export_name=export_name,
                filters={
                    **dict(request.GET),
                    "category_parts": category_parts,  # Pass parts info to task
                },
                split_count=split_count,
            )

            # Update task_id in history
            export_history.task_id = task.id
            export_history.save()

            return JsonResponse(
                {
                    "status": "success",
                    "task_id": task.id,
                    "message": "Export started successfully",
                }
            )
        except Exception as e:
            logger.error(f"Export initiation failed: {str(e)}", exc_info=True)
            return JsonResponse({"status": "error", "error": str(e)}, status=500)

    categories_with_data = (
        Category.objects.filter(noisedataset__isnull=False)
        .distinct()
        .order_by("name")
        .prefetch_related("noisedataset_set")
    )
    
    # Build enhanced category list with split options for large categories
    # Threshold for splitting: categories with more than 300 datasets
    SPLIT_THRESHOLD = 300
    enhanced_categories = []
    
    for category in categories_with_data:
        dataset_count = category.noisedataset_set.count()
        
        if dataset_count > SPLIT_THRESHOLD:
            # Split large categories into 3 parts
            third = dataset_count // 3
            remainder = dataset_count % 3
            
            # Calculate ranges for each part
            part1_end = third + (1 if remainder > 0 else 0)
            part2_end = part1_end + third + (1 if remainder > 1 else 0)
            part3_end = dataset_count
            
            enhanced_categories.append({
                'id': category.id,
                'name': category.name,
                'count': dataset_count,
                'is_split': True,
                'parts': [
                    {'part': 1, 'start': 0, 'end': part1_end, 'count': part1_end},
                    {'part': 2, 'start': part1_end, 'end': part2_end, 'count': part2_end - part1_end},
                    {'part': 3, 'start': part2_end, 'end': part3_end, 'count': part3_end - part2_end},
                ]
            })
        else:
            enhanced_categories.append({
                'id': category.id,
                'name': category.name,
                'count': dataset_count,
                'is_split': False,
                'parts': []
            })

    return render(
        request,
        "export/export_with_audio.html",
        {
            "categories": categories_with_data,
            "enhanced_categories": enhanced_categories,
            "now": timezone.now(),
        },
    )


@login_required
def export_progress(request, task_id):
    """Check export progress"""
    try:
        # First check if we have a completed export in the database
        try:
            export_record = ExportHistory.objects.get(
                task_id=task_id, user=request.user
            )
            if export_record.status == "completed":
                # Use the download_url from the record (could be S3 URL or local path)
                download_url = (
                    export_record.download_url
                    or f"/export/download/{export_record.id}/"
                )

                # Check if this is a split export
                response_data = {
                    "status": "completed",
                    "download_url": download_url,
                    "file_size": export_record.file_size,
                    "total_files": export_record.total_files,
                    "split_count": export_record.split_count,
                }

                # Include split file information if available
                if export_record.split_count > 1 and export_record.download_urls:
                    response_data["download_urls"] = export_record.download_urls
                    response_data["file_sizes"] = export_record.file_sizes or []

                return JsonResponse(response_data)
            elif export_record.status == "failed":
                return JsonResponse(
                    {
                        "status": "failed",
                        "error": export_record.error_message or "Export failed",
                    },
                    status=500,
                )
        except ExportHistory.DoesNotExist:
            pass  # Continue with Celery task check

        # Fallback to Celery task status (only for progress tracking, not for results)
        # Results are stored in database, so we only check Celery for progress updates
        try:
            task = AsyncResult(task_id)

            if task.ready():
                # Task completed - check database first (results are stored there)
                # If database doesn't have it, task might have just finished
                try:
                    export_record = ExportHistory.objects.get(
                        task_id=task_id, user=request.user
                    )
                    if export_record.status == "completed":
                        response_data = {
                            "status": "completed",
                            "download_url": export_record.download_url
                            or f"/export/download/{export_record.id}/",
                            "file_size": export_record.file_size,
                            "total_files": export_record.total_files,
                            "split_count": export_record.split_count,
                        }
                        if (
                            export_record.split_count > 1
                            and export_record.download_urls
                        ):
                            response_data["download_urls"] = export_record.download_urls
                            response_data["file_sizes"] = export_record.file_sizes or []
                        return JsonResponse(response_data)
                    elif export_record.status == "failed":
                        return JsonResponse(
                            {
                                "status": "failed",
                                "error": export_record.error_message or "Export failed",
                            },
                            status=500,
                        )
                except ExportHistory.DoesNotExist:
                    # Task finished but database record not updated yet
                    if task.successful():
                        # Task succeeded but database not updated - wait a bit
                        return JsonResponse(
                            {
                                "status": "processing",
                                "progress": 95,
                                "message": "Finalizing export...",
                            }
                        )
                    else:
                        # Task failed
                        error_msg = str(task.info) if task.info else "Export failed"
                        return JsonResponse(
                            {"status": "failed", "error": error_msg}, status=500
                        )
            else:
                # Task is still processing - get progress from Celery
                progress = (
                    task.info.get("progress", 0) if isinstance(task.info, dict) else 0
                )
                response_data = {
                    "status": "processing",
                    "progress": progress,
                    "current": (
                        task.info.get("current", 0)
                        if isinstance(task.info, dict)
                        else 0
                    ),
                    "total": (
                        task.info.get("total", 0)
                        if isinstance(task.info, dict)
                        else 0
                    ),
                }
                
                # For split exports, include info about completed parts
                if isinstance(task.info, dict):
                    completed_parts = task.info.get("completed_parts", 0)
                    total_parts = task.info.get("total_parts", 1)
                    if completed_parts > 0:
                        response_data["completed_parts"] = completed_parts
                        response_data["total_parts"] = total_parts
                        response_data["download_urls"] = task.info.get("download_urls", [])
                        response_data["file_sizes"] = task.info.get("file_sizes", [])
                
                # Also check database for already completed parts
                try:
                    export_record = ExportHistory.objects.get(
                        task_id=task_id, user=request.user
                    )
                    if export_record.download_urls and len(export_record.download_urls) > 0:
                        response_data["completed_parts"] = len(export_record.download_urls)
                        response_data["total_parts"] = export_record.split_count
                        response_data["download_urls"] = export_record.download_urls
                        response_data["file_sizes"] = export_record.file_sizes or []
                except ExportHistory.DoesNotExist:
                    pass
                
                return JsonResponse(response_data)
        except Exception as celery_error:
            # If Celery check fails, try database one more time
            logger.warning(f"Celery task check failed: {str(celery_error)}")
            try:
                export_record = ExportHistory.objects.get(
                    task_id=task_id, user=request.user
                )
                if export_record.status == "completed":
                    response_data = {
                        "status": "completed",
                        "download_url": export_record.download_url
                        or f"/export/download/{export_record.id}/",
                        "file_size": export_record.file_size,
                        "total_files": export_record.total_files,
                        "split_count": export_record.split_count,
                    }
                    if export_record.split_count > 1 and export_record.download_urls:
                        response_data["download_urls"] = export_record.download_urls
                        response_data["file_sizes"] = export_record.file_sizes or []
                    return JsonResponse(response_data)
            except ExportHistory.DoesNotExist:
                pass
            # Return processing status if we can't determine
            return JsonResponse(
                {
                    "status": "processing",
                    "progress": 0,
                    "message": "Checking export status...",
                }
            )
    except Exception as e:
        logger.error(f"Progress check failed: {str(e)}")
        return JsonResponse({"status": "error", "error": str(e)}, status=500)


class ExportHistoryView(LoginRequiredMixin, ListView):
    model = ExportHistory
    template_name = "export/export_history.html"
    context_object_name = "exports"
    ordering = ["-created_at"]
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
                Q(export_name__icontains=search) | Q(task_id__icontains=search)
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

        # Get part number for split exports (1-indexed)
        try:
            part = int(request.GET.get("part", 1))
        except (ValueError, TypeError):
            part = 1

        # Validate part number
        if export_record.split_count > 1:
            if part < 1 or part > export_record.split_count:
                part = 1
        else:
            part = 1  # Single file export

        # Log the attempt
        logger.info(
            f"Download request for export {export_id} (part {part}/{export_record.split_count}): status={export_record.status}, export_name={export_record.export_name}, user={request.user.id}"
        )

        if export_record.status != "completed":
            error_msg = f'Export "{export_record.export_name}" is not ready yet. Status: {export_record.status}'
            logger.warning(error_msg)
            if request.headers.get("Accept") == "application/json":
                return JsonResponse({"status": "error", "error": error_msg}, status=404)
            messages.error(request, error_msg)
            return redirect("export:export_history")

        # Check if S3 is enabled
        use_s3 = hasattr(settings, "USE_S3") and settings.USE_S3

        # Determine the correct download URL for split exports
        if export_record.split_count > 1 and export_record.download_urls:
            # Use the URL for the specific part (0-indexed array)
            part_idx = part - 1
            if part_idx < len(export_record.download_urls):
                download_url = export_record.download_urls[part_idx]
            else:
                download_url = export_record.download_url
        else:
            download_url = export_record.download_url

        # Check if download_url is an S3 URL (starts with http:// or https://)
        is_s3_url = download_url and (
            download_url.startswith("http://") or download_url.startswith("https://")
        )

        # Log current state
        logger.info(
            f"S3 enabled: {use_s3}, download_url: {download_url}, is_s3_url: {is_s3_url}"
        )

        # If S3 is enabled but download_url is not an S3 URL, log warning
        if use_s3 and not is_s3_url:
            logger.warning(
                f"S3 is enabled but download_url is not an S3 URL: {download_url}"
            )
            logger.warning("This might indicate the S3 upload failed. Check task logs.")

        # For API requests, return the download URL(s)
        if (
            request.headers.get("Accept") == "application/json"
            or request.GET.get("format") == "json"
        ):
            response_data = {
                "status": "success",
                "download_url": download_url or f"/export/download/{export_record.id}/",
                "file_size": export_record.file_size_mb,
                "split_count": export_record.split_count,
            }
            if export_record.split_count > 1 and export_record.download_urls:
                response_data["download_urls"] = export_record.download_urls
                response_data["file_sizes"] = export_record.file_sizes or []
            return JsonResponse(response_data)

        # If S3 URL, redirect to it
        if is_s3_url:
            logger.info(f"Redirecting to S3 URL: {download_url}")
            return redirect(download_url)

        # If S3 is enabled but no S3 URL, this is an error
        if use_s3:
            error_msg = f"Export file is not available. The S3 upload may have failed. Please check the export status or contact support."
            logger.error(
                f"S3 is enabled but export {export_id} does not have an S3 URL"
            )
            if request.headers.get("Accept") == "application/json":
                return JsonResponse({"status": "error", "error": error_msg}, status=404)
            messages.error(request, error_msg)
            return redirect("export:export_history")

        # For direct browser requests, serve the file from local storage (only if S3 is not enabled)
        # Ensure we use absolute path
        media_root = (
            os.path.abspath(settings.MEDIA_ROOT)
            if settings.MEDIA_ROOT
            else os.path.abspath("media")
        )

        # Determine the file name based on whether it's a split export
        if export_record.split_count > 1:
            file_name = f"{export_record.export_name}_part{part}of{export_record.split_count}.zip"
        else:
            file_name = f"{export_record.export_name}.zip"

        # Build list of possible file locations to check
        possible_paths = [
            # Primary location
            os.path.join(
                media_root,
                "exports",
                f"user_{request.user.id}",
                file_name,
            ),
            # Alternative locations (Docker, relative paths, etc.)
            os.path.join(
                os.path.abspath("."),
                "media",
                "exports",
                f"user_{request.user.id}",
                file_name,
            ),
            os.path.join(
                os.path.abspath("."),
                "exports",
                f"user_{request.user.id}",
                file_name,
            ),
            os.path.join(
                "/app",
                "media",
                "exports",
                f"user_{request.user.id}",
                file_name,
            ),
            os.path.join(
                "/app",
                "exports",
                f"user_{request.user.id}",
                file_name,
            ),
            # Also try without user directory (in case of misconfiguration)
            os.path.join(media_root, "exports", file_name),
        ]

        # Convert all to absolute paths
        possible_paths = [os.path.abspath(p) for p in possible_paths]

        # Log the paths being checked
        logger.info(
            f"Attempting to download export {export_id} (part {part}): file_name={file_name}"
        )
        logger.info(f"MEDIA_ROOT: {settings.MEDIA_ROOT} (absolute: {media_root})")
        logger.info(f"Checking {len(possible_paths)} possible file locations...")

        # Try to find the file
        file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                logger.info(f"Found export file at: {file_path}")
                break
            else:
                logger.debug(f"File not found at: {path}")

        if not file_path:
            # File not found anywhere - list what we checked
            error_msg = f'Export file "{file_name}" not found. The file may have been deleted or the export may have failed.'
            logger.error(
                f"Export file not found after checking {len(possible_paths)} locations"
            )
            logger.error(f"Checked paths: {possible_paths}")

            # Try to list files in the primary export directory for debugging
            primary_dir = os.path.join(media_root, "exports", f"user_{request.user.id}")
            if os.path.exists(primary_dir):
                try:
                    files_in_dir = os.listdir(primary_dir)
                    logger.error(
                        f"Files in primary export directory ({primary_dir}): {files_in_dir}"
                    )
                except Exception as e:
                    logger.error(f"Error listing files in primary directory: {str(e)}")

            # Update export status to failed if it was marked as completed but file doesn't exist
            if export_record.status == "completed":
                export_record.status = "failed"
                export_record.error_message = "File not found on server"
                export_record.save()
                logger.warning(
                    f"Updated export {export_id} status to 'failed' because file was not found"
                )

            if request.headers.get("Accept") == "application/json":
                return JsonResponse(
                    {
                        "status": "error",
                        "error": error_msg,
                        "debug_info": {
                            "export_name": export_record.export_name,
                            "file_name": file_name,
                            "part": part,
                            "user_id": request.user.id,
                            "checked_paths": possible_paths[
                                :3
                            ],  # Only return first 3 for brevity
                        },
                    },
                    status=404,
                )
            messages.error(request, error_msg)
            return redirect("export:export_history")

        # Verify file is readable
        if not os.access(file_path, os.R_OK):
            error_msg = (
                f"Export file exists but cannot be read. Please contact support."
            )
            logger.error(f"Export file exists but is not readable: {file_path}")
            if request.headers.get("Accept") == "application/json":
                return JsonResponse({"status": "error", "error": error_msg}, status=403)
            messages.error(request, error_msg)
            return redirect("export:export_history")

        # Serve the file
        try:
            with open(file_path, "rb") as f:
                file_content = f.read()
                response = HttpResponse(file_content, content_type="application/zip")
                response["Content-Disposition"] = f'attachment; filename="{file_name}"'
                response["Content-Length"] = len(file_content)
                logger.info(
                    f"Successfully serving file: {file_path} ({len(file_content)} bytes)"
                )
                return response
        except IOError as e:
            error_msg = f"Error reading export file: {str(e)}"
            logger.error(f"IOError reading file {file_path}: {str(e)}", exc_info=True)
            if request.headers.get("Accept") == "application/json":
                return JsonResponse({"status": "error", "error": error_msg}, status=500)
            messages.error(request, error_msg)
            return redirect("export:export_history")

    except ExportHistory.DoesNotExist:
        error_msg = f"Export with ID {export_id} not found or you do not have permission to access it."
        logger.error(
            f"ExportHistory not found for export_id: {export_id}, user: {request.user.id}"
        )
        if request.headers.get("Accept") == "application/json":
            return JsonResponse({"status": "error", "error": error_msg}, status=404)
        messages.error(request, error_msg)
        return redirect("export:export_history")
    except Exception as e:
        error_msg = (
            f"An unexpected error occurred while downloading the export: {str(e)}"
        )
        logger.error(f"Error downloading export {export_id}: {str(e)}", exc_info=True)
        if request.headers.get("Accept") == "application/json":
            return JsonResponse({"status": "error", "error": error_msg}, status=500)
        messages.error(request, error_msg)
        return redirect("export:export_history")
