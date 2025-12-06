# tasks.py
from celery import shared_task
from .utils import process_audio_file, safe_process_audio_file
from django.core.files import File
import os
from .models import BulkAudioUpload, NoiseDataset
from django.core.exceptions import ValidationError
import logging
from django.db import transaction
from datetime import time
from .utils import generate_dataset_name, generate_noise_id
from django.contrib.auth import get_user_model
from django.utils import timezone
from django.utils.dateparse import parse_datetime
from django.conf import settings
from celery import group
from django.db.models import Q
from celery.result import AsyncResult

logger = logging.getLogger(__name__)


@shared_task(bind=True, time_limit=3600 * 6, soft_time_limit=3600 * 5)
def export_with_audio_task(self, export_history_id, folder_structure, category_ids, export_name, filters=None):
    """Celery task to create export with audio files"""
    from export.models import ExportHistory
    from authentication.models import CustomUser
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter
    import zipfile
    import shutil
    import os
    import math
    from django.db import connection

    try:
        # Get export history record
        export_history = ExportHistory.objects.get(id=export_history_id)
        export_history.status = 'processing'
        export_history.save()

        user = export_history.user

        # Create export directory
        export_base_dir = os.path.join(settings.MEDIA_ROOT, 'exports', f'user_{user.id}')
        os.makedirs(export_base_dir, exist_ok=True)

        # Build folder structure from JSON
        excel_path = folder_structure.get('excel_path', '')
        audio_path = folder_structure.get('audio_path', '')
        audio_structure_template = export_history.audio_structure_template or ''

        # Create root export directory
        root_export_dir = os.path.join(export_base_dir, f'temp_{export_name}')
        os.makedirs(root_export_dir, exist_ok=True)

        # Create Excel directory structure
        excel_dir = os.path.join(root_export_dir, excel_path)
        os.makedirs(excel_dir, exist_ok=True)

        # Create audio directory structure
        audio_base_dir = os.path.join(root_export_dir, audio_path)
        os.makedirs(audio_base_dir, exist_ok=True)

        # Build queryset
        queryset = NoiseDataset.objects.select_related(
            'category', 'region', 'community', 'class_name', 'subclass',
            'collector', 'dataset_type', 'microphone_type', 'time_of_day'
        )

        # Apply category filter
        if category_ids:
            queryset = queryset.filter(category_id__in=category_ids)

        # Apply other filters (similar to _filtered_noise_queryset)
        if filters:
            if filters.get('search'):
                queryset = queryset.filter(
                    Q(name__icontains=filters['search']) |
                    Q(noise_id__icontains=filters['search']) |
                    Q(description__icontains=filters['search'])
                )
            if filters.get('region'):
                queryset = queryset.filter(region_id=filters['region'])
            if filters.get('community'):
                queryset = queryset.filter(community_id=filters['community'])
            if filters.get('dataset_type'):
                queryset = queryset.filter(dataset_type_id=filters['dataset_type'])
            if filters.get('collector'):
                queryset = queryset.filter(collector_id=filters['collector'])

        total = queryset.count()
        self.update_state(state='PROGRESS', meta={'progress': 0, 'current': 0, 'total': total})

        # Create Excel workbook
        wb = Workbook()
        ws = wb.active
        ws.title = "Audio Datasets"

        # ALL metadata columns
        headers = [
            'Noise ID', 'Name', 'Category', 'Class', 'Subclass', 'Region', 'Community',
            'Collector', 'Recording Device', 'Recording Date', 'Time of Day',
            'Microphone Type', 'Description', 'Dataset Type',
            'Audio File Path', 'Audio File Link',
            'Duration (s)', 'Sample Rate', 'Num Samples', 'RMS Energy',
            'Zero Crossing Rate', 'Spectral Centroid', 'Spectral Bandwidth',
            'Spectral Rolloff', 'Spectral Flatness', 'Harmonic Ratio', 'Percussive Ratio',
            'Mean dB', 'Max dB', 'Min dB', 'Std dB',
            'Peak Count', 'Peak Interval Mean', 'Dominant Frequency (Hz)', 'Frequency Range', 'Event Count'
        ]

        # Style headers
        header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        header_font = Font(bold=True, color="FFFFFF")

        for col_num, header in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col_num)
            cell.value = header
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal='center', vertical='center')

        # Helper function to build audio subfolder path
        def build_audio_subfolder(dataset, template):
            if not template:
                return ''

            path = template
            path = path.replace('{category}', dataset.category.name if dataset.category else 'uncategorized')
            path = path.replace('{class}', dataset.class_name.name if dataset.class_name else 'unclassified')
            path = path.replace('{subclass}', dataset.subclass.name if dataset.subclass else 'unclassified')
            path = path.replace('{region}', dataset.region.name if dataset.region else 'unknown')
            path = path.replace('{community}', dataset.community.name if dataset.community else 'unknown')
            path = path.replace('{collector}', dataset.collector.username if dataset.collector else 'unknown')
            path = path.replace('{time_of_day}', dataset.time_of_day.name if dataset.time_of_day else 'unknown')
            path = path.replace('{recording_date}', dataset.recording_date.strftime('%Y-%m-%d') if dataset.recording_date else 'unknown')
            path = path.replace('{noise_id}', dataset.noise_id or 'unknown')
            path = path.replace('{dataset_type}', dataset.dataset_type.name if dataset.dataset_type else 'unknown')

            import re
            path = re.sub(r'[<>:"|?*]', '_', path)
            path = path.replace('\\', '/').strip('/')

            return path

        # Process datasets in batches
        processed = 0
        for dataset in queryset.iterator(chunk_size=100):
            processed += 1

            # Build audio subdirectory
            subfolder = build_audio_subfolder(dataset, audio_structure_template)
            if subfolder:
                audio_subdir = os.path.join(audio_base_dir, subfolder)
            else:
                audio_subdir = audio_base_dir

            os.makedirs(audio_subdir, exist_ok=True)

            # Download audio file from S3 (READ-ONLY)
            audio_filename = None
            audio_relative_path = None
            if dataset.audio:
                try:
                    source_path = dataset.audio.path if hasattr(dataset.audio, 'path') else None
                    if not source_path and hasattr(dataset.audio, 'url'):
                        source_path = dataset.audio.name

                    if source_path:
                        original_filename = os.path.basename(dataset.audio.name)
                        file_ext = os.path.splitext(original_filename)[1] or '.mp3'
                        audio_filename = f"{dataset.noise_id}{file_ext}"
                        dest_path = os.path.join(audio_subdir, audio_filename)

                        # Download from S3 using storage.open() (READ-ONLY)
                        try:
                            with dataset.audio.storage.open(dataset.audio.name, 'rb') as source_file:
                                with open(dest_path, 'wb') as dest_file:
                                    shutil.copyfileobj(source_file, dest_file)
                        except Exception as s3_error:
                            logger.error(f"S3 download failed for {dataset.noise_id}: {s3_error}")
                            # Fallback: try direct file copy if local
                            if os.path.exists(source_path):
                                shutil.copy2(source_path, dest_path)
                            else:
                                logger.warning(f"Could not download audio for {dataset.noise_id}")

                        # Calculate relative path from Excel location to audio file
                        if audio_filename:
                            excel_dir_normalized = os.path.normpath(excel_dir)
                            audio_file_normalized = os.path.normpath(dest_path)

                            try:
                                audio_relative_path = os.path.relpath(audio_file_normalized, excel_dir_normalized).replace('\\', '/')
                            except ValueError:
                                # If paths are on different drives, use absolute path from root
                                audio_relative_path = os.path.join(audio_path, subfolder, audio_filename).replace('\\', '/') if subfolder else os.path.join(audio_path, audio_filename).replace('\\', '/')
                except Exception as e:
                    logger.error(f"Error downloading audio for {dataset.noise_id}: {e}")

            # Get audio features and analysis (handle missing OneToOne relationships)
            from .models import AudioFeature, NoiseAnalysis
            audio_feature = AudioFeature.objects.filter(noise_dataset=dataset).first()
            noise_analysis = NoiseAnalysis.objects.filter(noise_dataset=dataset).first()

            # Add row to Excel with ALL metadata
            row = [
                dataset.noise_id or '',
                dataset.name or '',
                dataset.category.name if dataset.category else '',
                dataset.class_name.name if dataset.class_name else '',
                dataset.subclass.name if dataset.subclass else '',
                dataset.region.name if dataset.region else '',
                dataset.community.name if dataset.community else '',
                dataset.collector.username if dataset.collector else '',
                dataset.recording_device or '',
                dataset.recording_date.strftime('%Y-%m-%d %H:%M:%S') if dataset.recording_date else '',
                dataset.time_of_day.name if dataset.time_of_day else '',
                dataset.microphone_type.name if dataset.microphone_type else '',
                dataset.description or '',
                dataset.dataset_type.name if dataset.dataset_type else '',
                audio_relative_path or '',
                f'=HYPERLINK("{audio_relative_path}", "Open Audio")' if audio_relative_path else '',
                audio_feature.duration if audio_feature else '',
                audio_feature.sample_rate if audio_feature else '',
                audio_feature.num_samples if audio_feature else '',
                audio_feature.rms_energy if audio_feature else '',
                audio_feature.zero_crossing_rate if audio_feature else '',
                audio_feature.spectral_centroid if audio_feature else '',
                audio_feature.spectral_bandwidth if audio_feature else '',
                audio_feature.spectral_rolloff if audio_feature else '',
                audio_feature.spectral_flatness if audio_feature else '',
                audio_feature.harmonic_ratio if audio_feature else '',
                audio_feature.percussive_ratio if audio_feature else '',
                noise_analysis.mean_db if noise_analysis else '',
                noise_analysis.max_db if noise_analysis else '',
                noise_analysis.min_db if noise_analysis else '',
                noise_analysis.std_db if noise_analysis else '',
                noise_analysis.peak_count if noise_analysis else '',
                noise_analysis.peak_interval_mean if noise_analysis else '',
                noise_analysis.dominant_frequency if noise_analysis else '',
                noise_analysis.frequency_range if noise_analysis else '',
                noise_analysis.event_count if noise_analysis else '',
            ]

            ws.append(row)

            # Update progress every 10 records
            if processed % 10 == 0:
                progress = int((processed / total) * 100)
                self.update_state(state='PROGRESS', meta={
                    'progress': progress,
                    'current': processed,
                    'total': total
                })

        # Auto-adjust column widths
        for col_num, header in enumerate(headers, 1):
            column_letter = get_column_letter(col_num)
            max_length = len(header)
            for row in ws.iter_rows(min_row=2, max_row=ws.max_row, min_col=col_num, max_col=col_num):
                if row[0].value:
                    max_length = max(max_length, len(str(row[0].value)))
            ws.column_dimensions[column_letter].width = min(max_length + 2, 50)

        # Save Excel file
        excel_file_path = os.path.join(excel_dir, f'{export_name}.xlsx')
        wb.save(excel_file_path)

        # Create ZIP file
        zip_path = os.path.join(export_base_dir, f'{export_name}.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(root_export_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, root_export_dir)
                    zipf.write(file_path, arcname)

        # Clean up temporary directory
        shutil.rmtree(root_export_dir)

        # Calculate file size
        file_size = os.path.getsize(zip_path)

        # Create download URL (point to download endpoint)
        download_url = f"/export/download/{export_history.id}/"

        # Update export history
        export_history.status = 'completed'
        export_history.download_url = download_url
        export_history.file_size = file_size
        export_history.total_files = total
        export_history.completed_at = timezone.now()
        export_history.save()

        return {
            'download_url': download_url,
            'file_size': file_size,
            'total_files': total,
            'export_name': export_name
        }

    except Exception as e:
        logger.error(f"Export task failed: {str(e)}", exc_info=True)

        # Update export history with error
        try:
            export_history = ExportHistory.objects.get(id=export_history_id)
            export_history.status = 'failed'
            export_history.error_message = str(e)
            export_history.completed_at = timezone.now()
            export_history.save()
        except:
            pass

        raise


@shared_task
def process_audio_task(noise_dataset_id):
    try:
        instance = NoiseDataset.objects.get(id=noise_dataset_id)
        success = safe_process_audio_file(instance)
        if not success:
            logger.warning(
                f"Audio processing failed for NoiseDataset {noise_dataset_id}"
            )
    except NoiseDataset.DoesNotExist:

        logging.warning(f"NoiseDataset with ID {noise_dataset_id} not found.")


@shared_task
def check_task_revocation(task_id):
    """Check if a task has been revoked - separate from main processing to avoid Numba issues"""
    try:

        result = AsyncResult(task_id)
        return result.revoked()
    except Exception:
        return False


@shared_task(bind=True, time_limit=3600 * 6, soft_time_limit=3600 * 5)
def bulk_reprocess_audio_analysis(self, dataset_ids, user_id=None):
    """
    Bulk reprocess audio analysis for multiple datasets with progress tracking
    Optimized for large datasets (400+ items)
    """
    total_datasets = len(dataset_ids)
    processed_count = 0
    failed_count = 0
    failed_datasets = []

    # Limit failed datasets list to prevent memory issues
    MAX_FAILED_DETAILS = 100

    logger.info(f"Starting bulk reprocessing for {total_datasets} datasets")

    # Update task state
    self.update_state(
        state="PROGRESS",
        meta={
            "current": 0,
            "total": total_datasets,
            "status": "Starting reprocessing...",
            "processed": 0,
            "failed": 0,
            "failed_datasets": [],
        },
    )

    try:
        # Process in smaller batches to prevent memory issues
        batch_size = 50
        for batch_start in range(0, total_datasets, batch_size):
            batch_end = min(batch_start + batch_size, total_datasets)
            batch_ids = dataset_ids[batch_start:batch_end]

            logger.info(
                f"Processing batch {batch_start//batch_size + 1}: datasets {batch_start+1}-{batch_end}"
            )

            for i, dataset_id in enumerate(batch_ids):
                global_index = batch_start + i
                dataset = None  # Initialize dataset variable

                try:
                    # Get the dataset with select_related to reduce DB queries
                    try:
                        dataset = NoiseDataset.objects.select_related(
                            "collector", "category", "region", "community"
                        ).get(id=dataset_id)
                    except NoiseDataset.DoesNotExist:
                        logger.warning(f"Dataset {dataset_id} not found, skipping")
                        failed_count += 1
                        if len(failed_datasets) < MAX_FAILED_DETAILS:
                            failed_datasets.append(
                                {"id": dataset_id, "error": "Dataset not found"}
                            )
                        continue

                    # Check if dataset has audio file
                    if not dataset.audio:
                        logger.warning(
                            f"Dataset {dataset_id} has no audio file, skipping"
                        )
                        failed_count += 1
                        if len(failed_datasets) < MAX_FAILED_DETAILS:
                            failed_datasets.append(
                                {
                                    "id": dataset_id,
                                    "noise_id": dataset.noise_id,
                                    "error": "No audio file",
                                }
                            )
                        continue

                    # Process the audio file
                    logger.info(
                        f"Processing dataset {dataset_id} ({global_index+1}/{total_datasets})"
                    )
                    success = safe_process_audio_file(dataset)

                    if success:
                        processed_count += 1
                        logger.info(f"Successfully processed dataset {dataset_id}")
                    else:
                        failed_count += 1
                        logger.warning(f"Failed to process dataset {dataset_id}")
                        if len(failed_datasets) < MAX_FAILED_DETAILS:
                            failed_datasets.append(
                                {
                                    "id": dataset_id,
                                    "noise_id": (
                                        getattr(dataset, "noise_id", "Unknown")
                                        if dataset
                                        else "Unknown"
                                    ),
                                    "error": "Audio processing failed",
                                }
                            )

                except Exception as e:
                    logger.error(f"Failed to process dataset {dataset_id}: {str(e)}")
                    failed_count += 1
                    if len(failed_datasets) < MAX_FAILED_DETAILS:
                        failed_datasets.append(
                            {
                                "id": dataset_id,
                                "noise_id": (
                                    getattr(dataset, "noise_id", "Unknown")
                                    if dataset
                                    else "Unknown"
                                ),
                                "error": str(e)[:200],  # Limit error message length
                            }
                        )

                # Update progress less frequently for large datasets
                if (global_index + 1) % 10 == 0 or global_index + 1 == total_datasets:
                    progress_percentage = int(
                        ((global_index + 1) / total_datasets) * 100
                    )
                    self.update_state(
                        state="PROGRESS",
                        meta={
                            "current": global_index + 1,
                            "total": total_datasets,
                            "status": f"Processing dataset {global_index+1} of {total_datasets} ({progress_percentage}%)",
                            "processed": processed_count,
                            "failed": failed_count,
                            "failed_datasets": failed_datasets,
                            "progress_percentage": progress_percentage,
                        },
                    )

    except Exception as e:
        logger.error(f"Critical error in bulk reprocessing: {str(e)}")
        # Return partial results
        result = {
            "current": processed_count + failed_count,
            "total": total_datasets,
            "status": f"Failed with error: {str(e)}",
            "processed": processed_count,
            "failed": failed_count,
            "failed_datasets": failed_datasets,
            "progress_percentage": int(
                ((processed_count + failed_count) / total_datasets) * 100
            ),
            "final_status": "failed",
            "error": str(e),
        }
        return result

    # Final state
    final_status = "completed"
    if failed_count > 0 and processed_count == 0:
        final_status = "failed"
    elif failed_count > 0:
        final_status = "completed_with_errors"

    result = {
        "current": total_datasets,
        "total": total_datasets,
        "status": f"Completed: {processed_count} processed, {failed_count} failed",
        "processed": processed_count,
        "failed": failed_count,
        "failed_datasets": failed_datasets,
        "progress_percentage": 100,
        "final_status": final_status,
    }

    logger.info(
        f"Bulk reprocessing completed: {processed_count} processed, {failed_count} failed"
    )
    return result


@shared_task(bind=True)
def process_bulk_upload(self, bulk_upload_id, file_paths, user_id):
    try:
        User = get_user_model()
        user = User.objects.get(id=user_id)

        bulk_upload = BulkAudioUpload.objects.get(id=bulk_upload_id)
        bulk_upload.status = "processing"
        bulk_upload.save()

        # Validate metadata before processing
        try:
            bulk_upload.clean_metadata()
        except ValidationError as e:
            bulk_upload.status = "failed"
            bulk_upload.save()
            logger.error(
                f"Metadata validation failed for bulk upload {bulk_upload_id}: {e}"
            )
            return {
                "bulk_upload_id": bulk_upload_id,
                "processed": 0,
                "failed": len(file_paths),
                "error": str(e),
            }

        metadata = bulk_upload.metadata

        for i, file_path in enumerate(file_paths):
            try:
                # Check for cancellation
                bulk_upload.refresh_from_db()
                if getattr(bulk_upload, "status", "").lower() == "cancelled":
                    logger.info(
                        f"Bulk upload {bulk_upload_id} cancelled. Stopping processing."
                    )
                    break
                # Validate file exists and is accessible
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found at path: {file_path}")

                if not os.access(file_path, os.R_OK):
                    raise PermissionError(f"No read permission for file: {file_path}")

                # Create NoiseDataset instance
                # Parse recording_date to datetime
                recording_dt = metadata["recording_date"]
                if isinstance(recording_dt, str):
                    recording_dt_parsed = parse_datetime(recording_dt)
                    if recording_dt_parsed and timezone.is_naive(recording_dt_parsed):
                        recording_dt_parsed = timezone.make_aware(recording_dt_parsed)
                else:
                    recording_dt_parsed = recording_dt

                noise_dataset = NoiseDataset(
                    collector=user,
                    description=metadata.get("description"),
                    region_id=metadata["region_id"],
                    category_id=metadata["category_id"],
                    time_of_day_id=metadata["time_of_day_id"],
                    community_id=metadata["community_id"],
                    class_name_id=metadata["class_name_id"],
                    subclass_id=metadata.get("subclass_id"),
                    microphone_type_id=metadata.get("microphone_type_id"),
                    recording_date=recording_dt_parsed,
                    recording_device=metadata["recording_device"],
                )

                # Generate IDs and names
                noise_dataset.noise_id = generate_noise_id(user)
                noise_dataset.name = generate_dataset_name(noise_dataset)

                # Validate model before saving (excluding audio field)
                noise_dataset.full_clean(exclude=["audio"])

                # Save the file with proper naming: rename to noise_id + original ext
                original_name = os.path.basename(file_path)
                with open(file_path, "rb") as f:
                    # Derive extension
                    ext = os.path.splitext(original_name)[1] or ""
                    final_name = f"{noise_dataset.noise_id}{ext}"
                    noise_dataset.audio.save(final_name, File(f))

                noise_dataset.save()

                with transaction.atomic():
                    bulk_upload.processed_files += 1
                    bulk_upload.save()

                # Update task state
                self.update_state(
                    state="PROGRESS",
                    meta={
                        "current": i + 1,
                        "total": len(file_paths),
                        "bulk_upload_id": bulk_upload_id,
                    },
                )

            except Exception as e:
                logger.error(
                    f"Failed to process file {file_path}: {str(e)}", exc_info=True
                )
                with transaction.atomic():
                    bulk_upload.failed_files += 1
                    bulk_upload.save()
                continue
            finally:
                # Clean up temporary file
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception as e:
                    logger.error(f"Failed to remove temp file {file_path}: {str(e)}")

        # Update final status
        if bulk_upload.status == "cancelled":
            # Clean up any remaining files for the user
            try:
                upload_dir = os.path.join(
                    settings.SHARED_UPLOADS_DIR, f"user_{user.id}"
                )
                for p in os.listdir(upload_dir):
                    full = os.path.join(upload_dir, p)
                    if os.path.isfile(full):
                        os.remove(full)
            except Exception:
                pass
        else:
            if bulk_upload.failed_files == 0:
                bulk_upload.status = "completed"
            elif bulk_upload.processed_files > 0:
                bulk_upload.status = "completed_with_errors"
            else:
                bulk_upload.status = "failed"

        bulk_upload.save()

        return {
            "bulk_upload_id": bulk_upload_id,
            "processed": bulk_upload.processed_files,
            "failed": bulk_upload.failed_files,
        }

    except Exception as e:
        logger.error(
            f"Critical error processing bulk upload {bulk_upload_id}: {str(e)}",
            exc_info=True,
        )
        try:
            bulk_upload.status = "failed"
            bulk_upload.save()
        except Exception:
            pass
        return {
            "bulk_upload_id": bulk_upload_id,
            "processed": bulk_upload.processed_files,
            "failed": bulk_upload.failed_files,
            "error": str(e),
        }


@shared_task
def cleanup_shared_uploads(days_old=1):
    shared_dir = getattr(settings, "SHARED_UPLOADS_DIR", "/shared_uploads")
    now = time.time()
    cutoff = now - (days_old * 86400)

    for filename in os.listdir(shared_dir):
        filepath = os.path.join(shared_dir, filename)
        if os.path.isfile(filepath):
            file_time = os.path.getmtime(filepath)
            if file_time < cutoff:
                try:
                    os.remove(filepath)
                except Exception as e:
                    logger.error(f"Failed to remove {filepath}: {str(e)}")
