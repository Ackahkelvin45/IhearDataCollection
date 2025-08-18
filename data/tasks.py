# tasks.py
from celery import shared_task
from .utils import process_audio_file
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

logger = logging.getLogger(__name__)


@shared_task
def process_audio_task(noise_dataset_id):
    try:
        instance = NoiseDataset.objects.get(id=noise_dataset_id)
        process_audio_file(instance)
    except NoiseDataset.DoesNotExist:
        # Log and skip if instance was deleted
        import logging

        logging.warning(f"NoiseDataset with ID {noise_dataset_id} not found.")


@shared_task(bind=True, time_limit=3600*6, soft_time_limit=3600*5)
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

    def is_task_revoked():
        """Check if the task has been revoked"""
        try:
            # Check if task is revoked via request
            if hasattr(self.request, 'revoked') and self.request.revoked:
                return True
            # Check if task is revoked via AsyncResult
            from celery.result import AsyncResult
            result = AsyncResult(self.request.id)
            return result.revoked()
        except Exception:
            return False

    try:
        # Process in smaller batches to prevent memory issues
        batch_size = 50
        for batch_start in range(0, total_datasets, batch_size):
            batch_end = min(batch_start + batch_size, total_datasets)
            batch_ids = dataset_ids[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start//batch_size + 1}: datasets {batch_start+1}-{batch_end}")
            
            for i, dataset_id in enumerate(batch_ids):
                global_index = batch_start + i
                dataset = None  # Initialize dataset variable
                
                try:
                    # Check if task was revoked
                    if is_task_revoked():
                        logger.info("Bulk reprocessing task was revoked")
                        break

                    # Get the dataset with select_related to reduce DB queries
                    try:
                        dataset = NoiseDataset.objects.select_related(
                            'collector', 'category', 'region', 'community'
                        ).get(id=dataset_id)
                    except NoiseDataset.DoesNotExist:
                        logger.warning(f"Dataset {dataset_id} not found, skipping")
                        failed_count += 1
                        if len(failed_datasets) < MAX_FAILED_DETAILS:
                            failed_datasets.append({"id": dataset_id, "error": "Dataset not found"})
                        continue

                    # Check if dataset has audio file
                    if not dataset.audio:
                        logger.warning(f"Dataset {dataset_id} has no audio file, skipping")
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
                    logger.info(f"Processing dataset {dataset_id} ({global_index+1}/{total_datasets})")
                    process_audio_file(dataset)

                    processed_count += 1
                    logger.info(f"Successfully processed dataset {dataset_id}")

                except Exception as e:
                    logger.error(f"Failed to process dataset {dataset_id}: {str(e)}")
                    failed_count += 1
                    if len(failed_datasets) < MAX_FAILED_DETAILS:
                        failed_datasets.append(
                            {
                                "id": dataset_id,
                                "noise_id": getattr(dataset, "noise_id", "Unknown") if dataset else "Unknown",
                                "error": str(e)[:200],  # Limit error message length
                            }
                        )

                # Update progress less frequently for large datasets
                if (global_index + 1) % 10 == 0 or global_index + 1 == total_datasets:
                    progress_percentage = int(((global_index + 1) / total_datasets) * 100)
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
            
            # Check if task was revoked after each batch
            if is_task_revoked():
                break
                
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
            "progress_percentage": int(((processed_count + failed_count) / total_datasets) * 100),
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
