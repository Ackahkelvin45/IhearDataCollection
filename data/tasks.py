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


@shared_task(bind=True)
def bulk_reprocess_audio_analysis(self, task_id, dataset_ids, user_id):
    """
    Bulk reprocess audio analysis for multiple datasets with progress tracking
    """
    from .models import BulkReprocessingTask
    from django.utils import timezone

    try:
        # Get the task record
        task_record = BulkReprocessingTask.objects.get(task_id=task_id)
        task_record.status = "processing"
        task_record.started_at = timezone.now()
        task_record.total_datasets = len(dataset_ids)
        task_record.save()

        logger.info(
            f"Starting bulk reprocessing task {task_id} for {len(dataset_ids)} datasets"
        )

        successful_count = 0
        failed_count = 0
        failed_dataset_ids = []

        for i, dataset_id in enumerate(dataset_ids):
            try:
                # Check if task was cancelled
                task_record.refresh_from_db()
                if task_record.status == "cancelled":
                    logger.info(f"Task {task_id} was cancelled")
                    break

                # Process the dataset
                process_audio_task.delay(dataset_id)
                successful_count += 1

                # Update progress
                task_record.processed_datasets = i + 1
                task_record.successful_datasets = successful_count
                task_record.failed_datasets = failed_count
                task_record.save()

                # Update task state for monitoring
                self.update_state(
                    state="PROGRESS",
                    meta={
                        "current": i + 1,
                        "total": len(dataset_ids),
                        "successful": successful_count,
                        "failed": failed_count,
                        "task_id": task_id,
                    },
                )

                logger.info(
                    f"Processed dataset {dataset_id} ({i+1}/{len(dataset_ids)})"
                )

            except Exception as e:
                logger.error(f"Failed to process dataset {dataset_id}: {str(e)}")
                failed_count += 1
                failed_dataset_ids.append(dataset_id)

                # Update progress
                task_record.processed_datasets = i + 1
                task_record.successful_datasets = successful_count
                task_record.failed_datasets = failed_count
                task_record.failed_dataset_ids = failed_dataset_ids
                task_record.save()

        # Mark task as completed
        task_record.status = "completed"
        task_record.completed_at = timezone.now()
        task_record.save()

        logger.info(
            f"Bulk reprocessing task {task_id} completed. "
            f"Successful: {successful_count}, Failed: {failed_count}"
        )

        return {
            "task_id": task_id,
            "total": len(dataset_ids),
            "successful": successful_count,
            "failed": failed_count,
            "failed_dataset_ids": failed_dataset_ids,
        }

    except BulkReprocessingTask.DoesNotExist:
        logger.error(f"BulkReprocessingTask with ID {task_id} not found")
        return {
            "task_id": task_id,
            "error": "Task record not found",
        }
    except Exception as e:
        logger.error(f"Critical error in bulk reprocessing task {task_id}: {str(e)}")

        # Update task record with error
        try:
            task_record = BulkReprocessingTask.objects.get(task_id=task_id)
            task_record.status = "failed"
            task_record.error_message = str(e)
            task_record.completed_at = timezone.now()
            task_record.save()
        except:
            pass

        return {
            "task_id": task_id,
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
