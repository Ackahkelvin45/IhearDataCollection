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
                # Validate file exists and is accessible
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found at path: {file_path}")

                if not os.access(file_path, os.R_OK):
                    raise PermissionError(f"No read permission for file: {file_path}")

                # Create NoiseDataset instance
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
                    recording_date=metadata["recording_date"],
                    recording_device=metadata["recording_device"],
                )

                # Generate IDs and names
                noise_dataset.noise_id = generate_noise_id(user)
                noise_dataset.name = generate_dataset_name(noise_dataset)

                # Validate model before saving (excluding audio field)
                noise_dataset.full_clean(exclude=["audio"])

                # Save the file with proper naming
                file_name = os.path.basename(file_path)
                with open(file_path, "rb") as f:
                    noise_dataset.audio.save(file_name, File(f))

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
        bulk_upload.status = "failed"
        bulk_upload.save()
        return {
            "bulk_upload_id": bulk_upload_id,
            "processed": bulk_upload.processed_files,
            "failed": bulk_upload.failed_files,
            "error": str(e),
        }


@shared_task
def cleanup_shared_uploads(days_old=1):
    shared_dir = "/shared_uploads"
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
