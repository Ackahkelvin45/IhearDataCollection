# tasks.py
from celery import shared_task
from .models import NoiseDataset
from .utils import process_audio_file

@shared_task
def process_audio_task(noise_dataset_id):
    try:
        instance = NoiseDataset.objects.get(id=noise_dataset_id)
        process_audio_file(instance)
    except NoiseDataset.DoesNotExist:
        # Log and skip if instance was deleted
        import logging
        logging.warning(f"NoiseDataset with ID {noise_dataset_id} not found.")
