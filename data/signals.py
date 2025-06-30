# signals.py
from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import NoiseDataset
from .tasks import process_audio_task

@receiver(post_save, sender=NoiseDataset)
def trigger_audio_processing(sender, instance, created, **kwargs):
    if not instance.audio:
        return
    process_audio_task.delay(instance.id)
