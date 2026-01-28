# signals.py
from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import Document
from .tasks import process_document_task
import logging

logger = logging.getLogger(__name__)


@receiver(post_save, sender=Document)
def trigger_document_processing(sender, instance, created, **kwargs):
    """
    Automatically trigger document processing when a new document is uploaded
    """
    # Only process if:
    # 1. This is a new document (created=True)
    # 2. The document has a file
    # 3. The document hasn't been processed yet
    # 4. There's no error message (to avoid reprocessing failed documents)
    if (
        created
        and instance.file
        and not instance.processed
        and not instance.error_message
    ):
        try:
            # Check if file actually exists
            if hasattr(instance.file, "name") and instance.file.name:
                logger.info(
                    f"Triggering processing for document {instance.id}: {instance.title}"
                )
                process_document_task.delay(str(instance.id))
            else:
                logger.warning(
                    f"Document {instance.id} has no file name, skipping processing"
                )
        except Exception as e:
            logger.error(
                f"Error triggering document processing for {instance.id}: {e}"
            )
