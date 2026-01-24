import os
import uuid
import logging

from celery import shared_task
from django.utils import timezone

from .models import (
    Document,
    DocumentChunk,
    ChunkMetadata,
)
from .services import DocumentProcessor, RAGService

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=3)
def process_document_task(self, document_id: str):
    """
    Celery task to process an uploaded document

    Args:
        document_id: UUID of the document to process
    """
    try:
        document = Document.objects.get(id=document_id)
        document.processing_started_at = timezone.now()
        document.save(update_fields=["processing_started_at"])

        logger.info(f"Processing document {document_id}: {document.title}")

        doc_processor = DocumentProcessor()

        # ===============================
        # READ FILE FROM DJANGO STORAGE
        # ===============================
        logger.info(f"Reading file content for document {document_id}")
        file_content = document.file.read()
        logger.info(f"File content read successfully, size: {len(file_content)} bytes")

        # ===============================
        # CREATE TEMP FILE FOR PROCESSOR
        # ===============================
        temp_filename = f"temp_doc_{uuid.uuid4().hex}.{document.file_type}"
        temp_path = os.path.join("/tmp", temp_filename)

        logger.info(f"Creating temp file at {temp_path}")
        with open(temp_path, "wb") as temp_file:
            temp_file.write(file_content)

        logger.info("Temp file created successfully")

        # ===============================
        # PROCESS DOCUMENT
        # ===============================
        result = doc_processor.process_document(
            file_path=temp_path,
            doc_id=str(document_id),
            title=document.title,
        )

        logger.info("Document processing completed successfully")

        # ===============================
        # CLEAN UP TEMP FILE
        # ===============================
        os.unlink(temp_path)
        logger.info("Temp file cleaned up")

        # ===============================
        # UPDATE DOCUMENT METADATA
        # ===============================
        document.content_hash = result["content_hash"]
        document.total_chunks = result["total_chunks"]
        document.processing_metadata = result["file_metadata"]

        # ===============================
        # SAVE CHUNKS
        # ===============================
        chunk_objects = []
        chunk_texts = []
        chunk_metadatas = []

        for chunk_data in result["chunks"]:
            chunk = DocumentChunk(
                document=document,
                chunk_index=chunk_data["chunk_index"],
                content=chunk_data["content"],
                metadata=chunk_data.get("metadata", {}),
            )
            chunk_objects.append(chunk)

            chunk_texts.append(chunk_data["content"])
            chunk_metadatas.append(
                {
                    "doc_id": str(document_id),
                    "title": document.title,
                    "chunk_index": chunk_data["chunk_index"],
                    "total_chunks": result["total_chunks"],
                    **chunk_data.get("metadata", {}),
                }
            )

        DocumentChunk.objects.bulk_create(chunk_objects)
        logger.info(f"Created {len(chunk_objects)} chunks")

        # ===============================
        # VECTOR STORE (TEMP DISABLED)
        # ===============================
        # rag_service = RAGService()
        # vector_ids = rag_service.add_documents(chunk_texts, chunk_metadatas)

        vector_ids = list(range(len(chunk_texts)))  # dummy IDs for now

        chunks = DocumentChunk.objects.filter(document=document).order_by("chunk_index")

        for chunk, vector_id in zip(chunks, vector_ids):
            chunk.vector_id = vector_id

        DocumentChunk.objects.bulk_update(chunks, ["vector_id"])

        # ===============================
        # FINALIZE DOCUMENT
        # ===============================
        document.processed = True
        document.processed_at = timezone.now()
        document.save(
            update_fields=[
                "processed",
                "processed_at",
                "content_hash",
                "total_chunks",
                "processing_metadata",
            ]
        )

        logger.info(f"Successfully processed document {document_id}")

        return {
            "success": True,
            "document_id": str(document_id),
            "chunks_created": len(chunk_objects),
        }

    except Document.DoesNotExist:
        logger.error(f"Document {document_id} not found")
        raise

    except Exception as e:
        logger.error(f"Error processing document {document_id}: {e}")

        try:
            document = Document.objects.get(id=document_id)
            document.error_message = str(e)
            document.save(update_fields=["error_message"])
        except Exception:
            pass

        raise self.retry(
            exc=e,
            countdown=60 * (self.request.retries + 1),
        )


@shared_task
def delete_document_vectors_task(document_id: str):
    """
    Delete vectors for a document
    """
    try:
        logger.info(f"Deleting vectors for document {document_id}")

        rag_service = RAGService()
        rag_service.delete_document(str(document_id))

        logger.info(f"Successfully deleted vectors for document {document_id}")
        return {"success": True, "document_id": str(document_id)}

    except Exception as e:
        logger.error(f"Error deleting vectors for document {document_id}: {e}")
        raise


@shared_task
def cleanup_expired_cache_task():
    """
    Delete expired query cache entries
    """
    from .models import QueryCache

    try:
        deleted_count, _ = QueryCache.objects.filter(
            expires_at__lt=timezone.now()
        ).delete()

        logger.info(f"Cleaned up {deleted_count} expired cache entries")
        return {"success": True, "deleted": deleted_count}

    except Exception as e:
        logger.error(f"Error cleaning up cache: {e}")
        raise


@shared_task
def cleanup_inactive_sessions_task():
    """
    Mark inactive chat sessions
    """
    from .models import ChatSession
    from datetime import timedelta

    try:
        threshold = timezone.now() - timedelta(days=7)

        updated_count = ChatSession.objects.filter(
            updated_at__lt=threshold,
            is_active=True,
        ).update(is_active=False)

        logger.info(f"Marked {updated_count} sessions inactive")
        return {"success": True, "updated": updated_count}

    except Exception as e:
        logger.error(f"Error cleaning up sessions: {e}")
        raise


@shared_task(bind=True)
def reprocess_document_task(self, document_id: str):
    """
    Reprocess a document
    """
    try:
        document = Document.objects.get(id=document_id)

        logger.info(f"Reprocessing document {document_id}: {document.title}")

        rag_service = RAGService()
        rag_service.delete_document(str(document_id))

        DocumentChunk.objects.filter(document=document).delete()

        document.processed = False
        document.processing_started_at = None
        document.processed_at = None
        document.error_message = ""
        document.save()

        process_document_task.delay(str(document_id))

        logger.info(f"Queued reprocessing for document {document_id}")

        return {"success": True, "document_id": str(document_id)}

    except Exception as e:
        logger.error(f"Error reprocessing document {document_id}: {e}")
        raise
