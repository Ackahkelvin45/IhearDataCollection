import hashlib
import os
import uuid
import logging

from celery import shared_task
from django.utils import timezone
import PyPDF2

from .models import (
    Document,
    DocumentChunk,
    ChunkMetadata,
)
from .services import DocumentProcessor, RAGService

logger = logging.getLogger(__name__)


def _sanitize_for_postgres(text: str) -> str:
    """Remove NUL bytes - PostgreSQL text fields cannot contain them."""
    if not isinstance(text, str):
        return str(text) if text else ""
    return text.replace("\x00", "")


@shared_task(bind=True, max_retries=3)
def process_document_task(self, document_id: str):
    """
    Celery task to process an uploaded document

    Args:
        document_id: UUID of the document to process
    """
    temp_path = None
    try:
        document = Document.objects.get(id=document_id)
        document.processing_started_at = timezone.now()
        document.save(update_fields=["processing_started_at"])

        logger.info(f"Processing document {document_id}: {document.title}")

        # ===============================
        # VALIDATE FILE EXISTS
        # ===============================
        if not document.file:
            error_msg = "Document file is missing"
            logger.error(f"{error_msg} for document {document_id}")
            document.error_message = error_msg
            document.save(update_fields=["error_message"])
            return {"success": False, "error": error_msg}

        doc_processor = DocumentProcessor()

        # ===============================
        # READ FILE FROM DJANGO STORAGE
        # ===============================
        logger.info(f"Reading file content for document {document_id}")
        try:
            # Reset file pointer to beginning
            document.file.seek(0)
            file_content = document.file.read()
            logger.info(
                f"File content read successfully, size: {len(file_content)} bytes"
            )

            # Validate file is not empty
            if len(file_content) == 0:
                error_msg = "Document file is empty"
                logger.error(f"{error_msg} for document {document_id}")
                document.error_message = error_msg
                document.save(update_fields=["error_message"])
                return {"success": False, "error": error_msg}
        except Exception as e:
            error_msg = f"Failed to read document file: {str(e)}"
            logger.error(f"{error_msg} for document {document_id}")
            document.error_message = error_msg
            document.save(update_fields=["error_message"])
            return {"success": False, "error": error_msg}

        # ===============================
        # CREATE TEMP FILE FOR PROCESSOR
        # ===============================
        temp_filename = f"temp_doc_{uuid.uuid4().hex}.{document.file_type}"
        temp_path = os.path.join("/tmp", temp_filename)

        logger.info(f"Creating temp file at {temp_path}")
        try:
            with open(temp_path, "wb") as temp_file:
                temp_file.write(file_content)
            logger.info("Temp file created successfully")
        except Exception as e:
            error_msg = f"Failed to create temporary file: {str(e)}"
            logger.error(f"{error_msg} for document {document_id}")
            document.error_message = error_msg
            document.save(update_fields=["error_message"])
            return {"success": False, "error": error_msg}

        # ===============================
        # VALIDATE PDF FILE (if PDF)
        # ===============================
        if document.file_type == "pdf":
            try:
                # Quick validation: try to read PDF structure
                with open(temp_path, "rb") as test_file:
                    test_reader = PyPDF2.PdfReader(test_file, strict=False)
                    # Just check if we can read the PDF structure
                    _ = len(test_reader.pages)
                logger.info("PDF file validation passed")
            except PyPDF2.errors.PdfReadError as pdf_error:
                error_msg = f"PDF file is corrupted or incomplete: {str(pdf_error)}"
                logger.error(f"{error_msg} for document {document_id}")
                document.error_message = error_msg
                document.save(update_fields=["error_message"])
                # Clean up temp file
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except Exception:
                        pass
                # Don't retry for corrupted PDFs
                return {"success": False, "error": error_msg}
            except Exception as e:
                # Other PDF errors - log but continue
                logger.warning(
                    f"PDF validation warning for document {document_id}: {e}"
                )

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
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                logger.info("Temp file cleaned up")
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {temp_path}: {e}")

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
            content = _sanitize_for_postgres(chunk_data["content"])
            chunk = DocumentChunk(
                document=document,
                chunk_index=chunk_data["chunk_index"],
                content=content,
                metadata=chunk_data.get("metadata", {}),
            )
            chunk_objects.append(chunk)

            chunk_texts.append(content)
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
        # ADD TO VECTOR STORE FOR RAG
        # ===============================
        vectors_added = False
        try:
            from .services import RAGService

            rag_service = RAGService()
            logger.info(
                f"Adding {len(chunk_texts)} chunks to vector store for document {document_id}"
            )
            rag_service.add_documents(chunk_texts, chunk_metadatas)
            vectors_added = True
            logger.info(f"Successfully added document {document_id} to vector store")

            # Update vector_ids for chunks (optional, for tracking)
            chunks = DocumentChunk.objects.filter(document=document).order_by(
                "chunk_index"
            )
            # Note: FAISS doesn't return vector IDs, so we'll just mark them as indexed
            for chunk in chunks:
                if not chunk.vector_id:
                    chunk.vector_id = f"indexed_{chunk.chunk_index}"
            DocumentChunk.objects.bulk_update(chunks, ["vector_id"])
        except Exception as e:
            logger.error(f"Failed to add document {document_id} to vector store: {e}")
            # Don't fail the whole task - document is still processed and chunks are saved
            # Return reflects partial success (chunks saved but not searchable via RAG)

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

        logger.info(
            "Successfully processed document %s (vectors_added=%s)",
            document_id,
            vectors_added,
        )

        return {
            "success": True,
            "document_id": str(document_id),
            "chunks_created": len(chunk_objects),
            "vectors_added": vectors_added,
        }

    except Document.DoesNotExist:
        logger.error(f"Document {document_id} not found")
        # Clean up temp file if it exists
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass
        raise

    except PyPDF2.errors.PdfReadError as pdf_error:
        # PDF corruption errors - don't retry
        error_msg = f"PDF file is corrupted or incomplete: {str(pdf_error)}"
        logger.error(f"{error_msg} for document {document_id}")

        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass

        try:
            document = Document.objects.get(id=document_id)
            document.error_message = error_msg
            document.save(update_fields=["error_message"])
        except Exception:
            pass

        # Don't retry for corrupted PDFs
        return {"success": False, "error": error_msg}

    except Exception as e:
        logger.error(f"Error processing document {document_id}: {e}", exc_info=True)

        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass

        try:
            document = Document.objects.get(id=document_id)
            document.error_message = f"Processing failed: {str(e)}"
            document.save(update_fields=["error_message"])
        except Exception:
            pass

        # Only retry if we haven't exceeded max retries and it's not a file corruption error
        if self.request.retries < self.max_retries:
            raise self.retry(
                exc=e,
                countdown=60 * (self.request.retries + 1),
            )
        else:
            # Max retries exceeded
            logger.error(f"Max retries exceeded for document {document_id}")
            return {
                "success": False,
                "error": f"Processing failed after {self.max_retries} retries: {str(e)}",
            }


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


# ── Knowledge Base URL tasks ────────────────────────────────────────────


@shared_task(bind=True, max_retries=2)
def refresh_kb_url_task(self, kb_url_id: str):
    """
    Fetch a KnowledgeBaseURL, chunk its content, and add to FAISS.

    This is the task that keeps URL-based knowledge fresh.
    Run periodically via Celery Beat or manually via management command.
    """
    from .models import KnowledgeBaseURL
    from .services import RAGService
    from .services.web_fetcher import fetch_page_text

    try:
        kb_url = KnowledgeBaseURL.objects.get(id=kb_url_id)
    except KnowledgeBaseURL.DoesNotExist:
        logger.error(f"KnowledgeBaseURL {kb_url_id} not found")
        return {"success": False, "error": "KB URL not found"}

    if not kb_url.is_active:
        logger.info(f"KnowledgeBaseURL {kb_url_id} is inactive, skipping")
        return {"success": False, "error": "KB URL is inactive"}

    logger.info(f"Refreshing KB URL: {kb_url.title} ({kb_url.url})")

    # 1. Fetch live content
    text, error = fetch_page_text(kb_url.url, max_chars=kb_url.max_chars_per_fetch)
    if error:
        logger.error(f"Failed to fetch {kb_url.url}: {error}")
        kb_url.last_fetch_success = False
        kb_url.last_fetch_error = error
        kb_url.last_fetched_at = timezone.now()
        kb_url.save(
            update_fields=["last_fetch_success", "last_fetch_error", "last_fetched_at"]
        )
        return {"success": False, "error": error}

    # 2. Remove old chunks for this URL from FAISS (rebuild without them).
    #    We use the doc_id prefix to identify web-fetched chunks.
    rag_service = RAGService()
    doc_id_prefix = f"web_{hashlib.md5(kb_url.url.encode()).hexdigest()[:12]}"
    rag_service.delete_document(doc_id_prefix)

    # 3. Chunk and add to FAISS
    chunks = rag_service.text_splitter.split_text(text)
    if not chunks:
        logger.warning(f"No chunks extracted from {kb_url.url}")
        kb_url.last_fetch_success = True
        kb_url.last_fetch_error = ""
        kb_url.last_fetched_at = timezone.now()
        kb_url.total_chunks = 0
        kb_url.total_chars = len(text)
        kb_url.save(
            update_fields=[
                "last_fetch_success",
                "last_fetch_error",
                "last_fetched_at",
                "total_chunks",
                "total_chars",
            ]
        )
        return {"success": True, "chunks": 0, "chars": len(text)}

    texts = []
    metadatas = []
    for i, chunk_text in enumerate(chunks):
        texts.append(chunk_text)
        metadatas.append(
            {
                "doc_id": doc_id_prefix,
                "title": kb_url.title,
                "url": kb_url.url,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "source_type": "kb_url",
                "fetched_from": kb_url.url,
                "kb_url_id": str(kb_url.id),
            }
        )

    rag_service.add_documents(texts, metadatas)

    # 4. Update model
    kb_url.last_fetch_success = True
    kb_url.last_fetch_error = ""
    kb_url.last_fetched_at = timezone.now()
    kb_url.total_chunks = len(chunks)
    kb_url.total_chars = len(text)
    kb_url.save(
        update_fields=[
            "last_fetch_success",
            "last_fetch_error",
            "last_fetched_at",
            "total_chunks",
            "total_chars",
        ]
    )

    logger.info(
        f"Refreshed KB URL {kb_url.title}: {len(chunks)} chunks, {len(text)} chars"
    )
    return {
        "success": True,
        "chunks": len(chunks),
        "chars": len(text),
        "url": kb_url.url,
    }


@shared_task
def refresh_all_active_kb_urls_task():
    """
    Refresh all active KnowledgeBaseURLs that are due for refresh.
    Meant to be run via Celery Beat on a schedule.
    """
    from .models import KnowledgeBaseURL
    from datetime import timedelta

    now = timezone.now()
    refresh_map = {
        "hourly": now - timedelta(hours=1),
        "daily": now - timedelta(days=1),
        "weekly": now - timedelta(weeks=1),
    }

    active_urls = KnowledgeBaseURL.objects.filter(is_active=True)
    refreshed = 0

    for kb_url in active_urls:
        if kb_url.refresh_frequency == "manual":
            continue

        threshold = refresh_map.get(kb_url.refresh_frequency)
        if threshold and kb_url.last_fetched_at and kb_url.last_fetched_at > threshold:
            continue  # Not due yet

        refresh_kb_url_task.delay(str(kb_url.id))
        refreshed += 1

    logger.info(
        f"Queued {refreshed} KB URLs for refresh (out of {active_urls.count()} active)"
    )
    return {"success": True, "queued": refreshed}
