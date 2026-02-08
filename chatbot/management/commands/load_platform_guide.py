"""
Management command to load the Platform Guide (docs/platform_guide.md) into the RAG vector store.

Run: python manage.py load_platform_guide

This reads the markdown file, splits it into chunks, and adds them to the FAISS index
so the chatbot can answer questions about platform navigation and features.
"""

import hashlib
import logging
import os
from pathlib import Path

from django.conf import settings
from django.core.files.base import ContentFile
from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from django.utils import timezone

from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

PLATFORM_GUIDE_TITLE = "Platform Guide"
PLATFORM_GUIDE_FILENAME = "platform_guide.md"


class Command(BaseCommand):
    help = "Load the Platform Guide (docs/platform_guide.md) into the RAG vector store for the chatbot"

    def add_arguments(self, parser):
        parser.add_argument(
            "--force",
            action="store_true",
            help="Replace existing platform guide even if content hash matches",
        )
        parser.add_argument(
            "--path",
            type=str,
            default=None,
            help="Custom path to platform_guide.md (default: docs/platform_guide.md relative to project root)",
        )

    def handle(self, *args, **options):
        from chatbot.models import Document, DocumentChunk
        from chatbot.services import RAGService

        force = options["force"]
        custom_path = options["path"]

        # Resolve path to platform guide
        if custom_path:
            guide_path = Path(custom_path)
        else:
            base_dir = Path(settings.BASE_DIR)
            guide_path = base_dir / "docs" / PLATFORM_GUIDE_FILENAME

        if not guide_path.exists():
            self.stderr.write(
                self.style.ERROR(f"Platform guide not found at {guide_path}")
            )
            self.stderr.write(
                "Create docs/platform_guide.md or specify --path to the markdown file."
            )
            return

        # Read content
        try:
            with open(guide_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"Failed to read {guide_path}: {e}"))
            return

        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Get user for Document.uploaded_by (required field)
        User = get_user_model()
        user = User.objects.filter(is_superuser=True).first()
        if not user:
            user = User.objects.first()
        if not user:
            self.stderr.write(
                self.style.ERROR("No user found. Create a user (or superuser) first.")
            )
            return

        # Get or remove existing platform guide document
        existing = Document.objects.filter(title=PLATFORM_GUIDE_TITLE).first()
        if existing:
            if not force and existing.content_hash == content_hash:
                self.stdout.write(
                    self.style.SUCCESS(
                        "Platform guide unchanged (same content hash). Use --force to replace."
                    )
                )
                return

            # Remove existing from vector store and delete chunks
            rag_service = RAGService()
            rag_service.delete_document(str(existing.id))
            DocumentChunk.objects.filter(document=existing).delete()
            existing.delete()
            self.stdout.write("Removed existing platform guide from vector store.")

        # Chunk the content
        config = settings.CHATBOT
        chunk_config = config.get("RAG", {})
        chunk_size = chunk_config.get("CHUNK_SIZE", 1000)
        chunk_overlap = chunk_config.get("CHUNK_OVERLAP", 200)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )
        chunk_texts = splitter.split_text(content)

        # Create Document record (required for DocumentChunk)
        document = Document(
            title=PLATFORM_GUIDE_TITLE,
            file_type="md",
            uploaded_by=user,
            content_hash=content_hash,
            total_chunks=len(chunk_texts),
            processed=True,
            processed_at=timezone.now(),
        )
        document.file.save(
            PLATFORM_GUIDE_FILENAME, ContentFile(content.encode("utf-8")), save=False
        )
        document.save()

        # Create DocumentChunk records
        chunk_objects = []
        chunk_texts_for_rag = []
        chunk_metadatas = []

        for idx, text in enumerate(chunk_texts):
            chunk = DocumentChunk(
                document=document,
                chunk_index=idx,
                content=text,
                metadata={"section": "platform_guide", "chunk_index": idx},
            )
            chunk_objects.append(chunk)
            chunk_texts_for_rag.append(text)
            chunk_metadatas.append(
                {
                    "doc_id": str(document.id),
                    "title": PLATFORM_GUIDE_TITLE,
                    "chunk_index": idx,
                    "total_chunks": len(chunk_texts),
                    "section": "platform_guide",
                }
            )

        DocumentChunk.objects.bulk_create(chunk_objects)

        # Add to RAG vector store
        rag_service = RAGService()
        rag_service.add_documents(chunk_texts_for_rag, chunk_metadatas)

        self.stdout.write(
            self.style.SUCCESS(
                f"Loaded platform guide: {len(chunk_texts)} chunks added to RAG vector store."
            )
        )
