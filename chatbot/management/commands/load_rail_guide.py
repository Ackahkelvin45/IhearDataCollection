"""
Management command to load the RAIL guide (docs/rail.md) into the RAG vector store.

Run: python manage.py load_rail_guide

This reads the rail.md file (about RAIL lab and i-HEAR project), splits it into chunks,
and adds them to the FAISS index so the chatbot can answer questions about RAIL.
"""

import hashlib
import logging
from pathlib import Path

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
from django.core.management.base import BaseCommand
from django.utils import timezone
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

RAIL_GUIDE_TITLE = "RAIL Guide"
RAIL_GUIDE_FILENAME = "rail.md"


class Command(BaseCommand):
    help = "Load the RAIL guide (docs/rail.md) into the RAG vector store for the chatbot"

    def add_arguments(self, parser):
        parser.add_argument(
            "--force",
            action="store_true",
            help="Replace existing RAIL guide even if content hash matches",
        )
        parser.add_argument(
            "--path",
            type=str,
            default=None,
            help="Custom path to rail.md (default: docs/rail.md)",
        )

    def handle(self, *args, **options):
        from chatbot.models import Document, DocumentChunk
        from chatbot.services import RAGService

        force = options["force"]
        custom_path = options["path"]

        if custom_path:
            guide_path = Path(custom_path)
        else:
            base_dir = Path(settings.BASE_DIR)
            guide_path = base_dir / "docs" / RAIL_GUIDE_FILENAME

        if not guide_path.exists():
            self.stderr.write(self.style.ERROR(f"RAIL guide not found at {guide_path}"))
            self.stderr.write(
                "Run 'python manage.py scrape_rail' first to create docs/rail.md, "
                "or specify --path to the markdown file."
            )
            return

        try:
            with open(guide_path, "r", encoding="utf-8") as f:
                content = f.read()
            # PostgreSQL text fields cannot contain NUL bytes
            content = content.replace("\x00", "")
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"Failed to read {guide_path}: {e}"))
            return

        content_hash = hashlib.sha256(content.encode()).hexdigest()

        User = get_user_model()
        user = User.objects.filter(is_superuser=True).first()
        if not user:
            user = User.objects.first()
        if not user:
            self.stderr.write(
                self.style.ERROR("No user found. Create a user (or superuser) first.")
            )
            return

        existing = Document.objects.filter(title=RAIL_GUIDE_TITLE).first()
        if existing:
            if not force and existing.content_hash == content_hash:
                self.stdout.write(
                    self.style.SUCCESS(
                        "RAIL guide unchanged (same content hash). Use --force to replace."
                    )
                )
                return

            rag_service = RAGService()
            rag_service.delete_document(str(existing.id))
            DocumentChunk.objects.filter(document=existing).delete()
            existing.delete()
            self.stdout.write("Removed existing RAIL guide from vector store.")

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

        document = Document(
            title=RAIL_GUIDE_TITLE,
            file_type="md",
            uploaded_by=user,
            content_hash=content_hash,
            total_chunks=len(chunk_texts),
            processed=True,
            processed_at=timezone.now(),
        )
        document.file.save(
            RAIL_GUIDE_FILENAME, ContentFile(content.encode("utf-8")), save=False
        )
        document.save()

        chunk_objects = []
        chunk_texts_for_rag = []
        chunk_metadatas = []

        for idx, text in enumerate(chunk_texts):
            chunk = DocumentChunk(
                document=document,
                chunk_index=idx,
                content=text,
                metadata={"section": "rail", "chunk_index": idx},
            )
            chunk_objects.append(chunk)
            chunk_texts_for_rag.append(text)
            chunk_metadatas.append(
                {
                    "doc_id": str(document.id),
                    "title": RAIL_GUIDE_TITLE,
                    "chunk_index": idx,
                    "total_chunks": len(chunk_texts),
                    "section": "rail",
                }
            )

        DocumentChunk.objects.bulk_create(chunk_objects)

        rag_service = RAGService()
        rag_service.add_documents(chunk_texts_for_rag, chunk_metadatas)

        self.stdout.write(
            self.style.SUCCESS(
                f"Loaded RAIL guide: {len(chunk_texts)} chunks added to RAG vector store."
            )
        )
