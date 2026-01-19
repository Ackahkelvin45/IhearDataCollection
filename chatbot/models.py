from django.db import models
from django.conf import settings
import uuid


class Document(models.Model):
    """Store uploaded documents for RAG"""

    FILE_TYPE_CHOICES = [
        ("pdf", "PDF"),
        ("docx", "Word Document"),
        ("txt", "Text File"),
        ("md", "Markdown"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to="chatbot/documents/")
    file_type = models.CharField(max_length=10, choices=FILE_TYPE_CHOICES)
    uploaded_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="chatbot_documents"
    )
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)
    processing_started_at = models.DateTimeField(null=True, blank=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    content_hash = models.CharField(max_length=64, blank=True)
    total_chunks = models.IntegerField(default=0)
    processing_metadata = models.JSONField(default=dict, blank=True)
    error_message = models.TextField(blank=True)

    class Meta:
        ordering = ["-uploaded_at"]
        indexes = [
            models.Index(fields=["uploaded_by", "-uploaded_at"]),
            models.Index(fields=["processed"]),
        ]

    def __str__(self):
        return self.title


class DocumentVersion(models.Model):
    """Track document versions for change management"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(
        Document, on_delete=models.CASCADE, related_name="versions"
    )
    version_number = models.IntegerField(null=True, blank=True)
    content_hash = models.CharField(max_length=64)
    created_at = models.DateTimeField(auto_now_add=True)
    changes_description = models.TextField(blank=True)

    class Meta:
        ordering = ["-version_number"]
        unique_together = [["document", "version_number"]]

    def __str__(self):
        return f"{self.document.title} v{self.version_number}"


class DocumentChunk(models.Model):
    """Store document chunks with metadata for vector search"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(
        Document, on_delete=models.CASCADE, related_name="chunks"
    )
    chunk_index = models.IntegerField()
    content = models.TextField()
    vector_id = models.CharField(max_length=255, blank=True)  # ChromaDB ID
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["chunk_index"]
        unique_together = [["document", "chunk_index"]]
        indexes = [
            models.Index(fields=["document", "chunk_index"]),
        ]

    def __str__(self):
        return f"{self.document.title} - Chunk {self.chunk_index}"


class ChunkMetadata(models.Model):
    """Extended metadata for document chunks"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    chunk = models.OneToOneField(
        DocumentChunk, on_delete=models.CASCADE, related_name="extended_metadata"
    )
    page_number = models.IntegerField(null=True, blank=True)
    section_title = models.CharField(max_length=255, blank=True)
    images = models.JSONField(default=list, blank=True)  # Store image references/OCR
    relevance_score = models.FloatField(default=0.0)

    def __str__(self):
        return f"Metadata for {self.chunk}"


class ChatSession(models.Model):
    """Store chat sessions for context"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="chatbot_sessions"
    )
    title = models.CharField(max_length=255, default="New Chat")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    settings = models.JSONField(default=dict, blank=True)  # Store session-specific settings
    documents = models.ManyToManyField(
        Document, blank=True, related_name="chat_sessions"
    )  # Link to relevant documents

    class Meta:
        ordering = ["-updated_at"]
        indexes = [
            models.Index(fields=["user", "-updated_at"]),
            models.Index(fields=["is_active"]),
        ]

    def __str__(self):
        return f"{self.title} - {self.user.username}"


class Message(models.Model):
    """Store individual messages in a session"""

    ROLE_CHOICES = [
        ("user", "User"),
        ("assistant", "Assistant"),
        ("system", "System"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(
        ChatSession, on_delete=models.CASCADE, related_name="messages"
    )
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    tokens_used = models.IntegerField(default=0)
    response_time = models.FloatField(
        default=0.0, help_text="Response time in seconds"
    )
    sources = models.JSONField(default=list, blank=True)  # Store source documents
    chunks_referenced = models.ManyToManyField(
        DocumentChunk, blank=True, related_name="messages"
    )

    class Meta:
        ordering = ["created_at"]
        indexes = [
            models.Index(fields=["session", "created_at"]),
        ]

    def __str__(self):
        return f"{self.role}: {self.content[:50]}..."


class MessageFeedback(models.Model):
    """Store user feedback on messages"""

    RATING_CHOICES = [
        ("positive", "Positive"),
        ("negative", "Negative"),
        ("neutral", "Neutral"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    message = models.OneToOneField(
        Message, on_delete=models.CASCADE, related_name="feedback"
    )
    rating = models.CharField(max_length=10, choices=RATING_CHOICES)
    comment = models.TextField(blank=True)
    metadata = models.JSONField(
        default=dict, blank=True
    )  # Store additional feedback data
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.rating} feedback for message {self.message.id}"


class QueryCache(models.Model):
    """Cache frequent queries for performance"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    query_hash = models.CharField(max_length=64, unique=True, db_index=True)
    query_text = models.TextField()
    response = models.TextField()
    sources = models.JSONField(default=list, blank=True)
    hit_count = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    last_accessed = models.DateTimeField(auto_now=True)
    expires_at = models.DateTimeField()

    class Meta:
        ordering = ["-hit_count"]
        indexes = [
            models.Index(fields=["query_hash"]),
            models.Index(fields=["-hit_count"]),
        ]

    def __str__(self):
        return f"Cache: {self.query_text[:50]}"
