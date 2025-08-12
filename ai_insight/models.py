from django.db import models
from django.conf import settings
from django.core.validators import MaxLengthValidator
from django.utils import timezone
from datetime import timedelta


class ChatSession(models.Model):
    class Status(models.TextChoices):
        ACTIVE = "active", "Active"
        ARCHIVED = "archived", "Archived"
        DELETED = "deleted", "Deleted"

    id = models.BigAutoField(primary_key=True)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="ai_sessions"
    )
    title = models.CharField(max_length=255)
    status = models.CharField(
        max_length=20, choices=Status.choices, default=Status.ACTIVE
    )
    total_messages = models.PositiveIntegerField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    archived_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        permissions = [
            ("use_ai_insight", "Can use AI Insight agent"),
            ("manage_ai_sessions", "Can manage AI insight sessions"),
        ]
        indexes = [
            models.Index(fields=["user", "status"]),
            models.Index(fields=["created_at"]),
        ]

    def __str__(self):
        return f"{self.title} ({self.user})"

    def archive(self):
        from django.utils import timezone

        self.status = self.Status.ARCHIVED
        self.archived_at = timezone.now()
        self.save(update_fields=["status", "archived_at"])

    def increment_message_count(self):
        self.total_messages = models.F("total_messages") + 1
        self.save(update_fields=["total_messages"])


class ChatMessage(models.Model):
    class MessageStatus(models.TextChoices):
        PENDING = "pending", "Pending"
        PROCESSING = "processing", "Processing"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"
        CANCELLED = "cancelled", "Cancelled"

    id = models.BigAutoField(primary_key=True)
    session = models.ForeignKey(
        ChatSession, on_delete=models.CASCADE, related_name="messages"
    )

    user_input = models.TextField(validators=[MaxLengthValidator(10000)])
    assistant_response = models.TextField(blank=True, default="")
    status = models.CharField(
        max_length=20, choices=MessageStatus.choices, default=MessageStatus.PENDING
    )

    tool_call = models.JSONField(null=True, blank=True, default=None)
    processing_time_ms = models.PositiveIntegerField(
        null=True, blank=True, help_text="Processing time in milliseconds"
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["created_at"]
        indexes = [
            models.Index(fields=["session", "created_at"]),
            models.Index(fields=["session", "status"]),
            models.Index(fields=["status"]),
            models.Index(fields=["created_at"]),
        ]

    def __str__(self):
        return f"{self.session.title} - {self.created_at}: {self.user_input[:50]}..."

    def mark_processing(self):
        self.status = self.MessageStatus.PROCESSING
        self.save(update_fields=["status"])

    def mark_completed(self, processing_time_ms=None):
        self.status = self.MessageStatus.COMPLETED
        if processing_time_ms:
            self.processing_time_ms = processing_time_ms
        self.save(update_fields=["status", "processing_time_ms"])


class QueryCacheModel(models.Model):
    query_id = models.CharField(max_length=100, unique=True)
    query_type = models.CharField(max_length=50)
    query_sql = models.TextField()
    result_count = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="query_caches"
    )
    metadata = models.JSONField(default=dict)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["query_id"]),
            models.Index(fields=["created_by", "created_at"]),
            models.Index(fields=["expires_at"]),
        ]

    def __str__(self):
        return f"{self.query_type} - {self.query_id}"

    def save(self, *args, **kwargs):
        if not self.expires_at:
            self.expires_at = timezone.now() + timedelta(hours=1)
        super().save(*args, **kwargs)

    @property
    def is_expired(self):
        return timezone.now() > self.expires_at
