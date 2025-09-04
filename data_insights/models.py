from django.db import models
from django.conf import settings
import uuid
from django.core.validators import MaxLengthValidator
from django.utils import timezone
from datetime import timedelta


class ChatSession(models.Model):
    class Status(models.TextChoices):
        ACTIVE = "active"
        INACTIVE = "inactive"
        ARCHIVED = "archived"
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    session_id = models.UUIDField(unique=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=255)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.ACTIVE)
    total_messages = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    archived_at = models.DateTimeField(null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    def archive(self):
        self.status = self.Status.ARCHIVED
        self.archived_at = self.updated_at
        self.save()

    def increment_total_messages(self): 
        self.total_messages += 1
        self.save()
    


    def __str__(self):
        return self.title
    

  

class ChatMessage(models.Model):
    class MessageStatus(models.TextChoices):
        PENDING="pending"
        PROCESSING="processing"
        COMPLETED="completed"
        FAILED="failed"
        CANCELLED="cancelled"

    id =models.BigAutoField(primary_key=True)
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE,related_name="messages")
    user_input = models.TextField(validators=[MaxLengthValidator(10000)])
    assistant_response = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    visulization = models.JSONField(null=True)
    status=models.CharField(max_length=20, choices=MessageStatus.choices, default=MessageStatus.PENDING)
    tool_called=models.JSONField(null=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return self.user_input






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