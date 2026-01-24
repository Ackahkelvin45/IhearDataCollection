from rest_framework import serializers
from .models import (
    Document,
    DocumentChunk,
    ChatSession,
    Message,
    MessageFeedback,
    QueryCache,
)


class DocumentSerializer(serializers.ModelSerializer):
    """Serializer for Document model"""

    uploaded_by_username = serializers.CharField(
        source="uploaded_by.username", read_only=True
    )
    file_url = serializers.SerializerMethodField()
    processing_status = serializers.SerializerMethodField()

    class Meta:
        model = Document
        fields = [
            "id",
            "title",
            "file",
            "file_url",
            "file_type",
            "uploaded_by",
            "uploaded_by_username",
            "uploaded_at",
            "processed",
            "processing_started_at",
            "processed_at",
            "total_chunks",
            "processing_metadata",
            "error_message",
            "processing_status",
        ]
        read_only_fields = [
            "id",
            "uploaded_by",
            "uploaded_at",
            "processed",
            "processing_started_at",
            "processed_at",
            "total_chunks",
            "processing_metadata",
            "error_message",
        ]

    def get_file_url(self, obj):
        if obj.file:
            request = self.context.get("request")
            if request:
                return request.build_absolute_uri(obj.file.url)
            return obj.file.url
        return None

    def get_processing_status(self, obj):
        if obj.error_message:
            return "error"
        elif obj.processed:
            return "completed"
        elif obj.processing_started_at:
            return "processing"
        return "pending"


class DocumentChunkSerializer(serializers.ModelSerializer):
    """Serializer for DocumentChunk model"""

    document_title = serializers.CharField(source="document.title", read_only=True)

    class Meta:
        model = DocumentChunk
        fields = [
            "id",
            "document",
            "document_title",
            "chunk_index",
            "content",
            "metadata",
            "created_at",
        ]
        read_only_fields = ["id", "created_at"]


class MessageSerializer(serializers.ModelSerializer):
    """Serializer for Message model"""

    has_feedback = serializers.SerializerMethodField()

    class Meta:
        model = Message
        fields = [
            "id",
            "session",
            "role",
            "content",
            "created_at",
            "tokens_used",
            "response_time",
            "sources",
            "has_feedback",
        ]
        read_only_fields = [
            "id",
            "created_at",
            "tokens_used",
            "response_time",
            "sources",
        ]

    def get_has_feedback(self, obj):
        return hasattr(obj, "feedback")


class MessageFeedbackSerializer(serializers.ModelSerializer):
    """Serializer for MessageFeedback model"""

    class Meta:
        model = MessageFeedback
        fields = ["id", "message", "rating", "comment", "metadata", "created_at"]
        read_only_fields = ["id", "created_at"]


class ChatSessionSerializer(serializers.ModelSerializer):
    """Serializer for ChatSession model"""

    user_username = serializers.CharField(source="user.username", read_only=True)
    message_count = serializers.SerializerMethodField()
    last_message = serializers.SerializerMethodField()
    document_titles = serializers.SerializerMethodField()

    class Meta:
        model = ChatSession
        fields = [
            "id",
            "user",
            "user_username",
            "title",
            "created_at",
            "updated_at",
            "is_active",
            "settings",
            "documents",
            "document_titles",
            "message_count",
            "last_message",
        ]
        read_only_fields = ["id", "user", "created_at", "updated_at"]

    def get_message_count(self, obj):
        return obj.messages.count()

    def get_last_message(self, obj):
        last_msg = obj.messages.order_by("-created_at").first()
        if last_msg:
            return {
                "role": last_msg.role,
                "content": last_msg.content[:100],
                "created_at": last_msg.created_at,
            }
        return None

    def get_document_titles(self, obj):
        return list(obj.documents.values_list("title", flat=True))


class ChatSessionDetailSerializer(ChatSessionSerializer):
    """Detailed serializer for ChatSession with messages"""

    messages = MessageSerializer(many=True, read_only=True)

    class Meta(ChatSessionSerializer.Meta):
        fields = ChatSessionSerializer.Meta.fields + ["messages"]


class SendMessageSerializer(serializers.Serializer):
    """Serializer for sending a message"""

    message = serializers.CharField(max_length=10000)
    stream = serializers.BooleanField(default=False)


class CreateSessionSerializer(serializers.Serializer):
    """Serializer for creating a chat session"""

    title = serializers.CharField(max_length=255, required=False, default="New Chat")
    document_ids = serializers.ListField(
        child=serializers.UUIDField(), required=False, allow_empty=True
    )


class DocumentUploadSerializer(serializers.Serializer):
    """Serializer for document upload"""

    file = serializers.FileField()
    title = serializers.CharField(max_length=255, required=False)

    def validate_file(self, value):
        """Validate uploaded file"""
        from .services import DocumentProcessor
        from pathlib import Path

        processor = DocumentProcessor()
        is_valid, error_message = processor.validate_file(value)

        if not is_valid:
            raise serializers.ValidationError(error_message)

        return value
