from django.contrib import admin
from unfold.admin import ModelAdmin
from .models import (
    Document,
    DocumentVersion,
    DocumentChunk,
    ChunkMetadata,
    ChatSession,
    Message,
    MessageFeedback,
    QueryCache,
)
from .tasks import process_document_task


@admin.register(Document)
class DocumentAdmin(ModelAdmin):
    list_display = [
        "title",
        "file_type",
        "uploaded_by",
        "uploaded_at",
        "processed",
        "total_chunks",
    ]
    list_filter = ["processed", "file_type", "uploaded_at"]
    search_fields = ["title", "uploaded_by__username"]
    readonly_fields = [
        "id",
        "uploaded_at",
        "processing_started_at",
        "processed_at",
        "content_hash",
        "total_chunks",
        "processing_metadata",
    ]
    fieldsets = (
        (
            "Basic Information",
            {"fields": ("id", "title", "file", "file_type", "uploaded_by")},
        ),
        (
            "Processing Status",
            {
                "fields": (
                    "processed",
                    "processing_started_at",
                    "processed_at",
                    "total_chunks",
                    "error_message",
                )
            },
        ),
        (
            "Metadata",
            {"fields": ("content_hash", "processing_metadata"), "classes": ("collapse",)},
        ),
    )

    def get_queryset(self, request):
        return super().get_queryset(request).select_related("uploaded_by")


@admin.register(DocumentChunk)
class DocumentChunkAdmin(ModelAdmin):
    list_display = ["document", "chunk_index", "vector_id", "created_at"]
    list_filter = ["created_at"]
    search_fields = ["document__title", "content"]
    readonly_fields = ["id", "created_at", "vector_id"]

    def get_queryset(self, request):
        return super().get_queryset(request).select_related("document")


@admin.register(ChatSession)
class ChatSessionAdmin(ModelAdmin):
    list_display = ["title", "user", "created_at", "updated_at", "is_active"]
    list_filter = ["is_active", "created_at", "updated_at"]
    search_fields = ["title", "user__username"]
    readonly_fields = ["id", "created_at", "updated_at"]
    filter_horizontal = ["documents"]

    fieldsets = (
        ("Basic Information", {"fields": ("id", "user", "title", "is_active")}),
        ("Timestamps", {"fields": ("created_at", "updated_at")}),
        ("Documents", {"fields": ("documents",)}),
        ("Settings", {"fields": ("settings",), "classes": ("collapse",)}),
    )

    def get_queryset(self, request):
        return super().get_queryset(request).select_related("user")


@admin.register(Message)
class MessageAdmin(ModelAdmin):
    list_display = [
        "session",
        "role",
        "content_preview",
        "created_at",
        "tokens_used",
        "response_time",
    ]
    list_filter = ["role", "created_at"]
    search_fields = ["content", "session__title"]
    readonly_fields = [
        "id",
        "created_at",
        "tokens_used",
        "response_time",
        "sources",
    ]

    fieldsets = (
        ("Basic Information", {"fields": ("id", "session", "role", "content")}),
        (
            "Performance Metrics",
            {"fields": ("tokens_used", "response_time", "created_at")},
        ),
        ("Sources", {"fields": ("sources",), "classes": ("collapse",)}),
    )

    def content_preview(self, obj):
        return obj.content[:50] + "..." if len(obj.content) > 50 else obj.content

    content_preview.short_description = "Content"

    def get_queryset(self, request):
        return super().get_queryset(request).select_related("session")


@admin.register(MessageFeedback)
class MessageFeedbackAdmin(ModelAdmin):
    list_display = ["message", "rating", "created_at"]
    list_filter = ["rating", "created_at"]
    search_fields = ["comment", "message__content"]
    readonly_fields = ["id", "created_at"]

    def get_queryset(self, request):
        return super().get_queryset(request).select_related("message")


@admin.register(QueryCache)
class QueryCacheAdmin(ModelAdmin):
    list_display = ["query_preview", "hit_count", "created_at", "last_accessed"]
    list_filter = ["created_at", "last_accessed"]
    search_fields = ["query_text"]
    readonly_fields = ["id", "query_hash", "created_at", "last_accessed", "hit_count"]

    def query_preview(self, obj):
        return obj.query_text[:100] + "..." if len(obj.query_text) > 100 else obj.query_text

    query_preview.short_description = "Query"


@admin.register(DocumentVersion)
class DocumentVersionAdmin(ModelAdmin):
    list_display = ["document", "version_number", "created_at"]
    list_filter = ["created_at"]
    search_fields = ["document__title"]
    readonly_fields = ["id", "created_at"]

    def get_queryset(self, request):
        return super().get_queryset(request).select_related("document")


@admin.register(ChunkMetadata)
class ChunkMetadataAdmin(ModelAdmin):
    list_display = ["chunk", "page_number", "section_title", "relevance_score"]
    list_filter = ["page_number"]
    search_fields = ["section_title", "chunk__content"]
    readonly_fields = ["id"]

    def get_queryset(self, request):
        return super().get_queryset(request).select_related("chunk")
