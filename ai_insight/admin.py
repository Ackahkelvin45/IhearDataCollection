from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
import json

from ai_insight.models import ChatSession, ChatMessage


@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ["title", "user", "status", "total_messages", "created_at"]
    list_filter = ["status", "created_at"]
    search_fields = ["title", "user__email", "user__username"]
    readonly_fields = [
        "id",
        "total_messages",
        "created_at",
        "updated_at",
        "archived_at",
    ]
    fieldsets = (
        ("Basic Information", {"fields": ("id", "title", "user", "status")}),
        ("Statistics", {"fields": ("total_messages",)}),
        (
            "Timestamps",
            {
                "fields": ("created_at", "updated_at", "archived_at"),
                "classes": ("collapse",),
            },
        ),
    )

    def get_queryset(self, request):
        return super().get_queryset(request).select_related("user")

    actions = ["archive_sessions", "activate_sessions"]

    def archive_sessions(self, request, queryset):
        count = 0
        for session in queryset:
            if session.status == ChatSession.Status.ACTIVE:
                session.archive()
                count += 1
        self.message_user(request, f"Archived {count} sessions.")

    archive_sessions.short_description = "Archive selected sessions"

    def activate_sessions(self, request, queryset):
        count = queryset.filter(status=ChatSession.Status.ARCHIVED).update(
            status=ChatSession.Status.ACTIVE, archived_at=None
        )
        self.message_user(request, f"Activated {count} sessions.")

    activate_sessions.short_description = "Activate selected sessions"


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ["session_title", "status", "processing_time_ms", "created_at"]
    list_filter = ["status", "created_at"]
    search_fields = [
        "user_input",
        "assistant_response",
        "session__title",
        "session__user__email",
    ]
    readonly_fields = [
        "id",
        "created_at",
        "updated_at",
        "processing_time_ms",
        "session_link",
        "formatted_tool_call",
    ]
    fieldsets = (
        ("Basic Information", {"fields": ("id", "session_link", "status")}),
        ("Content", {"fields": ("user_input", "assistant_response")}),
        (
            "Timestamps",
            {"fields": ("created_at", "updated_at"), "classes": ("collapse",)},
        ),
    )

    def get_queryset(self, request):
        return super().get_queryset(request).select_related("session", "session__user")

    def session_title(self, obj):
        return obj.session.title

    session_title.short_description = "Session"
    session_title.admin_order_field = "session__title"

    def session_link(self, obj):
        if obj.session:
            url = reverse("admin:ai_insight_chatsession_change", args=[obj.session.id])
            return format_html('<a href="{}">{}</a>', url, obj.session.title)
        return "-"

    session_link.short_description = "Session"

    def formatted_tool_call(self, obj):
        if obj.tool_call:
            try:
                formatted = json.dumps(obj.tool_call, indent=2)
                return format_html("<pre>{}</pre>", formatted)
            except:
                return str(obj.tool_call)
        return "-"

    formatted_tool_call.short_description = "Tool Call Data"

    def mark_as_completed(self, request, queryset):
        count = queryset.exclude(status=ChatMessage.MessageStatus.COMPLETED).update(
            status=ChatMessage.MessageStatus.COMPLETED
        )
        self.message_user(request, f"Marked {count} messages as completed.")

    mark_as_completed.short_description = "Mark as completed"
