from django.contrib import admin
from .models import ExportHistory


@admin.register(ExportHistory)
class ExportHistoryAdmin(admin.ModelAdmin):
    list_display = ['export_name', 'user', 'status', 'created_at', 'completed_at', 'total_files', 'file_size_mb']
    list_filter = ['status', 'created_at', 'user']
    search_fields = ['export_name', 'user__username', 'task_id']
    readonly_fields = ['task_id', 'download_url', 'file_size', 'created_at', 'completed_at']
    ordering = ['-created_at']

    fieldsets = (
        ('Basic Info', {
            'fields': ('user', 'export_name', 'status', 'created_at', 'completed_at')
        }),
        ('Configuration', {
            'fields': ('folder_structure', 'audio_structure_template', 'category_ids', 'applied_filters'),
            'classes': ('collapse',)
        }),
        ('Results', {
            'fields': ('download_url', 'file_size', 'total_files', 'task_id'),
            'classes': ('collapse',)
        }),
        ('Error Handling', {
            'fields': ('error_message',),
            'classes': ('collapse',)
        }),
    )

    def has_add_permission(self, request):
        return False  # Only allow creation through the export interface

    def has_delete_permission(self, request, obj=None):
        return request.user.is_superuser
