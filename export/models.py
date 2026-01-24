from django.db import models
from authentication.models import CustomUser


class ExportHistory(models.Model):
    """Tracks user export downloads for history and analytics"""

    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("processing", "Processing"),
        ("completed", "Completed"),
        ("failed", "Failed"),
        ("cancelled", "Cancelled"),
    ]

    user = models.ForeignKey(
        CustomUser, on_delete=models.CASCADE, help_text="User who initiated the export"
    )
    export_name = models.CharField(max_length=255, help_text="Name of the export")
    task_id = models.CharField(max_length=255, blank=True, help_text="Celery task ID")

    # Export configuration
    folder_structure = models.JSONField(
        help_text="Folder structure configuration (JSON)"
    )
    audio_structure_template = models.TextField(
        blank=True, help_text="Audio subfolder template"
    )
    split_count = models.IntegerField(
        default=1, help_text="Number of files to split the export into (1, 2, or 3)"
    )

    # Export results
    download_url = models.TextField(
        blank=True, help_text="URL to download the ZIP file (or first file if split)"
    )
    file_size = models.BigIntegerField(
        null=True, help_text="Size of the ZIP file in bytes (or first file if split)"
    )
    total_files = models.IntegerField(
        default=0, help_text="Total number of audio files exported"
    )

    # Split file results (for exports split into multiple files)
    download_urls = models.JSONField(
        blank=True, null=True, help_text="List of download URLs for split exports"
    )
    file_sizes = models.JSONField(
        blank=True, null=True, help_text="List of file sizes for split exports"
    )

    # Filters applied
    category_ids = models.JSONField(
        blank=True, null=True, help_text="Selected category IDs"
    )
    applied_filters = models.JSONField(
        blank=True, null=True, help_text="Applied filters (JSON)"
    )

    # Status and timing
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    # Error tracking
    error_message = models.TextField(
        blank=True, help_text="Error message if export failed"
    )

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "-created_at"]),
            models.Index(fields=["status"]),
            models.Index(fields=["task_id"]),
        ]

    def __str__(self):
        return f"Export {self.export_name} by {self.user.username} - {self.status}"

    @property
    def is_completed(self):
        return self.status in ["completed", "failed", "cancelled"]

    @property
    def is_running(self):
        return self.status in ["pending", "processing"]

    @property
    def duration(self):
        """Calculate export duration if completed"""
        if self.completed_at and self.created_at:
            return self.completed_at - self.created_at
        return None

    @property
    def file_size_mb(self):
        """Return file size in MB"""
        if self.file_size:
            return round(self.file_size / (1024 * 1024), 2)
        return None

    @property
    def total_file_size(self):
        """Return total file size in bytes for all parts"""
        if self.file_sizes and isinstance(self.file_sizes, list):
            return sum(self.file_sizes)
        return self.file_size or 0

    @property
    def total_file_size_mb(self):
        """Return total file size in MB for all parts"""
        total = self.total_file_size
        if total:
            return round(total / (1024 * 1024), 2)
        return None
