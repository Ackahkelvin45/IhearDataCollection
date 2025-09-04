from django.shortcuts import render
from django.db.models import Count, Min, Max
from data.models import NoiseDataset, NoiseAnalysis
from core.models import Region, Community


def home(request):
    """Landing page for the insights chat UI."""
    # Order suggestions so the first one maps to a chart
    suggestions = [
        "Which region has the most data collected?",
        "Show me recent 20 data collected",
        "Which data has the highest decibel level?",
        "Which community has the lowest decibel level?",
    ]
    return render(request, "data_insights/home.html", {"suggestions": suggestions})


def session(request):
    """Chat session screen that displays a mock response and visualization.

    This is UI-only. It shows how a query and response would look while
    we integrate an LLM in a future step.
    """
    query = request.GET.get("q", "Which region has the most data collected?")
    # Example recent items for the left panel
    recents = [
        "Which region has the most data collected?",
        "Show me recent 20 data collected",
        "Which data has the highest decibel level?",
        "Which community has the lowest decibel level?",
    ]

    # Decide which visualization to show based on the query
    normalized = (query or "").lower()

    show_chart = False
    show_table = False
    chart = None
    table_headers = []
    table_rows = []
    subtitle = None
    assistant_text = None

    try:
        if "region" in normalized:
            # Chart: distribution of datasets by region
            show_chart = True
            region_counts = (
                NoiseDataset.objects.values("region__name")
                .annotate(count=Count("id"))
                .order_by("-count")
            )
            labels = [r.get("region__name") or "Unknown" for r in region_counts]
            values = [r["count"] for r in region_counts]
            chart = {
                "type": "doughnut",
                "labels": labels,
                "values": values,
                "title": "Recordings by Region",
            }
            subtitle = "The chart below shows dataset distribution per region"
            assistant_text = (
                "Here are the regions ranked by number of recordings collected."
            )

        elif "recent" in normalized or "20" in normalized:
            # Table: 20 most recent datasets
            show_table = True
            table_headers = [
                "Name",
                "Region",
                "Category",
                "Recording Date",
            ]
            recent = NoiseDataset.objects.select_related("region", "category").order_by(
                "-created_at"
            )[:20]
            table_rows = [
                [
                    d.name or d.noise_id,
                    getattr(d.region, "name", "-") if d.region else "-",
                    getattr(d.category, "name", "-") if d.category else "-",
                    d.recording_date.strftime("%d/%m/%Y") if d.recording_date else "-",
                ]
                for d in recent
            ]
            subtitle = "Recent 20 datasets"
            assistant_text = (
                "Here are the 20 most recent recordings added to the dataset."
            )

        elif "highest" in normalized and (
            "db" in normalized or "decibel" in normalized
        ):
            # Table: top datasets by maximum decibel level
            show_table = True
            table_headers = ["Name", "Region", "Max dB", "Recording Date"]
            top = (
                NoiseAnalysis.objects.select_related("noise_dataset__region")
                .exclude(max_db__isnull=True)
                .order_by("-max_db")[:20]
            )
            for a in top:
                d = a.noise_dataset
                table_rows.append(
                    [
                        d.name or d.noise_id,
                        getattr(d.region, "name", "-") if d.region else "-",
                        round(a.max_db, 2) if a.max_db is not None else "-",
                        (
                            d.recording_date.strftime("%d/%m/%Y")
                            if d.recording_date
                            else "-"
                        ),
                    ]
                )
            subtitle = "Top datasets by maximum decibel level"
            assistant_text = (
                "These are the recordings with the highest measured decibel levels."
            )

        elif "community" in normalized and (
            "lowest" in normalized and ("db" in normalized or "decibel" in normalized)
        ):
            # Chart: lowest average decibel by community (bar)
            show_chart = True
            qs = (
                NoiseAnalysis.objects.select_related("noise_dataset__community")
                .exclude(mean_db__isnull=True)
                .values("noise_dataset__community__name")
                .annotate(min_avg=Min("mean_db"))
                .order_by("min_avg")[:10]
            )
            labels = [r.get("noise_dataset__community__name") or "Unknown" for r in qs]
            values = [r["min_avg"] for r in qs]
            chart = {
                "type": "bar",
                "labels": labels,
                "values": values,
                "title": "Communities with Lowest Average dB",
            }
            subtitle = "Communities with the lowest average decibel levels"
            assistant_text = (
                "Lowest average decibel levels by community are shown below."
            )

        else:
            # Default to chart by region
            show_chart = True
            region_counts = (
                NoiseDataset.objects.values("region__name")
                .annotate(count=Count("id"))
                .order_by("-count")
            )
            labels = [r.get("region__name") or "Unknown" for r in region_counts]
            values = [r["count"] for r in region_counts]
            chart = {
                "type": "doughnut",
                "labels": labels,
                "values": values,
                "title": "Recordings by Region",
            }
            subtitle = "The chart below shows dataset distribution per region"
            assistant_text = (
                "Here are the regions ranked by number of recordings collected."
            )

    except Exception:
        # Fail silently to avoid UI errors
        pass

    context = {
        "query": query,
        "recents": recents,
        "page_title": "Customer Insights",
        "show_chart": show_chart,
        "show_table": show_table,
        "chart": chart,
        "table_headers": table_headers,
        "table_rows": table_rows,
        "subtitle": subtitle,
        "assistant_text": assistant_text,
    }
    return render(request, "data_insights/session.html", context)




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

        self.status = self.Status.ARCHIVE
        self.archived_at = timezone.now()
        self.save(update_fields=["status", "archived_at"])

    def increment_message_count(self):
        self.total_messages = models.F("total_messages") + 1
        self.save(update_fields=["total_messages"])



from django.db import models
from django.core.validators import MaxLengthValidator
from django.utils import timezone
from datetime import timedelta
from django.conf import settings

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
