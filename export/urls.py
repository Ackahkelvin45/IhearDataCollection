from django.urls import path
from . import views

app_name = "export"

urlpatterns = [
    path("", views.export_with_audio_view, name="export_with_audio"),
    path("history/", views.ExportHistoryView.as_view(), name="export_history"),
    path("download/<int:export_id>/", views.download_export, name="download_export"),
    path("api/progress/<str:task_id>/", views.export_progress, name="export_progress"),
]