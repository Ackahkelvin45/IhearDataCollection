from django.urls import path
from . import views


app_name = "data"

urlpatterns = [
    path("", views.view_dashboard, name="dashboard"),
    path("dashboard/", views.DashboardView.as_view(), name="dashboardapi"),
    path("datasetlist/", views.NoiseDatasetListView.as_view(), name="datasetlist"),
    path("dataset-create/", views.noise_dataset_create, name="noise_dataset_create"),
    path("clean-speech/", views.clean_speech_dataset_list, name="clean_speech_dataset_list"),
    path("clean-speech-dataset-create/", views.contribute_audio, name="clean_speech_dataset_create"),
    path("contribute/", views.contribute_audio, name="contribute_audio"),
    path("save-recording/", views.save_recording, name="save_recording"),
    path(
        "dataset/detail/<int:dataset_id>/",
        views.noise_detail,
        name="noise_dataset_detail",
    ),
    path(
        "noise-dataset/<int:pk>/edit/",
        views.noise_dataset_edit,
        name="noise_dataset_edit",
    ),
    path(
        "dataset/<int:pk>/delete/",
        views.NoiseDatasetDeleteView.as_view(),
        name="noise_dataset_delete",
    ),
    path(
        "clean-speech-dataset/<int:dataset_id>/",
        views.clean_speech_detail,
        name="clean_speech_dataset_detail",
    ),
    path(
        "clean-speech-dataset/<int:pk>/delete/",
        views.CleanSpeechDatasetDeleteView.as_view(),
        name="clean_speech_dataset_delete",
    ),
    path("api/export-data/", views.ExportDataAPIView.as_view(), name="api_export_data"),
    path("bulk-upload/", views.bulk_upload_view, name="bulk_upload"),
    path("api/upload-chunk/", views.upload_chunk, name="upload_chunk"),
    path(
        "api/bulk-upload-progress/<int:bulk_upload_id>/",
        views.bulk_upload_progress,
        name="bulk_upload_progress",
    ),
    path(
        "api/cancel-upload/<int:bulk_upload_id>/",
        views.cancel_upload,
        name="cancel_upload",
    ),
    path("load-categories/", views.load_categories, name="load_categories"),
    path("load-classes/", views.load_classes, name="ajax_load_classes"),
    path("load-subclasses/", views.load_subclasses, name="ajax_load_subclasses"),
    path("load-communities/", views.load_communities, name="load_communities"),
    path("load-clean-speech-categories/", views.load_clean_speech_categories, name="load_clean_speech_categories"),
    path("load-clean-speech-classes/", views.load_clean_speech_classes, name="load_clean_speech_classes"),
    path(
        "api/dataset/<int:dataset_id>/plot/waveform/",
        views.api_waveform,
        name="api_plot_waveform",
    ),
    path(
        "api/dataset/<int:dataset_id>/plot/spectrogram/",
        views.api_spectrogram,
        name="api_plot_spectrogram",
    ),
    path(
        "api/dataset/<int:dataset_id>/plot/mfcc/", views.api_mfcc, name="api_plot_mfcc"
    ),
    path(
        "api/dataset/<int:dataset_id>/plot/freq-features/",
        views.api_freq_features,
        name="api_plot_freq_features",
    ),
]
