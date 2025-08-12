from django.urls import path
from . import views


app_name = "data"

urlpatterns = [
    path("", views.view_dashboard, name="dashboard"),
    path("datasetlist/", views.NoiseDatasetListView.as_view(), name="datasetlist"),
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
    path("dataset-create/", views.noise_dataset_create, name="noise_dataset_create"),
    path("load-classes/", views.load_classes, name="ajax_load_classes"),
    path("load-subclasses/", views.load_subclasses, name="ajax_load_subclasses"),
    path("load-communities/", views.load_communities, name="load_communities"),
    path(
        "dataset/<int:pk>/delete/",
        views.NoiseDatasetDeleteView.as_view(),
        name="noise_dataset_delete",
    ),
    path("bulk-upload/", views.bulk_upload_view, name="bulk_upload"),
    path(
        "api/upload-chunk/",
        views.upload_chunk,
        name="upload_chunk",
    ),
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
]
