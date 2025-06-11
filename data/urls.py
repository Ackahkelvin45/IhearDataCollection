from django.urls import path
from . import views

app_name="data"

urlpatterns = [
    path("", views.view_dashboard, name="dashboard"),
    path("datasetlist/", views.view_datasetlist, name="datasetlist"),  
    path('dataset-create/', views.noise_dataset_create, name='noise_dataset_create'),
    path('load-classes/', views.load_classes, name='ajax_load_classes'),
    path('load-subclasses/', views.load_subclasses, name='ajax_load_subclasses'),
    path('load-communities/', views.load_communities, name='load_communities'),


]