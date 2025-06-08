from django.urls import path
from . import views

app_name="data"

urlpatterns = [
    path("", views.view_dashboard, name="dashboard"),
    path("datasetlist/", views.view_datasetlist, name="datasetlist"),  
    path('create/', views.noise_dataset_create, name='noise_dataset_create'),

]