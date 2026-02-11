from django.urls import path
from . import views

app_name = "approval"

urlpatterns = [
    path("clean-speech/", views.clean_speech_approval_list, name="clean_speech_approval_list"),
    path("clean-speech/<int:recording_id>/", views.clean_speech_approval_review, name="clean_speech_approval_review"),
]
