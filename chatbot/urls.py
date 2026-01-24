from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

app_name = "chatbot"

# Create router for viewsets
router = DefaultRouter()
router.register(r"documents", views.DocumentViewSet, basename="document")
router.register(r"sessions", views.ChatSessionViewSet, basename="session")
router.register(r"feedback", views.MessageFeedbackViewSet, basename="feedback")

urlpatterns = [
    # Frontend
    path("", views.chatbot_home, name="home"),
    # API routes
    path("api/", include(router.urls)),
    path("api/stats/", views.chatbot_stats, name="stats"),
]
