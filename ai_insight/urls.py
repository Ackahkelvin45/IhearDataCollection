from django.urls import path, include
from rest_framework.routers import DefaultRouter
from ai_insight.views import (
    ChatSessionViewSet,
)

router = DefaultRouter()
router.register(r"sessions", ChatSessionViewSet, basename="session")

urlpatterns = [
    # API router endpoints
    path("", include(router.urls)),
]
