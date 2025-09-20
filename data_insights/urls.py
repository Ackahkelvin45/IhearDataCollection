from . import views
from django.urls import path, include
from rest_framework.routers import DefaultRouter


app_name = "insights"

router = DefaultRouter()
router.register(r"sessions", views.ChatSessionView, basename="session")

urlpatterns = [
    path("", views.unified_chat, name="home"),
    path("chat/", views.chat, name="chat"),
    path("unified/", views.unified_chat, name="unified"),
    path("", include(router.urls)),
]
