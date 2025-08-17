from django.urls import path
from . import views

app_name = "insights"

urlpatterns = [
    path("", views.home, name="home"),
    path("session/", views.session, name="session"),
]
