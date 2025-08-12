from django.urls import path
from . import views

urlpatterns = [
    path('', views.data_insights, name='datainsights'),
    path("chatbot/", views.chatbot_view, name="chatbot"),
    
]


