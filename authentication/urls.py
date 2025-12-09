from django.urls import path 
from . import views


app_name="auth"

urlpatterns = [
    path("login/", views.login_view, name="login"),
    path('logout/', views.logout_view, name='logout'),
    path('signup/', views.signup_view, name='signup'),
    path('verify-otp/', views.verify_userotp, name='verify_otp'),
    path('resend-otp/', views.resend_otp, name='resend_otp'),
]