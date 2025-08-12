from django.urls import path
from django.shortcuts import render

app_name = "approval"


def review(request):
    # UI-only stub view to render the approval template
    return render(request, "approval/approval_review.html")


urlpatterns = [
    path("", review, name="review"),
]

