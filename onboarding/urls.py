from django.urls import path
from .views import HomepageView, HowYourRecordingHelpsView, TermsAndPrivacyView

app_name = "onboarding"

urlpatterns = [
    path("", HomepageView.as_view(), name="homepage"),
    path(
        "how-your-recording-helps/",
        HowYourRecordingHelpsView.as_view(),
        name="how_your_recording_helps",
    ),
    path("terms-and-privacy/", TermsAndPrivacyView.as_view(), name="terms_and_privacy"),
]
