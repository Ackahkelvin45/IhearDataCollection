from django.shortcuts import render
from django.views.generic import TemplateView

# Create your views here.
class HomepageView(TemplateView):
    template_name = "onboarding/homepage.html"

class HowYourRecordingHelpsView(TemplateView):
    template_name = "onboarding/how_your_recording_helps.html"

class TermsAndPrivacyView(TemplateView):
    template_name = "onboarding/terms_and_privacy.html"