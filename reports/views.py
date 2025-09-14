from django.shortcuts import render
from django.views.generic import TemplateView
from data.models import Dataset
from core.models import (
    Region,
    Microphone_Type,
    Time_Of_Day,
    Category,
    Class,
    SubClass,
    Community,
)
from django.views import View
from django.http import JsonResponse

# Create your views here.


class ReportView(TemplateView):
    template_name = "reports/report.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        context["datasets"] = Dataset.objects.filter().order_by("-created_at")
        context["regions"] = Region.objects.all()
        context["categories"] = Category.objects.all()
        context["microphones"] = Microphone_Type.objects.all()

        return context


class DependentDropdownQuery(View):
    def get(self, request, *args, **kwargs):
        category_id = request.GET.get("category_id")
        class_id = request.GET.get("class_id")
        region_id = request.GET.get("region_id")
        datatype_id=request.GET.get("datatype_id")

        if datatype_id:
            categories = Category.objects.filter(datatype_id=datatype_id).order_by("name")
            return JsonResponse(list(categories.values("id", "name")), safe=False)
        if category_id:
            classes = Class.objects.filter(category_id=category_id).order_by("name")
            return JsonResponse(list(classes.values("id", "name")), safe=False)
        elif class_id:
            subclasses = SubClass.objects.filter(parent_class_id=class_id).order_by(
                "name"
            )
            return JsonResponse(list(subclasses.values("id", "name")), safe=False)
        elif region_id:
            communities = Community.objects.filter(region_id=region_id).order_by("name")
            return JsonResponse(list(communities.values("id", "name")), safe=False)

        else:
            return JsonResponse([], safe=False)
