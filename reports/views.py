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
from data.serializers  import NoiseDatasetSerializer
from rest_framework.response import Response
from rest_framework.views import APIView
from data.models import NoiseDataset,AudioFeature
from django.db.models import Avg, Sum, F, FloatField, ExpressionWrapper
from  rest_framework.pagination import PageNumberPagination
from django.db.models.functions import Round

# Create your views here.


class ReportView(TemplateView):
    template_name = "reports/report.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        context["datatypes"] = Dataset.objects.filter().order_by("-created_at")
        context["regions"] = Region.objects.all()
        context["categories"] = Category.objects.all()
        context["microphones"] = Microphone_Type.objects.all()
        context['timeofday']=Time_Of_Day.objects.all()
        return context


class DependentDropdownQuery(View):
    def get(self, request, *args, **kwargs):
   
        category_ids = request.GET.get("category_ids")
        class_ids = request.GET.get("class_ids")
        region_ids = request.GET.get("region_ids")
        datatype_ids = request.GET.get("datatype_ids")

        try:
            if datatype_ids:
                ids_to_process = self._parse_ids(datatype_ids)
                if ids_to_process:
                    categories = Category.objects.filter(
                        data_type_id__in=ids_to_process
                    ).order_by("name").distinct()
                    return JsonResponse(list(categories.values("id", "name")), safe=False)
            

            elif category_ids:
                ids_to_process = self._parse_ids(category_ids )
                if ids_to_process:
                    classes = Class.objects.filter(
                        category_id__in=ids_to_process
                    ).order_by("name").distinct()
                    return JsonResponse(list(classes.values("id", "name")), safe=False)
            

            elif class_ids:
                ids_to_process = self._parse_ids(class_ids )
                if ids_to_process:
                    subclasses = SubClass.objects.filter(
                        parent_class_id__in=ids_to_process
                    ).order_by("name").distinct()
                    return JsonResponse(list(subclasses.values("id", "name")), safe=False)
            
            elif  region_ids:
                ids_to_process = self._parse_ids(region_ids )
                if ids_to_process:
                    communities = Community.objects.filter(
                        region_id__in=ids_to_process
                    ).order_by("name").distinct()
                    return JsonResponse(list(communities.values("id", "name")), safe=False)
            
            return JsonResponse([], safe=False)
            
        except Exception as e:
            print(f"Error in DependentDropdownQuery: {str(e)}")
            return JsonResponse({"error": "Invalid request parameters"}, status=400)

    def _parse_ids(self, ids_string):
        if not ids_string:
            return []
        
        try:
            if ',' in str(ids_string):
                id_list = [int(id_str.strip()) for id_str in str(ids_string).split(',') if id_str.strip().isdigit()]
            else:
                id_list = [int(ids_string)] if str(ids_string).isdigit() else []
            
            return id_list
        except (ValueError, TypeError):
            return []




class ReporFilterView(APIView):
    
    def post (self,request,*args,**kwargs):
        payload=request.data
        queryset=NoiseDataset.objects.all()

        if (payload['categories']):
            queryset=queryset.filter(category__id__in=payload['categories'])
        if (payload['classes']):
            queryset=queryset.filter(class_name__id__in=payload['classes'])
        if (payload['subclasses']):
            queryset=queryset.filter(subclass__id__in=payload['subclasses'])
        if payload["regions"]:
            queryset = queryset.filter(region__id__in=payload["regions"])
        if payload["communities"]:
            queryset = queryset.filter(community__id__in=payload["communities"])
        if payload["microphones"]:
            queryset = queryset.filter(microphone_type__id__in=payload["microphones"])
        if payload["timeOfDay"]:
            queryset = queryset.filter(time_of_day__id__in=payload["timeOfDay"])
        if payload["recordingDevice"]:
            queryset = queryset.filter(recording_device__icontains=payload["recordingDevice"])

        if payload["dateRange"]:
            try:
                start,end=payload["daterange"].split("to")
                queryset=queryset.filter(recorded_date__range=[start,end])
            except Exception:
                pass
        durations = AudioFeature.objects.filter(noise_dataset__in=queryset).aggregate(
        avg_duration=Round(Avg("duration"), 2),  
        total_duration=Round(
        ExpressionWrapper(Sum("duration") / 3600.0, output_field=FloatField()), 2
        ),  
        )

        

        paginator=PageNumberPagination()
        page = paginator.paginate_queryset(queryset, request)
        serializer =NoiseDatasetSerializer(page,many=True)

        response_data={
            "results": serializer.data,
            "total_count": queryset.count(),
            "average_duration": durations["avg_duration"] or 0,
            "total_duration": durations["total_duration"] or 0,
            "total_pages":  paginator.page.paginator.num_pages,
        }
        return paginator.get_paginated_response(response_data)

       


