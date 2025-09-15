from django.urls import path
from .views import ReportView, DependentDropdownQuery,ReporFilterView

app_name = "reports"

urlpatterns = [
    path("", ReportView.as_view(), name="report"),
    path(
        "dependent-dropdown/",
        DependentDropdownQuery.as_view(),
        name="dependent_dropdown",
    ),
    path ("report-writer/",ReporFilterView.as_view(),name="report_writer"),
]
