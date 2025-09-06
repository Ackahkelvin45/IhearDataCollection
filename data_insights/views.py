from django.shortcuts import render
from django.db.models import Count, Min, Max
from data.models import NoiseDataset, NoiseAnalysis
from core.models import Region, Community


def home(request):
    """Landing page for the insights chat UI."""
    # Order suggestions so the first one maps to a chart
    suggestions = [
        "Which region has the most data collected?",
        "Show me recent 20 data collected",
        "Which data has the highest decibel level?",
        "Which community has the lowest decibel level?",
    ]
    return render(request, "data_insights/home.html", {"suggestions": suggestions})


def session(request):
    """Chat session screen that displays a mock response and visualization.

    This is UI-only. It shows how a query and response would look while
    we integrate an LLM in a future step.
    """
    query = request.GET.get("q", "Which region has the most data collected?")
    # Example recent items for the left panel
    recents = [
        "Which region has the most data collected?",
        "Show me recent 20 data collected",
        "Which data has the highest decibel level?",
        "Which community has the lowest decibel level?",
    ]

    # Decide which visualization to show based on the query
    normalized = (query or "").lower()

    show_chart = False
    show_table = False
    chart = None
    table_headers = []
    table_rows = []
    subtitle = None
    assistant_text = None

    try:
        if "region" in normalized:
            # Chart: distribution of datasets by region
            show_chart = True
            region_counts = (
                NoiseDataset.objects.values("region__name")
                .annotate(count=Count("id"))
                .order_by("-count")
            )
            labels = [r.get("region__name") or "Unknown" for r in region_counts]
            values = [r["count"] for r in region_counts]
            chart = {
                "type": "doughnut",
                "labels": labels,
                "values": values,
                "title": "Recordings by Region",
            }
            subtitle = "The chart below shows dataset distribution per region"
            assistant_text = (
                "Here are the regions ranked by number of recordings collected."
            )

        elif "recent" in normalized or "20" in normalized:
            # Table: 20 most recent datasets
            show_table = True
            table_headers = [
                "Name",
                "Region",
                "Category",
                "Recording Date",
            ]
            recent = NoiseDataset.objects.select_related("region", "category").order_by(
                "-created_at"
            )[:20]
            table_rows = [
                [
                    d.name or d.noise_id,
                    getattr(d.region, "name", "-") if d.region else "-",
                    getattr(d.category, "name", "-") if d.category else "-",
                    d.recording_date.strftime("%d/%m/%Y") if d.recording_date else "-",
                ]
                for d in recent
            ]
            subtitle = "Recent 20 datasets"
            assistant_text = (
                "Here are the 20 most recent recordings added to the dataset."
            )

        elif "highest" in normalized and (
            "db" in normalized or "decibel" in normalized
        ):
            # Table: top datasets by maximum decibel level
            show_table = True
            table_headers = ["Name", "Region", "Max dB", "Recording Date"]
            top = (
                NoiseAnalysis.objects.select_related("noise_dataset__region")
                .exclude(max_db__isnull=True)
                .order_by("-max_db")[:20]
            )
            for a in top:
                d = a.noise_dataset
                table_rows.append(
                    [
                        d.name or d.noise_id,
                        getattr(d.region, "name", "-") if d.region else "-",
                        round(a.max_db, 2) if a.max_db is not None else "-",
                        (
                            d.recording_date.strftime("%d/%m/%Y")
                            if d.recording_date
                            else "-"
                        ),
                    ]
                )
            subtitle = "Top datasets by maximum decibel level"
            assistant_text = (
                "These are the recordings with the highest measured decibel levels."
            )

        elif "community" in normalized and (
            "lowest" in normalized and ("db" in normalized or "decibel" in normalized)
        ):
            # Chart: lowest average decibel by community (bar)
            show_chart = True
            qs = (
                NoiseAnalysis.objects.select_related("noise_dataset__community")
                .exclude(mean_db__isnull=True)
                .values("noise_dataset__community__name")
                .annotate(min_avg=Min("mean_db"))
                .order_by("min_avg")[:10]
            )
            labels = [r.get("noise_dataset__community__name") or "Unknown" for r in qs]
            values = [r["min_avg"] for r in qs]
            chart = {
                "type": "bar",
                "labels": labels,
                "values": values,
                "title": "Communities with Lowest Average dB",
            }
            subtitle = "Communities with the lowest average decibel levels"
            assistant_text = (
                "Lowest average decibel levels by community are shown below."
            )

        else:
            # Default to chart by region
            show_chart = True
            region_counts = (
                NoiseDataset.objects.values("region__name")
                .annotate(count=Count("id"))
                .order_by("-count")
            )
            labels = [r.get("region__name") or "Unknown" for r in region_counts]
            values = [r["count"] for r in region_counts]
            chart = {
                "type": "doughnut",
                "labels": labels,
                "values": values,
                "title": "Recordings by Region",
            }
            subtitle = "The chart below shows dataset distribution per region"
            assistant_text = (
                "Here are the regions ranked by number of recordings collected."
            )

    except Exception:
        # Fail silently to avoid UI errors
        pass

    context = {
        "query": query,
        "recents": recents,
        "page_title": "Customer Insights",
        "show_chart": show_chart,
        "show_table": show_table,
        "chart": chart,
        "table_headers": table_headers,
        "table_rows": table_rows,
        "subtitle": subtitle,
        "assistant_text": assistant_text,
    }
    return render(request, "data_insights/session.html", context)


