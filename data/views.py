from django.shortcuts import render, redirect
from .forms import NoiseDatasetForm, BulkAudioUploadForm
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from core.models import Class, SubClass, Community, Category, Region
from django.shortcuts import get_object_or_404
import plotly.graph_objects as go
import numpy as np
import json
from django.db.models import Count, Avg
from django.utils import timezone
from django.contrib.auth.mixins import LoginRequiredMixin
import hashlib
from django.views.generic import DeleteView, ListView
from django.urls import reverse_lazy
from django.contrib import messages
from .models import NoiseDataset
from datetime import timedelta
from django.db.models import Q
from .models import AudioFeature
from .models import BulkAudioUpload
from .tasks import process_bulk_upload
import logging
import os
from .utils import generate_dataset_name, generate_noise_id

logger = logging.getLogger(__name__)


class RenamedFile:
    """Wrapper class to rename an uploaded file without changing its content"""

    def __init__(self, file, new_name):
        self.file = file
        self.name = new_name
        self._name = new_name

    def __getattr__(self, attr):
        return getattr(self.file, attr)


@login_required
def view_dashboard(request):
    # Basic stats
    total_recordings = NoiseDataset.objects.count()
    user_recordings = (
        NoiseDataset.objects.filter(collector=request.user).count()
        if request.user.is_authenticated
        else 0
    )
    categories_count = Category.objects.count()
    regions_count = Region.objects.count()

    # Data for category pie chart
    category_data = (
        NoiseDataset.objects.values("category__name")
        .annotate(count=Count("id"))
        .order_by("-count")
    )
    category_labels = [item["category__name"] for item in category_data]
    category_counts = [item["count"] for item in category_data]

    # Data for region bar chart
    region_data = (
        NoiseDataset.objects.values("region__name")
        .annotate(count=Count("id"))
        .order_by("-count")
    )
    region_labels = [item["region__name"] for item in region_data]
    region_counts = [item["count"] for item in region_data]

    # Data for time line chart (last 12 months)
    time_labels = []
    time_counts = []
    now = timezone.now()
    for i in range(11, -1, -1):
        month = now - timedelta(days=30 * i)
        time_labels.append(month.strftime("%b %Y"))
        start_date = month.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if month.month == 12:
            end_date = month.replace(year=month.year + 1, month=1, day=1)
        else:
            end_date = month.replace(month=month.month + 1, day=1)
        count = NoiseDataset.objects.filter(
            recording_date__gte=start_date, recording_date__lt=end_date
        ).count()
        time_counts.append(count)

    # Data for audio features radar chart (averages)
    audio_features = AudioFeature.objects.aggregate(
        avg_rms=Avg("rms_energy"),
        avg_centroid=Avg("spectral_centroid"),
        avg_bandwidth=Avg("spectral_bandwidth"),
        avg_zcr=Avg("zero_crossing_rate"),
        avg_harmonic=Avg("harmonic_ratio"),
        avg_percussive=Avg("percussive_ratio"),
    )
    audio_features_data = [
        audio_features["avg_rms"] or 0,
        audio_features["avg_centroid"] or 0,
        audio_features["avg_bandwidth"] or 0,
        audio_features["avg_zcr"] or 0,
        audio_features["avg_harmonic"] or 0,
        audio_features["avg_percussive"] or 0,
    ]

    context = {
        "total_recordings": total_recordings,
        "user_recordings": user_recordings,
        "categories_count": categories_count,
        "regions_count": regions_count,
        "category_labels": json.dumps(category_labels),
        "category_data": json.dumps(category_counts),
        "region_labels": json.dumps(region_labels),
        "region_data": json.dumps(region_counts),
        "time_labels": json.dumps(time_labels),
        "time_data": json.dumps(time_counts),
        "audio_features_data": json.dumps(audio_features_data),
    }

    return render(request, "data/dashboard.html", context)


class NoiseDatasetDeleteView(LoginRequiredMixin, DeleteView):
    model = NoiseDataset
    success_url = reverse_lazy("data:datasetlist")

    def delete(self, request, *args, **kwargs):
        response = super().delete(request, *args, **kwargs)
        messages.success(
            request, f'Dataset "{self.object.noise_id}" was deleted successfully.'
        )
        return response


@login_required
def view_datasetlist(request):
    datasets = NoiseDataset.objects.filter().order_by("-updated_at")
    context = {
        "datasets": datasets,
    }
    return render(request, "data/datasetlist.html", context)


class NoiseDatasetListView(ListView):
    model = NoiseDataset
    template_name = "data/datasetlist.html"
    context_object_name = "datasets"
    ordering = ["-created_at"]
    paginate_by = 200  # More practical default

    def get_queryset(self):
        queryset = super().get_queryset()

        # Search functionality
        search = self.request.GET.get("search")
        if search:
            queryset = queryset.filter(
                Q(noise_id__icontains=search) | Q(name__icontains=search)
            )

        # Category filter
        category = self.request.GET.get("category")
        if category:
            queryset = queryset.filter(category__name=category)

        # Class filter
        class_name = self.request.GET.get("class")
        if class_name:
            queryset = queryset.filter(class_name__name=class_name)

        # Subclass filter
        subclass = self.request.GET.get("subclass")
        if subclass:
            queryset = queryset.filter(subclass__name=subclass)

        # Region filter
        region = self.request.GET.get("region")
        if region:
            queryset = queryset.filter(region__name=region)

        # Community filter
        community = self.request.GET.get("community")
        if community:
            queryset = queryset.filter(community__name=community)

        # Date range filter
        date_range = self.request.GET.get("date_range")
        if date_range:
            today = timezone.now().date()
            if date_range == "today":
                queryset = queryset.filter(recording_date__date=today)
            elif date_range == "week":
                start_date = today - timedelta(days=today.weekday())
                queryset = queryset.filter(recording_date__date__gte=start_date)
            elif date_range == "month":
                queryset = queryset.filter(
                    recording_date__month=today.month, recording_date__year=today.year
                )
            elif date_range == "year":
                queryset = queryset.filter(recording_date__year=today.year)

        # Sorting
        sort = self.request.GET.get("sort")
        if sort:
            order = self.request.GET.get("order", "asc")
            if order == "desc":
                sort = f"-{sort}"
            queryset = queryset.order_by(sort)

        return queryset

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        # Add filter options to context for the template
        context["categories"] = Category.objects.all()
        context["classes"] = Class.objects.all()
        context["subclasses"] = SubClass.objects.all()
        context["regions"] = Region.objects.all()
        context["communities"] = Community.objects.all()

        # Add current filters to context
        context["current_filters"] = {
            "search": self.request.GET.get("search"),
            "category": self.request.GET.get("category"),
            "class": self.request.GET.get("class"),
            "subclass": self.request.GET.get("subclass"),
            "region": self.request.GET.get("region"),
            "community": self.request.GET.get("community"),
            "date_range": self.request.GET.get("date_range"),
        }

        return context


@login_required
def noise_dataset_create(request):
    if request.method == "POST":
        form = NoiseDatasetForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                noise_dataset = form.save(commit=False)
                noise_dataset.collector = request.user

                # Generate the new noise ID format
                noise_dataset.noise_id = generate_noise_id(request.user)

                # Generate dataset name
                noise_dataset.name = generate_dataset_name(noise_dataset)

                # Handle audio file
                if "audio" in request.FILES:
                    audio_file = request.FILES["audio"]

                    # Generate hash of the new audio file for duplicate checking
                    hash_md5 = hashlib.md5()
                    for chunk in audio_file.chunks():
                        hash_md5.update(chunk)

                    # Check against existing records
                    # duplicates = NoiseDataset.objects.filter(collector=request.user)
                    # for dataset in duplicates:
                    # if dataset.audio:
                    # Reset file pointer after reading chunks for hash
                    #  audio_file.seek(0)

                    # Compare hashes
                    # if dataset.get_audio_hash() == new_audio_hash:
                    #  messages.error(request, "This audio file has already been uploaded!")
                    #  return redirect('data:noise_dataset_create')

                    # Reset file pointer again before processing
                    audio_file.seek(0)

                    # Get file extension
                    file_extension = audio_file.name.split(".")[-1].lower()

                    # Generate new filename using noise_id
                    new_filename = f"{noise_dataset.noise_id}.{file_extension}"

                    # Create a renamed file object
                    renamed_file = RenamedFile(audio_file, new_filename)

                    # Replace the file in the form data
                    request.FILES["audio"] = renamed_file

                    # Calculate duration

                noise_dataset.save()
                form.save_m2m()

                messages.success(
                    request,
                    f'Noise dataset "{noise_dataset.name}" created successfully!',
                )
                return redirect("data:noise_dataset_create")

            except Exception as e:
                messages.error(request, f"Error creating dataset: {str(e)}")
                print(f"Error creating noise dataset: {e}")
    else:
        form = NoiseDatasetForm()

    context = {"form": form}
    return render(request, "data/AddNewData.html", context)


@login_required
def bulk_upload_view(request):
    if request.method == "POST":
        form = BulkAudioUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                # Prepare metadata
                metadata = {
                    "description": form.cleaned_data["description"],
                    "region_id": form.cleaned_data["region"].id,
                    "category_id": form.cleaned_data["category"].id,
                    "time_of_day_id": form.cleaned_data["time_of_day"].id,
                    "community_id": form.cleaned_data["community"].id,
                    "class_name_id": form.cleaned_data["class_name"].id,
                    "subclass_id": (
                        form.cleaned_data["subclass"].id
                        if form.cleaned_data["subclass"]
                        else None
                    ),
                    "microphone_type_id": (
                        form.cleaned_data["microphone_type"].id
                        if form.cleaned_data["microphone_type"]
                        else None
                    ),
                    "recording_date": form.cleaned_data["recording_date"].isoformat(),
                    "recording_device": form.cleaned_data["recording_device"],
                }

                # Create bulk upload record
                bulk_upload = BulkAudioUpload.objects.create(
                    user=request.user,
                    metadata=metadata,
                    total_files=len(request.FILES.getlist("audio_files")),
                )

                # Create shared uploads directory if it doesn't exist
                shared_dir = "/shared_uploads"
                if not os.path.exists(shared_dir):
                    os.makedirs(shared_dir, exist_ok=True)

                # Save files to shared location
                file_paths = []
                for file in request.FILES.getlist("audio_files"):
                    try:
                        # Sanitize filename
                        file_name = file.name.replace(" ", "_")
                        temp_path = (
                            f"{shared_dir}/bulk_upload_{bulk_upload.id}_{file_name}"
                        )

                        with open(temp_path, "wb+") as destination:
                            for chunk in file.chunks():
                                destination.write(chunk)

                        if not os.path.exists(temp_path):
                            raise IOError(f"Failed to save temporary file: {temp_path}")

                        file_paths.append(temp_path)
                    except Exception as e:
                        logger.error(f"Failed to save file {file.name}: {str(e)}")
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        raise

                # Start async processing with user ID
                process_bulk_upload.delay(
                    bulk_upload.id,
                    file_paths,
                    request.user.id,  # Pass user ID instead of user object
                )

                return JsonResponse(
                    {
                        "status": "success",
                        "bulk_upload_id": bulk_upload.id,
                        "message": "Bulk upload started successfully",
                    }
                )

            except Exception as e:
                logger.error(
                    f"Bulk upload initialization failed: {str(e)}", exc_info=True
                )
                return JsonResponse({"status": "error", "error": str(e)}, status=500)
        else:
            return JsonResponse({"status": "error", "errors": form.errors}, status=400)

    else:
        form = BulkAudioUploadForm()

    return render(request, "data/bulk_upload.html", {"form": form})


@login_required
def bulk_upload_progress(request, bulk_upload_id):
    try:
        bulk_upload = BulkAudioUpload.objects.get(id=bulk_upload_id, user=request.user)
        return JsonResponse(
            {
                "status": bulk_upload.status,
                "processed": bulk_upload.processed_files,
                "total": bulk_upload.total_files,
                "failed": bulk_upload.failed_files,
            }
        )
    except BulkAudioUpload.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)


@login_required
def noise_dataset_edit(request, pk):

    noise_dataset = NoiseDataset.objects.get(pk=pk, collector=request.user)

    if request.method == "POST":
        form = NoiseDatasetForm(request.POST, request.FILES, instance=noise_dataset)
        if form.is_valid():
            try:
                updated_dataset = form.save(commit=False)

                # Handle audio file update
                if "audio" in request.FILES:
                    audio_file = request.FILES["audio"]

                    # Generate hash of the new audio file for duplicate checking
                    hash_md5 = hashlib.md5()
                    for chunk in audio_file.chunks():
                        hash_md5.update(chunk)
                    new_audio_hash = hash_md5.hexdigest()

                    # Check against existing records (excluding current record)
                    duplicates = NoiseDataset.objects.filter(
                        collector=request.user
                    ).exclude(pk=pk)
                    for dataset in duplicates:
                        if dataset.audio:
                            # Reset file pointer after reading chunks for hash
                            audio_file.seek(0)

                            # Compare hashes
                            if dataset.get_audio_hash() == new_audio_hash:
                                messages.error(
                                    request,
                                    "This audio file has already been uploaded!",
                                )
                                return redirect("data:noise_dataset_edit", pk=pk)

                    # Reset file pointer again before processing
                    audio_file.seek(0)

                    # Get file extension
                    file_extension = audio_file.name.split(".")[-1].lower()

                    # Generate new filename using existing noise_id
                    new_filename = f"{noise_dataset.noise_id}.{file_extension}"

                    # Create a renamed file object
                    renamed_file = RenamedFile(audio_file, new_filename)

                    # Replace the file in the form data
                    request.FILES["audio"] = renamed_file

                    # Calculate duration

                # Update dataset name based on new values
                updated_dataset.name = generate_dataset_name(updated_dataset)
                updated_dataset.save()
                form.save_m2m()

                messages.success(
                    request,
                    f'Noise dataset "{updated_dataset.name}" updated successfully!',
                )
                return redirect("data:noise_dataset_detail", dataset_id=pk)

            except Exception as e:
                messages.error(request, f"Error updating dataset: {str(e)}")
                print(f"Error updating noise dataset: {e}")
    else:
        form = NoiseDatasetForm(instance=noise_dataset)

    context = {
        "form": form,
        "noise_dataset": noise_dataset,
    }
    return render(request, "data/AddNewData.html", context)


def load_classes(request):
    category_id = request.GET.get("category_id")
    classes = Class.objects.filter(category_id=category_id).order_by("name")
    return JsonResponse(list(classes.values("id", "name")), safe=False)


def load_subclasses(request):
    class_id = request.GET.get("class_id")
    subclasses = SubClass.objects.filter(parent_class_id=class_id).order_by("name")
    return JsonResponse(list(subclasses.values("id", "name")), safe=False)


def load_communities(request):
    region_id = request.GET.get("region_id")
    communities = Community.objects.filter(region_id=region_id).order_by("name")
    data = [{"id": c.id, "name": c.name} for c in communities]
    return JsonResponse(data, safe=False)


def show_pages(request):
    return render(request, "data/datasetlist.html")


@login_required
def noise_detail(request, dataset_id):
    dataset = get_object_or_404(NoiseDataset, pk=dataset_id)

    # Get related data
    audio_features = getattr(dataset, "audio_features", None)
    noise_analysis = getattr(dataset, "noise_analysis", None)

    # Prepare visualization data
    context = {
        "noise_dataset": dataset,
        "audio_features": audio_features,
        "noise_analysis": noise_analysis,
    }

    # Add visualizations if features exist
    if audio_features:
        # Waveform plot

        if audio_features.waveform_data:
            waveform_fig = create_waveform_plot(audio_features.waveform_data)
            context["waveform_plot"] = waveform_fig.to_html(full_html=False)

        # Spectrogram plot
        if audio_features.mel_spectrogram:
            spectrogram_fig = create_spectrogram_plot(audio_features.mel_spectrogram)
            context["spectrogram_plot"] = spectrogram_fig.to_html(full_html=False)

        # MFCC plot
        if audio_features.mfccs:

            mfcc_fig = create_mfcc_plot(audio_features.mfccs)
            context["mfcc_plot"] = mfcc_fig.to_html(full_html=False)

        # Frequency features plot
        freq_fig = create_frequency_features_plot(audio_features)
        context["freq_plot"] = freq_fig.to_html(full_html=False)

    return render(request, "data/Noise_detail.html", context)


def create_waveform_plot(waveform_data):
    fig = go.Figure()

    if isinstance(waveform_data, dict):
        x = waveform_data.get("time", [])
        y = waveform_data.get("amplitude", [])

    elif isinstance(waveform_data, list):
        x = list(range(len(waveform_data)))
        y = waveform_data
    else:
        x = []
        y = []

    fig.add_trace(
        go.Scatter(x=x, y=y, mode="lines", name="Waveform", line=dict(color="#1f77b4"))
    )
    fig.update_layout(
        title="Audio Waveform",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        height=300,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


def create_spectrogram_plot(spectrogram_data):
    fig = go.Figure()

    if isinstance(spectrogram_data, dict):

        z = spectrogram_data.get("values", [])
        x = spectrogram_data.get("time", [])
        y = spectrogram_data.get("freq", [])
    elif isinstance(spectrogram_data, list):

        z = spectrogram_data
        x = list(range(len(spectrogram_data[0]))) if spectrogram_data else []
        y = list(range(len(spectrogram_data))) if spectrogram_data else []
    else:
        z, x, y = [], [], []

    if z:  # Only plot if we have data
        fig.add_trace(go.Heatmap(z=z, x=x, y=y, colorscale="Jet"))

    fig.update_layout(
        title="Mel Spectrogram",
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        height=300,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


def create_mfcc_plot(mfccs):
    # Debug: Print the input to see what we're working with
    print(f"Raw MFCCs data type: {type(mfccs)}")
    if hasattr(mfccs, "shape"):
        print(f"MFCCs shape: {mfccs.shape}")
    elif isinstance(mfccs, list):
        print(f"MFCCs list length: {len(mfccs)}")

    # Convert to numpy array if not already
    try:
        if not isinstance(mfccs, np.ndarray):
            mfccs = np.array(mfccs)
    except Exception as e:
        print(f"Error converting MFCCs to numpy array: {e}")
        mfccs = np.array([])

    # Validate the array
    if mfccs.size == 0:
        print("Warning: Empty MFCC data received")
        # Return empty figure with message
        fig = go.Figure()
        fig.update_layout(
            title="MFCC Coefficients (No Data)",
            height=300,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        return fig

    # Ensure 2D array
    if len(mfccs.shape) == 1:
        mfccs = np.expand_dims(mfccs, axis=0)

    print(f"Final MFCCs shape: {mfccs.shape}")

    # Create the plot
    fig = go.Figure(
        data=go.Heatmap(z=mfccs, colorscale="Viridis", colorbar=dict(title="Magnitude"))
    )

    fig.update_layout(
        title="MFCC Coefficients",
        xaxis_title="Coefficient Index",
        yaxis_title="Frame",
        height=300,
        margin=dict(l=40, r=40, t=40, b=40),
    )

    return fig


def create_frequency_features_plot(audio_features):
    fig = go.Figure()

    features = [
        ("spectral_centroid", "Spectral Centroid", "red"),
        ("spectral_bandwidth", "Spectral Bandwidth", "blue"),
        ("spectral_rolloff", "Spectral Rolloff", "green"),
    ]

    for i, (attr, name, color) in enumerate(features, start=1):
        value = getattr(audio_features, attr, None)
        if value is not None:
            fig.add_trace(
                go.Scatter(
                    x=[i],
                    y=[value],
                    mode="markers+text",
                    name=name,
                    text=[name],
                    textposition="top center",
                    marker=dict(size=12, color=color),
                )
            )

    fig.update_layout(
        title="Spectral Features Comparison",
        xaxis=dict(
            title="Feature Type",
            tickvals=[1, 2, 3],
            ticktext=["Centroid", "Bandwidth", "Rolloff"],
            range=[0.5, 3.5],  # Add some padding
        ),
        yaxis_title="Frequency (Hz)",
        height=300,
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=False,  # Since we're showing labels on markers
    )
    return fig

