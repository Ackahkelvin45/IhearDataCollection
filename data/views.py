from django.shortcuts import render
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.utils import timezone
from .forms import NoiseDatasetForm
from .models import NoiseDataset
import uuid
from mutagen import File as MutagenFile
import os
# Create your views here.
from django.contrib.auth.decorators import login_required


@login_required
def view_dashboard(request):
    return render(request, 'data/dashboard.html')


@login_required
def view_datasetlist(request):
    return render(request, 'data/datasetlist.html')





@login_required
def noise_dataset_create(request):
    if request.method == 'POST':
        form = NoiseDatasetForm(request.POST, request.FILES)
        if form.is_valid():
            # Create instance but don't save yet
            noise_dataset = form.save(commit=False)
            
            # Set the collector to current user
            noise_dataset.collector = request.user
            
            # Generate name and noise_id
            noise_dataset.noise_id = generate_noise_id()
            noise_dataset.name = generate_dataset_name(noise_dataset)
            
            # Calculate audio duration if audio file is provided
            if noise_dataset.audio:
                try:
                    duration = get_audio_duration(noise_dataset.audio)
                    noise_dataset.duration = str(duration) if duration else None
                except Exception as e:
                    print(f"Error getting audio duration: {e}")
                    noise_dataset.duration = None
            
            # Save the instance
            noise_dataset.save()
            
            messages.success(request, f'Noise dataset "{noise_dataset.name}" created successfully!')
            return redirect('noise_dataset_create')  # Redirect to same page for new entry
            
    else:
        form = NoiseDatasetForm()
    
    # Get recent datasets for display (optional)
    recent_datasets = NoiseDataset.objects.filter(collector=request.user).order_by('-updated_at')[:5]
    
    context = {
        'form': form,
        'recent_datasets': recent_datasets,
    }
    return render(request, 'data/AddNewData.html', context)


def generate_noise_id():
    """Generate a unique noise ID"""
    return f"NSE-{uuid.uuid4().hex[:8].upper()}"


def generate_dataset_name(noise_dataset):
    """Generate a descriptive name for the dataset"""
    parts = []
    
    if noise_dataset.region:
        parts.append(str(noise_dataset.region))
    
    if noise_dataset.community:
        parts.append(str(noise_dataset.community))
    
    if noise_dataset.category:
        parts.append(str(noise_dataset.category))
    
    if noise_dataset.environment_type:
        parts.append(str(noise_dataset.environment_type))
    
    # Add timestamp
    timestamp = timezone.now().strftime("%Y%m%d_%H%M")
    parts.append(timestamp)
    
    return "_".join(parts) if parts else f"NoiseDataset_{timestamp}"


def get_audio_duration(audio_file):
    """Get duration of audio file in seconds"""
    try:
        # Save temporary file if it's an uploaded file
        if hasattr(audio_file, 'temporary_file_path'):
            file_path = audio_file.temporary_file_path()
        else:
            # For already saved files
            file_path = audio_file.path
        
        # Use mutagen to get duration
        audio_info = MutagenFile(file_path)
        if audio_info is not None and hasattr(audio_info, 'info'):
            return round(audio_info.info.length, 2)
        
        return None
    except Exception as e:
        print(f"Error getting audio duration: {e}")
        return None


# Optional: List view for managing datasets
@login_required
def noise_dataset_list(request):
    datasets = NoiseDataset.objects.filter(collector=request.user).order_by('-updated_at')
    context = {
        'datasets': datasets,
    }
    return render(request, 'datacollection/noise_dataset_list.html', context)