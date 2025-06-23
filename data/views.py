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
from django.http import JsonResponse

from core.models import Class,SubClass,Community
from django.db.models import Count
from django.shortcuts import get_object_or_404
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
import numpy as np
import json
from .models import NoiseDataset, AudioFeature, NoiseAnalysis



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
    # Get total recordings count
    total_recordings = NoiseDataset.objects.count()
    
    # Get recordings count by current user
    user_recordings = NoiseDataset.objects.filter(collector=request.user).count()
    
   
    context = {
        'total_recordings': total_recordings,
        'user_recordings': user_recordings,
       
    }
    return render(request, 'data/dashboard.html', context)



@login_required
def view_datasetlist(request):   
    datasets = NoiseDataset.objects.filter().order_by('-updated_at')
    context = {
        'datasets': datasets,
    }
    return render(request,  'data/datasetlist.html',context)





import uuid
from datetime import datetime
import hashlib

@login_required
def noise_dataset_create(request):
    if request.method == 'POST':
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
                if 'audio' in request.FILES:
                    audio_file = request.FILES['audio']
                    
                    # Generate hash of the new audio file for duplicate checking
                    hash_md5 = hashlib.md5()
                    for chunk in audio_file.chunks():
                        hash_md5.update(chunk)
                    new_audio_hash = hash_md5.hexdigest()
                    
                    # Check against existing records
                    duplicates = NoiseDataset.objects.filter(collector=request.user)
                    for dataset in duplicates:
                        if dataset.audio:
                            # Reset file pointer after reading chunks for hash
                            audio_file.seek(0)
                            
                            # Compare hashes
                            if dataset.get_audio_hash() == new_audio_hash:
                                messages.error(request, "This audio file has already been uploaded!")
                                return redirect('data:noise_dataset_create')
                    
                    # Reset file pointer again before processing
                    audio_file.seek(0)
                    
                    # Get file extension
                    file_extension = audio_file.name.split('.')[-1].lower()
                    
                    # Generate new filename using noise_id
                    new_filename = f"{noise_dataset.noise_id}.{file_extension}"
                    
                    # Create a renamed file object
                    renamed_file = RenamedFile(audio_file, new_filename)
                    
                    # Replace the file in the form data
                    request.FILES['audio'] = renamed_file
                    
                    # Calculate duration
                    try:
                        duration = get_audio_duration(renamed_file)
                        if duration:
                            noise_dataset.duration = duration
                    except Exception as e:
                        print(f"Error getting audio duration: {e}")
                        noise_dataset.duration = None
                
                noise_dataset.save()
                form.save_m2m()
                
                messages.success(request, f'Noise dataset "{noise_dataset.name}" created successfully!')
                return redirect('data:noise_dataset_create')
                
            except Exception as e:
                messages.error(request, f'Error creating dataset: {str(e)}')
                print(f"Error creating noise dataset: {e}")
    else:
        form = NoiseDatasetForm()
    
    context = {'form': form}
    return render(request, 'data/AddNewData.html', context)
def generate_noise_id(user):
    """Generate a unique noise ID with format: NSE-{speaker_id}-{timestamp}-{3 random chars}"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_chars = uuid.uuid4().hex[:5].upper()
    
    # Get speaker_id from user, use 'UNK' if not available
    speaker_id = user.speaker_id if user.speaker_id else 'UNK'
    
    return f"NSE-{speaker_id}-{timestamp}-{random_chars}"

def generate_dataset_name(noise_dataset):
    """Generate a descriptive name for the dataset based on the new fields"""
    parts = []

    if noise_dataset.category:
        parts.append(str(noise_dataset.category))
    
    # Add class name if available
    if noise_dataset.class_name:
        parts.append(str(noise_dataset.class_name))

    if noise_dataset.subclass:
        parts.append(str(noise_dataset.subclass))
    
 

    
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


def generate_noise_id(user):
    """Generate a unique noise ID with format: NSE-{speaker_id}-{timestamp}-{3 random chars}"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_chars = uuid.uuid4().hex[:3].upper()
    
    # Get speaker_id from user, use 'UNK' if not available
    speaker_id = user.speaker_id if user.speaker_id else 'UNK'
    
    return f"NSE-{speaker_id}-{timestamp}-{random_chars}"

def generate_dataset_name(noise_dataset):
    """Generate a descriptive name for the dataset"""
    parts = []
    
    if noise_dataset.region:
        parts.append(str(noise_dataset.region))
    
    if noise_dataset.community:
        parts.append(str(noise_dataset.community))
    
    if noise_dataset.category:
        parts.append(str(noise_dataset.category))

    
    # Add timestamp
    timestamp = timezone.now().strftime("%Y%m%d_%H%M")
    parts.append(timestamp)
    
    return "_".join(parts) if parts else f"NoiseDataset_{timestamp}"

def get_audio_duration(audio_file):
    """Get duration of audio file in seconds"""
    try:
        # For newly uploaded files (InMemoryUploadedFile or TemporaryUploadedFile)
        if hasattr(audio_file, 'temporary_file_path'):
            # Temporary file on disk
            file_path = audio_file.temporary_file_path()
        elif hasattr(audio_file, 'read'):
            # In-memory file, we need to save to a temp file
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                for chunk in audio_file.chunks():
                    tmp.write(chunk)
                file_path = tmp.name
            # Reset file pointer for Django to save the file properly
            audio_file.seek(0)
        else:
            # For already saved files (FileField)
            file_path = audio_file.path
        
        # Use mutagen to get duration
        audio_info = MutagenFile(file_path)
        if audio_info is not None and hasattr(audio_info, 'info'):
            duration = round(audio_info.info.length, 2)
            
            # Clean up if we created a temp file
            if 'tmp' in locals():
                try:
                    os.unlink(file_path)
                except:
                    pass
            return duration
        
        return None
    except Exception as e:
        print(f"Error getting audio duration: {e}")
        return None


# Optional: List view for managing datasets
@login_required
def noise_dataset_list(request):
    datasets = NoiseDataset.objects.filter().order_by('-updated_at')
    context = {
        'datasets': datasets,
    }
    return render(request, 'datacollection/noise_dataset_list.html', context)










def load_classes(request):
    category_id = request.GET.get('category_id')
    classes = Class.objects.filter(category_id=category_id).order_by('name')
    return JsonResponse(list(classes.values('id', 'name')), safe=False)

# AJAX view for loading subclasses
def load_subclasses(request):
    class_id = request.GET.get('class_id')
    subclasses = SubClass.objects.filter(parent_class_id=class_id).order_by('name')
    return JsonResponse(list(subclasses.values('id', 'name')), safe=False)

def load_communities(request):
    region_id = request.GET.get('region_id')
    communities = Community.objects.filter(region_id=region_id).order_by('name')
    data = [{'id': c.id, 'name': c.name} for c in communities]
    return JsonResponse(data, safe=False)


def show_pages(request):
    return render(request, 'data/datasetlist.html')



def noise_detail(request,dataset_id):
    dataset = get_object_or_404(NoiseDataset, pk=dataset_id)
    context = {
        'noise_dataset': dataset,
    }
    
    return render(request, 'data/Noise_detail.html', context)







def noise_dataset_detail(request, pk):
    dataset = get_object_or_404(NoiseDataset, pk=pk)
    audio_features = get_object_or_404(AudioFeature, noise_dataset=dataset)
    noise_analysis = get_object_or_404(NoiseAnalysis, noise_dataset=dataset)
    
    # Get waveform data
    waveform = np.array(json.loads(audio_features.waveform_data))
    time_axis = np.arange(len(waveform)) / 44100  # Assuming 44100Hz sample rate
    
    # Create waveform plot
    waveform_fig = go.Figure()
    waveform_fig.add_trace(go.Scatter(
        x=time_axis,
        y=waveform,
        name='Waveform'
    ))
    waveform_fig.update_layout(
        title='Audio Waveform',
        xaxis_title='Time (s)',
        yaxis_title='Amplitude'
    )
    waveform_plot = plot(waveform_fig, output_type='div', include_plotlyjs=False)
    
    # Create spectrogram plot
    mel_spectrogram = np.array(json.loads(audio_features.mel_spectrogram))
    spectrogram_fig = go.Figure()
    spectrogram_fig.add_trace(go.Heatmap(
        z=mel_spectrogram,
        colorscale='Jet',
        colorbar=dict(title='dB')
    ))
    spectrogram_fig.update_layout(
        title='Mel Spectrogram',
        xaxis_title='Time',
        yaxis_title='Frequency Bin'
    )
    spectrogram_plot = plot(spectrogram_fig, output_type='div', include_plotlyjs=False)
    
    # Create MFCC plot
    mfcc_fig = go.Figure()
    for i, coeff in enumerate(audio_features.mfccs):
        mfcc_fig.add_trace(go.Scatter(
            y=[coeff],
            name=f'MFCC {i+1}',
            mode='markers'
        ))
    mfcc_fig.update_layout(
        title='MFCC Coefficients',
        xaxis_title='Coefficient Index',
        yaxis_title='Value'
    )
    mfcc_plot = plot(mfcc_fig, output_type='div', include_plotlyjs=False)
    
    # Create frequency features plot
    freq_features = {
        'Spectral Centroid': audio_features.spectral_centroid,
        'Spectral Bandwidth': audio_features.spectral_bandwidth,
        'Spectral Rolloff': audio_features.spectral_rolloff
    }
    freq_fig = go.Figure()
    freq_fig.add_trace(go.Bar(
        x=list(freq_features.keys()),
        y=list(freq_features.values())
    ))
    freq_fig.update_layout(
        title='Frequency Domain Features',
        yaxis_title='Hz'
    )
    freq_plot = plot(freq_fig, output_type='div', include_plotlyjs=False)
    
    context = {
        'dataset': dataset,
        'audio_features': audio_features,
        'noise_analysis': noise_analysis,
        'waveform_plot': waveform_plot,
        'spectrogram_plot': spectrogram_plot,
        'mfcc_plot': mfcc_plot,
        'freq_plot': freq_plot,
    }
    
    return render(request, 'noise_analysis/detail.html', context)





