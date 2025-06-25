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
from django.db.models import Count, Avg
from django.utils import timezone
from datetime import timedelta
from core.models import Category, Region, Microphone_Type, Class, SubClass

import uuid
from datetime import datetime
import hashlib
from django.views.generic import DeleteView
from django.urls import reverse_lazy
from django.contrib import messages


from django.shortcuts import render
from .models import NoiseDataset, Category, Region, AudioFeature
from django.db.models import Count
from datetime import datetime, timedelta
from django.utils import timezone
import random


class RenamedFile:
    """Wrapper class to rename an uploaded file without changing its content"""
    
    def __init__(self, file, new_name):
        self.file = file
        self.name = new_name
        self._name = new_name
        
    def __getattr__(self, attr):
        return getattr(self.file, attr)

def view_dashboard(request):
    # Basic stats
    total_recordings = NoiseDataset.objects.count()
    user_recordings = NoiseDataset.objects.filter(collector=request.user).count() if request.user.is_authenticated else 0
    categories_count = Category.objects.count()
    regions_count = Region.objects.count()
    
    # Data for category pie chart
    category_data = NoiseDataset.objects.values('category__name').annotate(count=Count('id')).order_by('-count')
    category_labels = [item['category__name'] for item in category_data]
    category_counts = [item['count'] for item in category_data]
    
    # Data for region bar chart
    region_data = NoiseDataset.objects.values('region__name').annotate(count=Count('id')).order_by('-count')
    region_labels = [item['region__name'] for item in region_data]
    region_counts = [item['count'] for item in region_data]
    
    # Data for time line chart (last 12 months)
    time_labels = []
    time_counts = []
    now = timezone.now()
    for i in range(11, -1, -1):
        month = now - timedelta(days=30*i)
        time_labels.append(month.strftime('%b %Y'))
        start_date = month.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if month.month == 12:
            end_date = month.replace(year=month.year+1, month=1, day=1)
        else:
            end_date = month.replace(month=month.month+1, day=1)
        count = NoiseDataset.objects.filter(recording_date__gte=start_date, recording_date__lt=end_date).count()
        time_counts.append(count)
    
    # Data for audio features radar chart (averages)
    audio_features = AudioFeature.objects.aggregate(
        avg_rms=Avg('rms_energy'),
        avg_centroid=Avg('spectral_centroid'),
        avg_bandwidth=Avg('spectral_bandwidth'),
        avg_zcr=Avg('zero_crossing_rate'),
        avg_harmonic=Avg('harmonic_ratio'),
        avg_percussive=Avg('percussive_ratio')
    )
    audio_features_data = [
        audio_features['avg_rms'] or 0,
        audio_features['avg_centroid'] or 0,
        audio_features['avg_bandwidth'] or 0,
        audio_features['avg_zcr'] or 0,
        audio_features['avg_harmonic'] or 0,
        audio_features['avg_percussive'] or 0
    ]
    
    context = {
        'total_recordings': total_recordings,
        'user_recordings': user_recordings,
        'categories_count': categories_count,
        'regions_count': regions_count,
        'category_labels': json.dumps(category_labels),
        'category_data': json.dumps(category_counts),
        'region_labels': json.dumps(region_labels),
        'region_data': json.dumps(region_counts),
        'time_labels': json.dumps(time_labels),
        'time_data': json.dumps(time_counts),
        'audio_features_data': json.dumps(audio_features_data),
    }
    
    return render(request, 'data/dashboard.html', context)


class NoiseDatasetDeleteView(DeleteView):
    model = NoiseDataset
    success_url = reverse_lazy('data:datasetlist')
    
    def delete(self, request, *args, **kwargs):
        response = super().delete(request, *args, **kwargs)
        messages.success(request, f'Dataset "{self.object.noise_id}" was deleted successfully.')
        return response

@login_required
def view_datasetlist(request):   
    datasets = NoiseDataset.objects.filter().order_by('-updated_at')
    context = {
        'datasets': datasets,
    }
    return render(request,  'data/datasetlist.html',context)





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


def generate_dataset_name(noise_dataset):
    parts = []

    if noise_dataset.category:
        parts.append(str(noise_dataset.category.name))
    
    # Add class name if available
    if noise_dataset.class_name:
        parts.append(str(noise_dataset.class_name.name))

    if noise_dataset.subclass:
        parts.append(str(noise_dataset.subclass.name))
    
 

    
    # Add timestamp
    timestamp = timezone.now().strftime("%Y%m%d_%H%M")
    parts.append(timestamp)
    
    return "_".join(parts) if parts else f"NoiseDataset_{timestamp}"



def generate_noise_id(user):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_chars = uuid.uuid4().hex[:3].upper()
    
    speaker_id = user.speaker_id if user.speaker_id else 'UNK'
    
    return f"NSE-{speaker_id}-{timestamp}-{random_chars}"




@login_required
def noise_dataset_edit(request, pk):
    
    noise_dataset = NoiseDataset.objects.get(pk=pk, collector=request.user)


    if request.method == 'POST':
        form = NoiseDatasetForm(request.POST, request.FILES, instance=noise_dataset)
        if form.is_valid():
            try:
                updated_dataset = form.save(commit=False)
                
                # Handle audio file update
                if 'audio' in request.FILES:
                    audio_file = request.FILES['audio']
                    
                    # Generate hash of the new audio file for duplicate checking
                    hash_md5 = hashlib.md5()
                    for chunk in audio_file.chunks():
                        hash_md5.update(chunk)
                    new_audio_hash = hash_md5.hexdigest()
                    
                    # Check against existing records (excluding current record)
                    duplicates = NoiseDataset.objects.filter(collector=request.user).exclude(pk=pk)
                    for dataset in duplicates:
                        if dataset.audio:
                            # Reset file pointer after reading chunks for hash
                            audio_file.seek(0)
                            
                            # Compare hashes
                            if dataset.get_audio_hash() == new_audio_hash:
                                messages.error(request, "This audio file has already been uploaded!")
                                return redirect('data:noise_dataset_edit', pk=pk)
                    
                    # Reset file pointer again before processing
                    audio_file.seek(0)
                    
                    # Get file extension
                    file_extension = audio_file.name.split('.')[-1].lower()
                    
                    # Generate new filename using existing noise_id
                    new_filename = f"{noise_dataset.noise_id}.{file_extension}"
                    
                    # Create a renamed file object
                    renamed_file = RenamedFile(audio_file, new_filename)
                    
                    # Replace the file in the form data
                    request.FILES['audio'] = renamed_file
                    
                    # Calculate duration

                
                # Update dataset name based on new values
                updated_dataset.name = generate_dataset_name(updated_dataset)
                updated_dataset.save()
                form.save_m2m()
                
                messages.success(request, f'Noise dataset "{updated_dataset.name}" updated successfully!')
                return redirect('data:noise_dataset_detail', dataset_id=pk)
                
            except Exception as e:
                messages.error(request, f'Error updating dataset: {str(e)}')
                print(f"Error updating noise dataset: {e}")
    else:
        form = NoiseDatasetForm(instance=noise_dataset)
    
    context = {
        'form': form,
        'noise_dataset': noise_dataset,
    }
    return render(request, 'data/AddNewData.html', context)


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



def noise_detail(request, dataset_id):
    dataset = get_object_or_404(NoiseDataset, pk=dataset_id)
    
    # Get related data
    audio_features = getattr(dataset, 'audio_features', None)
    noise_analysis = getattr(dataset, 'noise_analysis', None)
    
    # Prepare visualization data
    context = {
        'noise_dataset': dataset,
        'audio_features': audio_features,
        'noise_analysis': noise_analysis,
    }
    
    # Add visualizations if features exist
    if audio_features:
        # Waveform plot
       
        if audio_features.waveform_data:
            waveform_fig = create_waveform_plot(audio_features.waveform_data)
            context['waveform_plot'] = waveform_fig.to_html(full_html=False)
        
        # Spectrogram plot
        if audio_features.mel_spectrogram:
            spectrogram_fig = create_spectrogram_plot(audio_features.mel_spectrogram)
            context['spectrogram_plot'] = spectrogram_fig.to_html(full_html=False)
        
        # MFCC plot
        if audio_features.mfccs:
            
            mfcc_fig = create_mfcc_plot(audio_features.mfccs)
            context['mfcc_plot'] = mfcc_fig.to_html(full_html=False)
        
        # Frequency features plot
        freq_fig = create_frequency_features_plot(audio_features)
        context['freq_plot'] = freq_fig.to_html(full_html=False)
    
    return render(request, 'data/Noise_detail.html', context)



def create_waveform_plot(waveform_data):
    fig = go.Figure()
    

    if isinstance(waveform_data, dict):
        x = waveform_data.get('time', [])
        y = waveform_data.get('amplitude', [])
  
    elif isinstance(waveform_data, list):
        x = list(range(len(waveform_data)))  
        y = waveform_data
    else:
        x = []
        y = []
    
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        name='Waveform',
        line=dict(color='#1f77b4')
    ))
    fig.update_layout(
        title='Audio Waveform',
        xaxis_title='Time (s)',
        yaxis_title='Amplitude',
        height=300,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig


def create_spectrogram_plot(spectrogram_data):
    fig = go.Figure()
    
  
    if isinstance(spectrogram_data, dict):
    
        z = spectrogram_data.get('values', [])
        x = spectrogram_data.get('time', [])
        y = spectrogram_data.get('freq', [])
    elif isinstance(spectrogram_data, list):
    
        z = spectrogram_data
        x = list(range(len(spectrogram_data[0]))) if spectrogram_data else []
        y = list(range(len(spectrogram_data))) if spectrogram_data else []
    else:
        z, x, y = [], [], []
    
    if z:  # Only plot if we have data
        fig.add_trace(go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale='Jet'
        ))
    
    fig.update_layout(
        title='Mel Spectrogram',
        xaxis_title='Time (s)',
        yaxis_title='Frequency (Hz)',
        height=300,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig


def create_mfcc_plot(mfccs):
    # Debug: Print the input to see what we're working with
    print(f"Raw MFCCs data type: {type(mfccs)}")
    if hasattr(mfccs, 'shape'):
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
            title='MFCC Coefficients (No Data)',
            height=300,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        return fig
    
    # Ensure 2D array
    if len(mfccs.shape) == 1:
        mfccs = np.expand_dims(mfccs, axis=0)
    
    print(f"Final MFCCs shape: {mfccs.shape}")
    
    # Create the plot
    fig = go.Figure(data=go.Heatmap(
        z=mfccs,
        colorscale='Viridis',
        colorbar=dict(title='Magnitude')
    ))
    
    fig.update_layout(
        title='MFCC Coefficients',
        xaxis_title='Coefficient Index',
        yaxis_title='Frame',
        height=300,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig
def create_frequency_features_plot(audio_features):
    fig = go.Figure()
    
    features = [
        ('spectral_centroid', 'Spectral Centroid', 'red'),
        ('spectral_bandwidth', 'Spectral Bandwidth', 'blue'),
        ('spectral_rolloff', 'Spectral Rolloff', 'green'),
    ]
    
    for i, (attr, name, color) in enumerate(features, start=1):
        value = getattr(audio_features, attr, None)
        if value is not None:
            fig.add_trace(go.Scatter(
                x=[i],
                y=[value],
                mode='markers+text',
                name=name,
                text=[name],
                textposition='top center',
                marker=dict(size=12, color=color)
            ))
    
    fig.update_layout(
        title='Spectral Features Comparison',
        xaxis=dict(
            title='Feature Type',
            tickvals=[1, 2, 3],
            ticktext=['Centroid', 'Bandwidth', 'Rolloff'],
            range=[0.5, 3.5]  # Add some padding
        ),
        yaxis_title='Frequency (Hz)',
        height=300,
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=False  # Since we're showing labels on markers
    )
    return fig

