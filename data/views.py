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
                
                # Generate dataset name (you might want to update this too)
                noise_dataset.name = generate_dataset_name(noise_dataset)
                
                # Calculate audio duration if audio file is provided
                if 'audio' in request.FILES:
                    try:
                        # Rename the uploaded file to match the noise ID
                        audio_file = request.FILES['audio']
                        file_extension = audio_file.name.split('.')[-1]
                        new_filename = f"{noise_dataset.noise_id}.{file_extension}"
                        audio_file.name = new_filename
                        
                        duration = get_audio_duration(audio_file)
                        if duration:
                            noise_dataset.duration = duration
                        else:
                            print("Could not determine duration")  
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
