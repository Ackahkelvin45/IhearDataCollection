from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import NoiseDataset, AudioFeature, NoiseAnalysis
from data.utils.audio_processing import extract_audio_features, analyze_noise_characteristics
import numpy as np
import json
import threading

def process_audio_async(instance):
    """
    Helper function to process audio in a separate thread
    """
    try:
        # Extract audio features
        features = extract_audio_features(instance.audio)
        print(features)
        
        # Only proceed if feature extraction was successful
        if features is None:
            print(f"Failed to extract features for {instance.id}")
            return
            
        # Create AudioFeature record
        AudioFeature.objects.create(
            noise_dataset=instance,
            **features
        )
        
        # Perform noise analysis only if we have waveform data
        if 'waveform_data' in features:
            try:
                waveform = np.array(json.loads(features['waveform_data']))
                sample_rate = features.get('sample_rate', 44100)
                
                analysis_data = analyze_noise_characteristics(waveform, sample_rate)
                
                # Only create analysis record if we got valid data
                if analysis_data is not None:
                    NoiseAnalysis.objects.create(
                        noise_dataset=instance,
                        **analysis_data
                    )
            except Exception as e:
                print(f"Error in noise analysis: {e}")
        
    except Exception as e:
        print(f"Error processing audio file: {e}")
        # You might want to log this error properly in production

@receiver(post_save, sender=NoiseDataset)
def extract_features_on_save(sender, instance, created, **kwargs):
    """
    Automatically extract audio features when a new NoiseDataset is saved
    """
    if created and instance.audio:
        # Start a new thread to process the audio
        thread = threading.Thread(
            target=process_audio_async,
            args=(instance,)
        )
        thread.daemon = True  # This ensures the thread won't prevent program exit
        thread.start()