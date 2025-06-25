import os
import json
import numpy as np
import librosa
import hashlib
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.core.files.storage import default_storage
from scipy import signal as scipy_signal
from scipy.stats import skew, kurtosis
import tempfile
import logging

from data.models import NoiseDataset, AudioFeature, NoiseAnalysis, VisualizationPreset

logger = logging.getLogger(__name__)

@receiver(post_save, sender=NoiseDataset)
def process_audio_signal(sender, instance, created, **kwargs):
    """Post-save signal to process audio and create features/analysis"""
    if not instance.audio:
        logger.warning(f"No audio file found for NoiseDataset {instance.id}")
        return
    
    try:
        # Generate IDs and names if not exists
        if not instance.noise_id:
            audio_hash = instance.get_audio_hash()
            instance.noise_id = f"NOISE_{audio_hash[:8]}" if audio_hash else f"NOISE_{instance.id}"
            instance.save(update_fields=['noise_id'])
        
        if not instance.name:
            name_parts = filter(None, [
                str(instance.region),
                str(instance.community),
                str(instance.category),
                str(instance.class_name)
            ])
            instance.name = "_".join(name_parts) if name_parts else f"Audio_{instance.id}"
            instance.save(update_fields=['name'])
        
        # Process audio file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            for chunk in instance.audio.chunks():
                tmp_file.write(chunk)
            audio_path = tmp_file.name
        
        try:
            # Extract comprehensive features
            extract_audio_features(instance, audio_path)
            
            # Perform noise analysis
            perform_noise_analysis(instance, audio_path)
            
            # Create default visualization presets
            create_visualization_presets(instance)
            
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
                
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}", exc_info=True)

def extract_audio_features(noise_dataset, audio_path):
    """Extract comprehensive audio features for ML training"""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Update duration if needed
        if not noise_dataset.duration:
            noise_dataset.duration = duration
            noise_dataset.save(update_fields=['duration'])
        
        # Time-domain features
        rms_energy = librosa.feature.rms(y=y)[0]
        zcr = librosa.feature.zero_crossing_rate(y=y)[0]
        
        # Frequency-domain features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        
        # Advanced features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        
        # Temporal features
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
        
        # Harmonic/percussive separation
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        harmonic_ratio = np.sum(y_harmonic**2) / np.sum(y**2)
        percussive_ratio = np.sum(y_percussive**2) / np.sum(y**2)
        
        # Visualization data
        waveform_data = y[::max(1, len(y)//1000)]  # Downsample for visualization
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Create AudioFeature record
        AudioFeature.objects.update_or_create(
            noise_dataset=noise_dataset,
            defaults={
                'sample_rate': sr,
                'num_samples': len(y),
                'duration': duration,
                'rms_energy': float(np.mean(rms_energy)),
                'zero_crossing_rate': float(np.mean(zcr)),
                'amplitude_envelope': rms_energy.tolist(),
                'spectral_centroid': float(np.mean(spectral_centroid)),
                'spectral_bandwidth': float(np.mean(spectral_bandwidth)),
                'spectral_rolloff': float(np.mean(spectral_rolloff)),
                'spectral_flatness': float(np.mean(spectral_flatness)),
                'mfccs': np.mean(mfccs, axis=1).tolist(),
                'chroma_stft': np.mean(chroma_stft, axis=1).tolist(),
                'tonnetz': np.mean(tonnetz, axis=1).tolist(),
                'tempogram': np.mean(tempogram, axis=1).tolist(),
                'onset_strength': onset_env.tolist(),
                'harmonic_ratio': float(harmonic_ratio),
                'percussive_ratio': float(percussive_ratio),
                'waveform_data': waveform_data.tolist(),
                'spectrogram': librosa.amplitude_to_db(np.abs(librosa.stft(y))).tolist(),
                'mel_spectrogram': mel_spec_db.tolist(),
            }
        )
        
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}", exc_info=True)
        raise

def perform_noise_analysis(noise_dataset, audio_path):
    """Perform comprehensive noise analysis on the audio"""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        
        # Convert to dB scale
        y_db = librosa.amplitude_to_db(np.abs(y), ref=np.max)
        
        # Statistical features
        mean_db = float(np.mean(y_db))
        max_db = float(np.max(y_db))
        min_db = float(np.min(y_db))
        std_db = float(np.std(y_db))
        
        # Peak detection
        peaks, _ = scipy_signal.find_peaks(np.abs(y), height=np.std(y) * 2)
        peak_count = len(peaks)
        
        # Peak interval analysis
        if len(peaks) > 1:
            peak_intervals = np.diff(peaks) / sr
            peak_interval_mean = float(np.mean(peak_intervals))
        else:
            peak_interval_mean = 0.0
        
        # Frequency analysis
        fft = np.fft.fft(y)
        freqs = np.fft.fftfreq(len(fft), 1/sr)
        magnitude = np.abs(fft)
        
        # Dominant frequency
        dominant_freq_idx = np.argmax(magnitude[:len(magnitude)//2])
        dominant_frequency = float(abs(freqs[dominant_freq_idx]))
        
        # Frequency range (containing 90% of energy)
        energy_cumsum = np.cumsum(magnitude[:len(magnitude)//2])
        total_energy = energy_cumsum[-1]
        freq_range_low = freqs[np.where(energy_cumsum >= 0.05 * total_energy)[0][0]]
        freq_range_high = freqs[np.where(energy_cumsum >= 0.95 * total_energy)[0][0]]
        frequency_range = f"{freq_range_low:.1f}-{freq_range_high:.1f}"
        
        # Psychoacoustic metrics (simplified approximations)
        loudness = float(20 * np.log10(np.sqrt(np.mean(y**2)) + 1e-10))
        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0]))
        sharpness = float(spectral_centroid / 1000)  # Normalized approximation
        roughness = float(np.mean(np.std(librosa.stft(y), axis=1)))
        
        # Event detection (simplified)
        frame_length = sr // 10  # 100ms frames
        frames = [y[i:i+frame_length] for i in range(0, len(y), frame_length)]
        frame_energies = [np.sum(frame**2) for frame in frames if len(frame) == frame_length]
        
        # Detect events as frames with energy above threshold
        energy_threshold = np.mean(frame_energies) + 2 * np.std(frame_energies)
        event_frames = [i for i, energy in enumerate(frame_energies) if energy > energy_threshold]
        
        # Group consecutive event frames
        events = []
        if event_frames:
            current_event_start = event_frames[0]
            current_event_end = event_frames[0]
            
            for frame in event_frames[1:]:
                if frame == current_event_end + 1:
                    current_event_end = frame
                else:
                    events.append((current_event_start, current_event_end))
                    current_event_start = frame
                    current_event_end = frame
            events.append((current_event_start, current_event_end))
        
        event_count = len(events)
        event_durations = [float((end - start + 1) * 0.1) for start, end in events]  # 0.1s per frame
        
        # Create or update NoiseAnalysis record
        NoiseAnalysis.objects.update_or_create(
            noise_dataset=noise_dataset,
            defaults={
                'mean_db': mean_db,
                'max_db': max_db,
                'min_db': min_db,
                'std_db': std_db,
                'peak_count': peak_count,
                'peak_interval_mean': peak_interval_mean,
                'dominant_frequency': dominant_frequency,
                'frequency_range': frequency_range,
                'loudness': loudness,
                'sharpness': sharpness,
                'roughness': roughness,
                'event_count': event_count,
                'event_durations': event_durations,
            }
        )
        
    except Exception as e:
        logger.error(f"Error performing noise analysis: {str(e)}", exc_info=True)
        raise

def create_visualization_presets(noise_dataset):
    """Create default visualization presets for this audio"""
    try:
        # Waveform visualization
        VisualizationPreset.objects.get_or_create(
            name=f"Waveform - {noise_dataset.name}",
            noise_dataset=noise_dataset,
            defaults={
                'chart_type': 'waveform',
                'config': {
                    'title': 'Audio Waveform',
                    'x_label': 'Time (s)',
                    'y_label': 'Amplitude',
                    'color': '#1f77b4',
                    'height': 400
                },
                'alt_text_template': 'Waveform visualization of {noise_dataset.name}'
            }
        )
        
        # Spectrogram visualization
        VisualizationPreset.objects.get_or_create(
            name=f"Spectrogram - {noise_dataset.name}",
            noise_dataset=noise_dataset,
            defaults={
                'chart_type': 'spectrogram',
                'config': {
                    'title': 'Spectrogram',
                    'x_label': 'Time (s)',
                    'y_label': 'Frequency (Hz)',
                    'cmap': 'viridis',
                    'height': 500
                },
                'alt_text_template': 'Spectrogram of {noise_dataset.name}'
            }
        )
        
        # MFCC visualization
        VisualizationPreset.objects.get_or_create(
            name=f"MFCC - {noise_dataset.name}",
            noise_dataset=noise_dataset,
            defaults={
                'chart_type': 'mfcc',
                'config': {
                    'title': 'MFCC Coefficients',
                    'x_label': 'Frame',
                    'y_label': 'MFCC Coefficient',
                    'cmap': 'coolwarm',
                    'height': 500
                },
                'alt_text_template': 'MFCC coefficients of {noise_dataset.name}'
            }
        )
        
    except Exception as e:
        logger.error(f"Error creating visualization presets: {str(e)}")