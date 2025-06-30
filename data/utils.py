import os
import numpy as np
import librosa
import tempfile
import logging
from scipy import signal as scipy_signal

from .models import NoiseDataset, AudioFeature, NoiseAnalysis, VisualizationPreset

logger = logging.getLogger(__name__)

def process_audio_file(instance: NoiseDataset):
    """Full audio processing pipeline that mimics post_save signal"""
    if not instance.audio:
        logger.warning(f"No audio file found for NoiseDataset {instance.id}")
        return

    try:
        # Generate noise_id
        if not instance.noise_id:
            audio_hash = instance.get_audio_hash()
            instance.noise_id = f"NOISE_{audio_hash[:8]}" if audio_hash else f"NOISE_{instance.id}"
            instance.save(update_fields=['noise_id'])

        # Generate name
        if not instance.name:
            name_parts = filter(None, [
                str(instance.region),
                str(instance.community),
                str(instance.category),
                str(instance.class_name)
            ])
            instance.name = "_".join(name_parts) or f"Audio_{instance.id}"
            instance.save(update_fields=['name'])

        # Save audio temporarily
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            for chunk in instance.audio.chunks():
                tmp_file.write(chunk)
            audio_path = tmp_file.name

        try:
            extract_audio_features(instance, audio_path)
            perform_noise_analysis(instance, audio_path)
            create_visualization_presets(instance)
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)

    except Exception as e:
        logger.error(f"Error processing audio: {e}", exc_info=True)


def extract_audio_features(instance, audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)

        rms_energy = librosa.feature.rms(y=y)[0]
        zcr = librosa.feature.zero_crossing_rate(y=y)[0]

        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1).tolist()

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1).tolist()

        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        waveform_data = y[::max(1, len(y)//1000)].tolist()

        y_harmonic, y_percussive = librosa.effects.hpss(y)
        harmonic_ratio = float(np.sum(y_harmonic**2) / np.sum(y**2))
        percussive_ratio = float(np.sum(y_percussive**2) / np.sum(y**2))

        AudioFeature.objects.update_or_create(
            noise_dataset=instance,
            defaults={
                'sample_rate': sr,
                'num_samples': len(y),
                'duration': duration,
                'rms_energy': float(np.mean(rms_energy)),
                'zero_crossing_rate': float(np.mean(zcr)),
                'spectral_centroid': float(np.mean(spectral_centroid)),
                'spectral_bandwidth': float(np.mean(spectral_bandwidth)),
                'spectral_rolloff': float(np.mean(spectral_rolloff)),
                'spectral_flatness': float(np.mean(spectral_flatness)),
                'mfccs': mfccs_mean,
                'chroma_stft': chroma_mean,
                'mel_spectrogram': mel_spec_db.tolist(),
                'waveform_data': waveform_data,
                'harmonic_ratio': harmonic_ratio,
                'percussive_ratio': percussive_ratio,
            }
        )

    except Exception as e:
        logger.error(f"Error extracting audio features: {e}", exc_info=True)
        raise


def perform_noise_analysis(instance, audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        y_db = librosa.amplitude_to_db(np.abs(y), ref=np.max)

        mean_db = float(np.mean(y_db))
        max_db = float(np.max(y_db))
        min_db = float(np.min(y_db))
        std_db = float(np.std(y_db))

        peaks, _ = scipy_signal.find_peaks(np.abs(y), height=np.std(y) * 2)
        peak_count = len(peaks)
        peak_interval_mean = float(np.mean(np.diff(peaks)/sr)) if len(peaks) > 1 else 0.0

        fft = np.fft.fft(y)
        freqs = np.fft.fftfreq(len(fft), 1/sr)
        magnitude = np.abs(fft)
        dominant_freq_idx = np.argmax(magnitude[:len(magnitude)//2])
        dominant_frequency = float(abs(freqs[dominant_freq_idx]))

        energy_cumsum = np.cumsum(magnitude[:len(magnitude)//2])
        total_energy = energy_cumsum[-1]
        freq_range_low = freqs[np.where(energy_cumsum >= 0.05 * total_energy)[0][0]]
        freq_range_high = freqs[np.where(energy_cumsum >= 0.95 * total_energy)[0][0]]
        frequency_range = f"{freq_range_low:.1f}-{freq_range_high:.1f}"

        frame_length = sr // 10
        frames = [y[i:i+frame_length] for i in range(0, len(y), frame_length)]
        frame_energies = [np.sum(frame**2) for frame in frames if len(frame) == frame_length]
        energy_threshold = np.mean(frame_energies) + 2 * np.std(frame_energies)
        event_frames = [i for i, energy in enumerate(frame_energies) if energy > energy_threshold]

        events = []
        if event_frames:
            current_event = [event_frames[0], event_frames[0]]
            for frame in event_frames[1:]:
                if frame == current_event[1] + 1:
                    current_event[1] = frame
                else:
                    events.append(current_event)
                    current_event = [frame, frame]
            events.append(current_event)

        event_count = len(events)
        event_durations = [float((end - start + 1) * 0.1) for start, end in events]

        NoiseAnalysis.objects.update_or_create(
            noise_dataset=instance,
            defaults={
                'mean_db': mean_db,
                'max_db': max_db,
                'min_db': min_db,
                'std_db': std_db,
                'peak_count': peak_count,
                'peak_interval_mean': peak_interval_mean,
                'dominant_frequency': dominant_frequency,
                'frequency_range': frequency_range,
                'event_count': event_count,
                'event_durations': event_durations,
            }
        )

    except Exception as e:
        logger.error(f"Error performing noise analysis: {e}", exc_info=True)
        raise


def create_visualization_presets(instance):
    try:
        features = instance.audio_features
        VisualizationPreset.objects.get_or_create(
            noise_dataset=instance,
            chart_type='waveform',
            defaults={
                'name': f'Waveform - {instance.name}',
                'description': 'Waveform visualization of the audio',
                'config': {
                    'title': 'Waveform',
                    'x_label': 'Time (s)',
                    'y_label': 'Amplitude',
                    'color': '#1f77b4',
                    'data': features.waveform_data
                },
                'alt_text_template': f'Waveform of {instance.name}'
            }
        )
        VisualizationPreset.objects.get_or_create(
            noise_dataset=instance,
            chart_type='spectrogram',
            defaults={
                'name': f'Spectrogram - {instance.name}',
                'description': 'Spectrogram visualization of the audio',
                'config': {
                    'title': 'Spectrogram',
                    'x_label': 'Time (s)',
                    'y_label': 'Frequency (Hz)',
                    'cmap': 'viridis',
                    'data': features.mel_spectrogram
                },
                'alt_text_template': f'Spectrogram of {instance.name}'
            }
        )
        VisualizationPreset.objects.get_or_create(
            noise_dataset=instance,
            chart_type='mfcc',
            defaults={
                'name': f'MFCC - {instance.name}',
                'description': 'MFCC coefficients visualization',
                'config': {
                    'title': 'MFCC Coefficients',
                    'x_label': 'Frame',
                    'y_label': 'MFCC Coefficient',
                    'cmap': 'coolwarm',
                    'data': features.mfccs
                },
                'alt_text_template': f'MFCC coefficients of {instance.name}'
            }
        )
    except Exception as e:
        logger.error(f"Error creating visualization presets: {e}", exc_info=True)
        raise
