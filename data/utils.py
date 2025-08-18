import os
import numpy as np
import librosa
import tempfile
import logging
from scipy import signal as scipy_signal
import soundfile as sf
import audioread
from typing import Optional
from .models import NoiseDataset, AudioFeature, NoiseAnalysis, VisualizationPreset
import uuid
from datetime import datetime
from django.utils import timezone

logger = logging.getLogger(__name__)

# Supported audio formats and their common extensions
SUPPORTED_FORMATS = {
    "wav": [".wav", ".wave"],
    "mp3": [".mp3"],
    "flac": [".flac"],
    "ogg": [".ogg", ".oga", ".opus"],
    "aiff": [".aiff", ".aif"],
    "m4a": [".m4a"],
}


def check_audio_backends():
    """Check if required audio backends are available"""
    try:
        # Test soundfile backend
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            sf.write(tmp.name, np.zeros(1000), 44100)
            sf.read(tmp.name)

        # Test audioread backend
        with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp:
            try:
                with audioread.audio_open(tmp.name):
                    pass
            except audioread.exceptions.NoBackendError:
                logger.warning("audiored MP3 backend not available")

        return True
    except Exception as e:
        logger.error(f"Audio backend check failed: {e}")
        return False


def get_file_extension(filename: str) -> Optional[str]:
    """Get normalized file extension from filename"""
    if not filename:
        return None

    # Get the last part after dot (handle multiple dots like .tar.gz)
    base, ext = os.path.splitext(filename.lower())
    ext = ext.strip()

    # Check for compound extensions
    if not ext and "." in base:
        ext = "." + base.split(".")[-1]

    return ext if ext else None


def is_supported_extension(ext: str) -> bool:
    """Check if extension is in our supported formats list"""
    if not ext:
        return False
    return any(ext in extensions for extensions in SUPPORTED_FORMATS.values())


def get_audio_format(ext: str) -> Optional[str]:
    """Get audio format from extension"""
    for fmt, extensions in SUPPORTED_FORMATS.items():
        if ext in extensions:
            return fmt
    return None


def load_audio_file(audio_path: str, ext: str):
    """Robust audio file loading with multiple fallback strategies"""
    try:
        # First try librosa with native sample rate
        y, sr = librosa.load(audio_path, sr=None)
        logger.info(f"Loaded {audio_path} with librosa (native SR)")
        return y, sr
    except Exception as primary_error:
        logger.warning(f"Primary load failed for {audio_path}: {primary_error}")

        # Try with soundfile if format is supported
        if (
            ext
            in SUPPORTED_FORMATS["wav"]
            + SUPPORTED_FORMATS["flac"]
            + SUPPORTED_FORMATS["aiff"]
        ):
            try:
                y, sr = sf.read(audio_path)
                logger.info(f"Loaded {audio_path} with soundfile")
                return y, sr
            except Exception as sf_error:
                logger.warning(f"Soundfile load failed: {sf_error}")

        # Try with standard sample rate as last resort
        try:
            y, sr = librosa.load(audio_path, sr=22050)
            logger.info(f"Loaded {audio_path} with librosa (standard SR)")
            return y, sr
        except Exception as fallback_error:
            logger.error(
                f"All loading attempts failed for {audio_path}: {fallback_error}"
            )
            raise


def process_audio_file(instance: NoiseDataset):
    """Full audio processing pipeline that mimics post_save signal"""
    if not instance.audio:
        logger.warning(f"No audio file found for NoiseDataset {instance.id}")
        return

    try:
        # Generate noise_id
        if not instance.noise_id:
            audio_hash = instance.get_audio_hash()
            instance.noise_id = (
                f"NOISE_{audio_hash[:8]}" if audio_hash else f"NOISE_{instance.id}"
            )
            instance.save(update_fields=["noise_id"])

        # Generate name
        if not instance.name:
            name_parts = filter(
                None,
                [
                    str(instance.region),
                    str(instance.community),
                    str(instance.category),
                    str(instance.class_name),
                ],
            )
            instance.name = "_".join(name_parts) or f"Audio_{instance.id}"
            instance.save(update_fields=["name"])

        # Get file extension
        file_ext = get_file_extension(instance.audio.name)
        if not file_ext or not is_supported_extension(file_ext):
            file_ext = ".wav"  # Default to WAV if unknown extension
            logger.info(f"Using default .wav extension for file {instance.audio.name}")

        # Save audio temporarily with proper extension
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp_file:
            for chunk in instance.audio.chunks():
                tmp_file.write(chunk)
            audio_path = tmp_file.name

        logger.info(
            f"Processing audio file: {audio_path}, "
            f"size: {os.path.getsize(audio_path)} bytes, "
            f"ext: {file_ext}"
        )

        try:
            # Verify file exists and has content
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                raise ValueError("Temporary audio file is empty or doesn't exist")

            extract_audio_features(instance, audio_path, file_ext)
            perform_noise_analysis(instance, audio_path, file_ext)
            create_visualization_presets(instance)
        except Exception as processing_error:
            logger.error(
                f"Error processing audio file {audio_path}: {processing_error}",
                exc_info=True,
            )
            raise
        finally:
            if os.path.exists(audio_path):
                try:
                    os.unlink(audio_path)
                except Exception as cleanup_error:
                    logger.warning(
                        f"Could not delete temp file {audio_path}: {cleanup_error}"
                    )

    except Exception as e:
        logger.error(f"Error in audio processing pipeline: {e}", exc_info=True)
        raise


def extract_audio_features(instance, audio_path, file_ext):
    """Extract audio features with robust error handling"""
    try:
        logger.info(f"Extracting features from {audio_path}")

        # Load audio with our robust loader
        y, sr = load_audio_file(audio_path, file_ext)

        duration = librosa.get_duration(y=y, sr=sr)
        logger.info(
            f"Audio loaded successfully: duration={duration:.2f}s, sr={sr}, samples={len(y)}"
        )

        # Convert to mono if needed
        if len(y.shape) > 1:
            y = librosa.to_mono(y)
            logger.info(f"Converted audio to mono, new shape: {y.shape}")

        # RMS Energy
        rms_energy = librosa.feature.rms(y=y)[0]

        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y=y)[0]

        # Spectral Features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]

        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1).tolist()

        # Chroma Features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1).tolist()

        # Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Waveform data (downsampled for visualization)
        waveform_data = y[:: max(1, len(y) // 1000)].tolist()

        # Harmonic/Percussive Separation
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        harmonic_ratio = float(np.sum(y_harmonic**2) / np.sum(y**2))
        percussive_ratio = float(np.sum(y_percussive**2) / np.sum(y**2))

        # Create/update audio features
        AudioFeature.objects.update_or_create(
            noise_dataset=instance,
            defaults={
                "sample_rate": sr,
                "num_samples": len(y),
                "duration": duration,
                "rms_energy": float(np.mean(rms_energy)),
                "zero_crossing_rate": float(np.mean(zcr)),
                "spectral_centroid": float(np.mean(spectral_centroid)),
                "spectral_bandwidth": float(np.mean(spectral_bandwidth)),
                "spectral_rolloff": float(np.mean(spectral_rolloff)),
                "spectral_flatness": float(np.mean(spectral_flatness)),
                "mfccs": mfccs_mean,
                "chroma_stft": chroma_mean,
                "mel_spectrogram": mel_spec_db.tolist(),
                "waveform_data": waveform_data,
                "harmonic_ratio": harmonic_ratio,
                "percussive_ratio": percussive_ratio,
            },
        )

        logger.info(f"Successfully extracted features for {instance.noise_id}")

    except Exception as e:
        logger.error(
            f"Error extracting audio features from {audio_path}: {e}", exc_info=True
        )
        raise


def perform_noise_analysis(instance, audio_path, file_ext):
    """Perform detailed noise analysis with robust error handling"""
    try:
        logger.info(f"Performing noise analysis on {audio_path}")

        # Load audio with our robust loader
        y, sr = load_audio_file(audio_path, file_ext)

        # Convert to mono if needed
        if len(y.shape) > 1:
            y = librosa.to_mono(y)

        y_db = librosa.amplitude_to_db(np.abs(y), ref=np.max)

        # Basic dB statistics
        mean_db = float(np.mean(y_db))
        max_db = float(np.max(y_db))
        min_db = float(np.min(y_db))
        std_db = float(np.std(y_db))

        # Peak detection
        peaks, _ = scipy_signal.find_peaks(np.abs(y), height=np.std(y) * 2)
        peak_count = len(peaks)
        peak_interval_mean = (
            float(np.mean(np.diff(peaks) / sr)) if len(peaks) > 1 else 0.0
        )

        # Frequency analysis
        fft = np.fft.fft(y)
        freqs = np.fft.fftfreq(len(fft), 1 / sr)
        magnitude = np.abs(fft)
        dominant_freq_idx = np.argmax(magnitude[: len(magnitude) // 2])
        dominant_frequency = float(abs(freqs[dominant_freq_idx]))

        # Frequency range containing 90% of energy
        energy_cumsum = np.cumsum(magnitude[: len(magnitude) // 2])
        total_energy = energy_cumsum[-1]
        freq_range_low = freqs[np.where(energy_cumsum >= 0.05 * total_energy)[0][0]]
        freq_range_high = freqs[np.where(energy_cumsum >= 0.95 * total_energy)[0][0]]
        frequency_range = f"{freq_range_low:.1f}-{freq_range_high:.1f}"

        # Event detection
        frame_length = sr // 10  # 100ms frames
        frames = [y[i : i + frame_length] for i in range(0, len(y), frame_length)]
        frame_energies = [
            np.sum(frame**2) for frame in frames if len(frame) == frame_length
        ]
        energy_threshold = np.mean(frame_energies) + 2 * np.std(frame_energies)
        event_frames = [
            i for i, energy in enumerate(frame_energies) if energy > energy_threshold
        ]

        # Cluster consecutive events
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
        event_durations = [
            float((end - start + 1) * 0.1) for start, end in events
        ]  # Convert to seconds

        NoiseAnalysis.objects.update_or_create(
            noise_dataset=instance,
            defaults={
                "mean_db": mean_db,
                "max_db": max_db,
                "min_db": min_db,
                "std_db": std_db,
                "peak_count": peak_count,
                "peak_interval_mean": peak_interval_mean,
                "dominant_frequency": dominant_frequency,
                "frequency_range": frequency_range,
                "event_count": event_count,
                "event_durations": event_durations,
            },
        )

        logger.info(f"Completed noise analysis for {instance.noise_id}")

    except Exception as e:
        logger.error(
            f"Error performing noise analysis on {audio_path}: {e}", exc_info=True
        )
        raise


def create_visualization_presets(instance):
    """Create standard visualization presets for the audio"""
    try:
        features = instance.audio_features
        analysis = instance.noise_analysis

        # Waveform preset
        VisualizationPreset.objects.get_or_create(
            noise_dataset=instance,
            chart_type="waveform",
            defaults={
                "name": f"Waveform - {instance.name}",
                "description": "Waveform visualization of the audio",
                "config": {
                    "title": "Waveform",
                    "x_label": "Time (s)",
                    "y_label": "Amplitude",
                    "color": "#1f77b4",
                    "data": features.waveform_data,
                    "duration": features.duration,
                },
                "alt_text_template": f"Waveform of {instance.name} showing amplitude over time",
            },
        )

        # Spectrogram preset
        VisualizationPreset.objects.get_or_create(
            noise_dataset=instance,
            chart_type="spectrogram",
            defaults={
                "name": f"Spectrogram - {instance.name}",
                "description": "Spectrogram visualization of the audio",
                "config": {
                    "title": "Spectrogram",
                    "x_label": "Time (s)",
                    "y_label": "Frequency (Hz)",
                    "cmap": "viridis",
                    "data": features.mel_spectrogram,
                    "sample_rate": features.sample_rate,
                    "duration": features.duration,
                },
                "alt_text_template": f"Spectrogram of {instance.name}"
                f" showing frequency content over time",
            },
        )

        # MFCC preset
        VisualizationPreset.objects.get_or_create(
            noise_dataset=instance,
            chart_type="mfcc",
            defaults={
                "name": f"MFCC - {instance.name}",
                "description": "MFCC coefficients visualization",
                "config": {
                    "title": "MFCC Coefficients",
                    "x_label": "Frame",
                    "y_label": "MFCC Coefficient",
                    "cmap": "coolwarm",
                    "data": features.mfccs,
                    "num_coefficients": len(features.mfccs),
                },
                "alt_text_template": f"MFCC coefficients of {instance.name}"
                f" showing timbral characteristics",
            },
        )

        # Frequency analysis preset
        VisualizationPreset.objects.get_or_create(
            noise_dataset=instance,
            chart_type="frequency",
            defaults={
                "name": f"Frequency Analysis - {instance.name}",
                "description": "Frequency domain analysis",
                "config": {
                    "title": "Frequency Analysis",
                    "x_label": "Frequency (Hz)",
                    "y_label": "Magnitude",
                    "dominant_frequency": analysis.dominant_frequency,
                    "frequency_range": analysis.frequency_range,
                },
                "alt_text_template": f"Frequency analysis of {instance.name}"
                f" showing dominant frequency at"
                f" {analysis.dominant_frequency:.1f}Hz",
            },
        )

        logger.info(f"Created visualization presets for {instance.noise_id}")

    except Exception as e:
        logger.error(f"Error creating visualization presets: {e}", exc_info=True)
        raise


def generate_dataset_name(noise_dataset):
    parts = []

    if noise_dataset.category:
        parts.append(str(noise_dataset.category.name))

    if noise_dataset.class_name:
        parts.append(str(noise_dataset.class_name.name))

    if noise_dataset.subclass:
        parts.append(str(noise_dataset.subclass.name))

    timestamp = timezone.now().strftime("%Y%m%d_%H%M")
    parts.append(timestamp)

    return "_".join(parts) if parts else f"NoiseDataset_{timestamp}"


def generate_noise_id(user):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_chars = uuid.uuid4().hex[:3].upper()
    speaker_id = (
        user.speaker_id if hasattr(user, "speaker_id") and user.speaker_id else "UNK"
    )
    return f"NSE-{speaker_id}-{timestamp}-{random_chars}"


def safe_process_audio_file(instance: NoiseDataset):
    """Safe audio processing function that won't conflict with Numba JIT compilation"""
    try:
        # Import here to avoid Numba compilation issues
        import numba
        import os

        # Set environment variable to disable numba JIT completely
        os.environ['NUMBA_DISABLE_JIT'] = '1'
        
        # Also disable JIT in numba config
        original_disable = numba.config.DISABLE_JIT
        numba.config.DISABLE_JIT = True

        try:
            # Call the original processing function
            process_audio_file(instance)
        except AttributeError as e:
            if "get_call_template" in str(e):
                logger.warning(f"Numba compatibility issue detected for {instance.noise_id}, retrying with different approach")
                # Try processing without numba entirely
                _process_audio_without_numba(instance)
            else:
                raise
        finally:
            # Restore original setting
            numba.config.DISABLE_JIT = original_disable
            # Remove environment variable
            if 'NUMBA_DISABLE_JIT' in os.environ:
                del os.environ['NUMBA_DISABLE_JIT']

    except Exception as e:
        logger.error(f"Error in safe audio processing: {e}")
        raise


def _process_audio_without_numba(instance: NoiseDataset):
    """Process audio without using numba-optimized functions"""
    try:
        logger.info(f"Processing audio without numba for {instance.noise_id}")
        
        if not instance.audio:
            logger.warning(f"No audio file found for NoiseDataset {instance.id}")
            return

        # Generate noise_id
        if not instance.noise_id:
            audio_hash = instance.get_audio_hash()
            instance.noise_id = (
                f"NOISE_{audio_hash[:8]}" if audio_hash else f"NOISE_{instance.id}"
            )
            instance.save(update_fields=["noise_id"])

        # Generate name
        if not instance.name:
            name_parts = filter(
                None,
                [
                    str(instance.region),
                    str(instance.community),
                    str(instance.category),
                    str(instance.class_name),
                ],
            )
            instance.name = "_".join(name_parts) or f"Audio_{instance.id}"
            instance.save(update_fields=["name"])

        # Get file extension
        file_ext = get_file_extension(instance.audio.name)
        if not file_ext or not is_supported_extension(file_ext):
            file_ext = ".wav"  # Default to WAV if unknown extension
            logger.info(f"Using default .wav extension for file {instance.audio.name}")

        # Save audio temporarily with proper extension
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp_file:
            for chunk in instance.audio.chunks():
                tmp_file.write(chunk)
            audio_path = tmp_file.name

        logger.info(
            f"Processing audio file: {audio_path}, "
            f"size: {os.path.getsize(audio_path)} bytes, "
            f"ext: {file_ext}"
        )

        try:
            # Verify file exists and has content
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                raise ValueError("Temporary audio file is empty or doesn't exist")

            _extract_audio_features_simple(instance, audio_path, file_ext)
            _perform_noise_analysis_simple(instance, audio_path, file_ext)
            create_visualization_presets(instance)
        except Exception as processing_error:
            logger.error(
                f"Error processing audio file {audio_path}: {processing_error}",
                exc_info=True,
            )
            raise
        finally:
            if os.path.exists(audio_path):
                try:
                    os.unlink(audio_path)
                except Exception as cleanup_error:
                    logger.warning(
                        f"Could not delete temp file {audio_path}: {cleanup_error}"
                    )

    except Exception as e:
        logger.error(f"Error in non-numba audio processing pipeline: {e}", exc_info=True)
        raise


def _extract_audio_features_simple(instance, audio_path, file_ext):
    """Extract audio features without numba-optimized functions"""
    try:
        logger.info(f"Extracting features from {audio_path} (simple mode)")

        # Load audio with our robust loader
        y, sr = load_audio_file(audio_path, file_ext)

        duration = librosa.get_duration(y=y, sr=sr)
        logger.info(
            f"Audio loaded successfully: duration={duration:.2f}s, sr={sr}, samples={len(y)}"
        )

        # Convert to mono if needed
        if len(y.shape) > 1:
            y = librosa.to_mono(y)
            logger.info(f"Converted audio to mono, new shape: {y.shape}")

        # Basic features only to avoid numba issues
        rms_energy = float(np.mean(librosa.feature.rms(y=y)[0]))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)[0]))
        
        # Spectral features
        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0]))
        spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]))
        spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)[0]))
        spectral_flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)[0]))

        # MFCCs (simplified)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1).tolist()

        # Chroma Features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1).tolist()

        # Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Waveform data (downsampled for visualization)
        waveform_data = y[:: max(1, len(y) // 1000)].tolist()

        # Create/update audio features
        AudioFeature.objects.update_or_create(
            noise_dataset=instance,
            defaults={
                "sample_rate": sr,
                "num_samples": len(y),
                "duration": duration,
                "rms_energy": rms_energy,
                "zero_crossing_rate": zcr,
                "spectral_centroid": spectral_centroid,
                "spectral_bandwidth": spectral_bandwidth,
                "spectral_rolloff": spectral_rolloff,
                "spectral_flatness": spectral_flatness,
                "mfccs": mfccs_mean,
                "chroma_stft": chroma_mean,
                "mel_spectrogram": mel_spec_db.tolist(),
                "waveform_data": waveform_data,
                "harmonic_ratio": 0.5,  # Default values to avoid numba issues
                "percussive_ratio": 0.5,
            },
        )

        logger.info(f"Successfully extracted features for {instance.noise_id} (simple mode)")

    except Exception as e:
        logger.error(
            f"Error extracting audio features from {audio_path}: {e}", exc_info=True
        )
        raise


def _perform_noise_analysis_simple(instance, audio_path, file_ext):
    """Perform noise analysis without numba-optimized functions"""
    try:
        logger.info(f"Performing noise analysis on {audio_path} (simple mode)")

        # Load audio with our robust loader
        y, sr = load_audio_file(audio_path, file_ext)

        # Convert to mono if needed
        if len(y.shape) > 1:
            y = librosa.to_mono(y)

        y_db = librosa.amplitude_to_db(np.abs(y), ref=np.max)

        # Basic dB statistics
        mean_db = float(np.mean(y_db))
        max_db = float(np.max(y_db))
        min_db = float(np.min(y_db))
        std_db = float(np.std(y_db))

        # Peak detection (simplified)
        peaks, _ = scipy_signal.find_peaks(np.abs(y), height=np.std(y) * 2)
        peak_count = len(peaks)
        peak_interval_mean = (
            float(np.mean(np.diff(peaks) / sr)) if len(peaks) > 1 else 0.0
        )

        # Frequency analysis
        fft = np.fft.fft(y)
        freqs = np.fft.fftfreq(len(fft), 1 / sr)
        magnitude = np.abs(fft)
        dominant_freq_idx = np.argmax(magnitude[: len(magnitude) // 2])
        dominant_frequency = float(abs(freqs[dominant_freq_idx]))

        # Frequency range containing 90% of energy
        energy_cumsum = np.cumsum(magnitude[: len(magnitude) // 2])
        total_energy = energy_cumsum[-1]
        freq_range_low = freqs[np.where(energy_cumsum >= 0.05 * total_energy)[0][0]]
        freq_range_high = freqs[np.where(energy_cumsum >= 0.95 * total_energy)[0][0]]
        frequency_range = f"{freq_range_low:.1f}-{freq_range_high:.1f}"

        # Create/update noise analysis
        NoiseAnalysis.objects.update_or_create(
            noise_dataset=instance,
            defaults={
                "mean_db": mean_db,
                "max_db": max_db,
                "min_db": min_db,
                "std_db": std_db,
                "peak_count": peak_count,
                "peak_interval_mean": peak_interval_mean,
                "dominant_frequency": dominant_frequency,
                "frequency_range": frequency_range,
                "event_count": 0,  # Simplified
                "event_duration_mean": 0.0,  # Simplified
                "noise_type": "unknown",  # Simplified
                "analysis_notes": "Analysis performed in simple mode due to numba compatibility issues",
            },
        )

        logger.info(f"Successfully performed noise analysis for {instance.noise_id} (simple mode)")

    except Exception as e:
        logger.error(
            f"Error performing noise analysis on {audio_path}: {e}", exc_info=True
        )
        raise
