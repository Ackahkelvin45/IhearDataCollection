import os
import numpy as np
import tempfile
import logging
from scipy import signal as scipy_signal
import soundfile as sf
import audioread
from typing import Optional
from .models import NoiseDataset, AudioFeature, NoiseAnalysis, VisualizationPreset, CleanSpeechDataset, CleanSpeechAudioFeature, CleanSpeechAnalysis
import uuid
from datetime import datetime
from django.utils import timezone
from .audio_processing import (
    get_file_extension,
    is_supported_extension,
    load_audio_file,
    process_audio_file_alternative,
)

logger = logging.getLogger(__name__)

# Removed old audio format functions - now using alternative processing module


# Removed old load_audio_file function - now using alternative processing


def process_audio_file(instance: NoiseDataset):
    """Full audio processing pipeline using alternative methods (no librosa)"""
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

            # Use alternative processing (no librosa)
            audio_features, noise_analysis = process_audio_file_alternative(
                audio_path, file_ext
            )

            # Create/update audio features
            AudioFeature.objects.update_or_create(
                noise_dataset=instance,
                defaults=audio_features,
            )

            # Create/update noise analysis
            NoiseAnalysis.objects.update_or_create(
                noise_dataset=instance,
                defaults=noise_analysis,
            )

            # Create visualization presets
            create_visualization_presets(instance)

            logger.info(f"Successfully processed audio file for {instance.noise_id}")

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


def create_clean_speech_visualization_presets(instance):
    """Create standard visualization presets for clean speech audio"""
    try:
        features = instance.audio_features
        analysis = instance.analysis.first() if instance.analysis.exists() else None

        # Waveform preset
        VisualizationPreset.objects.get_or_create(
            clean_speech_dataset=instance,
            chart_type="waveform",
            defaults={
                "name": f"Waveform - {instance.name}",
                "description": "Waveform visualization of the clean speech audio",
                "config": {
                    "title": "Clean Speech Waveform",
                    "x_label": "Time (s)",
                    "y_label": "Amplitude",
                    "color": "#10b981",
                    "data": getattr(features, 'waveform_data', None),
                    "duration": getattr(features, 'duration', None),
                },
                "alt_text_template": f"Waveform of clean speech {instance.name} showing amplitude over time",
            },
        )

        # Spectrogram preset
        VisualizationPreset.objects.get_or_create(
            clean_speech_dataset=instance,
            chart_type="spectrogram",
            defaults={
                "name": f"Spectrogram - {instance.name}",
                "description": "Spectrogram visualization of the clean speech audio",
                "config": {
                    "title": "Clean Speech Spectrogram",
                    "x_label": "Time (s)",
                    "y_label": "Frequency (Hz)",
                    "cmap": "plasma",
                    "data": getattr(features, 'mel_spectrogram', None),
                    "sample_rate": getattr(features, 'sample_rate', None),
                    "duration": getattr(features, 'duration', None),
                },
                "alt_text_template": f"Spectrogram of clean speech {instance.name} showing frequency content over time",
            },
        )

        # MFCC preset
        VisualizationPreset.objects.get_or_create(
            clean_speech_dataset=instance,
            chart_type="mfcc",
            defaults={
                "name": f"MFCC - {instance.name}",
                "description": "Mel-frequency cepstral coefficients of the clean speech",
                "config": {
                    "title": "Clean Speech MFCC",
                    "x_label": "Time frames",
                    "y_label": "MFCC coefficients",
                    "cmap": "coolwarm",
                    "data": getattr(features, 'mfcc', None),
                    "mfcc_mean": getattr(features, 'mfcc_mean', None),
                    "mfcc_std": getattr(features, 'mfcc_std', None),
                },
                "alt_text_template": f"MFCC visualization of clean speech {instance.name}",
            },
        )

        # Speech quality analysis preset
        if analysis:
            VisualizationPreset.objects.get_or_create(
                clean_speech_dataset=instance,
                chart_type="time_analysis",
                defaults={
                    "name": f"Quality Analysis - {instance.name}",
                    "description": "Speech quality metrics and analysis",
                    "config": {
                        "title": "Speech Quality Analysis",
                        "metrics": {
                            "snr": getattr(analysis, 'snr', None),
                            "speech_rate": getattr(analysis, 'speech_rate', None),
                            "quality_score": getattr(analysis, 'overall_quality_score', None),
                            "clarity": getattr(analysis, 'articulation_clarity', None),
                        },
                        "recommendations": getattr(analysis, 'recommendations', ''),
                    },
                    "alt_text_template": f"Quality analysis of clean speech {instance.name}",
                },
            )

    except Exception as e:
        logger.error(f"Error creating visualization presets for clean speech {instance.id}: {e}")


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


def generate_clean_speech_dataset_name(clean_speech_dataset):
    parts = []

    if clean_speech_dataset.category:
        parts.append(str(clean_speech_dataset.category.name))

    if clean_speech_dataset.class_name:
        parts.append(str(clean_speech_dataset.class_name.name))

    # Join with underscores and add timestamp
    base_name = "_".join(parts) if parts else "CLEAN_SPEECH"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return f"{base_name}_{timestamp}"


def generate_clean_speech_id():
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_chars = uuid.uuid4().hex[:4].upper()
    return f"CLS-{timestamp}-{random_chars}"


def process_clean_speech_file(instance):
    """Process clean speech audio file and extract features"""
    if not instance.audio:
        logger.warning(f"No audio file found for CleanSpeechDataset {instance.id}")
        return

    try:
        # Generate clean_speech_id if not provided
        if not instance.clean_speech_id:
            audio_hash = instance.get_audio_hash()
            instance.clean_speech_id = (
                f"CLS_{audio_hash[:8]}" if audio_hash else f"CLS_{instance.id}"
            )
            instance.save(update_fields=["clean_speech_id"])

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
            f"Processing clean speech audio file: {audio_path}, "
            f"size: {os.path.getsize(audio_path)} bytes, "
            f"ext: {file_ext}"
        )

        try:
            # Verify file exists and has content
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                raise ValueError("Temporary audio file is empty or doesn't exist")

            # Use alternative processing (no librosa)
            audio_features, noise_analysis = process_audio_file_alternative(
                audio_path, file_ext
            )

            # Create/update clean speech audio features
            CleanSpeechAudioFeature.objects.update_or_create(
                clean_speech_dataset=instance,
                defaults=audio_features,
            )

            # Create/update clean speech analysis
            CleanSpeechAnalysis.objects.update_or_create(
                clean_speech_dataset=instance,
                analysis_type='speech_quality',
                defaults={
                    'snr': noise_analysis.get('snr', 0),
                    'overall_quality_score': noise_analysis.get('overall_quality_score', 50),
                    'analysis_data': noise_analysis,
                },
            )

            # Create visualization presets for clean speech
            create_clean_speech_visualization_presets(instance)

            logger.info(f"Successfully processed clean speech audio file for {instance.clean_speech_id}")

        except Exception as processing_error:
            logger.error(
                f"Error processing clean speech audio file {audio_path}: {processing_error}",
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
        logger.error(f"Error in clean speech audio processing pipeline: {e}", exc_info=True)
        raise


def safe_process_audio_file(instance: NoiseDataset):
    """Safe audio processing function with Numba JIT already disabled at module level"""
    try:
        # process audio
        process_audio_file(instance)
        return True

    except Exception as processing_error:
        logger.error(
            f"Error in audio processing for {instance.noise_id}: {processing_error}"
        )

        # error loggin
        error_str = str(processing_error)
        if "Could not load audio file" in error_str:
            if ".m4a" in error_str:
                logger.error(
                    "M4A file loading failed. This may be due to missing audio codecs in the Docker container."
                )
                logger.error(
                    "Solution: Ensure ffmpeg is installed in the Docker container."
                )
            elif ".mp3" in error_str:
                logger.error(
                    "MP3 file loading failed. This may be due to missing audio codecs."
                )
            else:
                logger.error(
                    "Audio file loading failed. Check if the file is corrupted or in an unsupported format."
                )

        return False
