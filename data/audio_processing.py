"""
Alternative audio processing module that avoids librosa's Numba-compiled functions.
This module provides basic audio feature extraction using only NumPy and SciPy.
"""

import os
import numpy as np
import tempfile
import logging
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq
import soundfile as sf
import audioread
from typing import Optional, Tuple, List

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


def load_audio_file(audio_path: str, ext: str) -> Tuple[np.ndarray, int]:
    """Load audio file using soundfile or audioread, avoiding librosa"""
    try:
        # Try with soundfile first for supported formats
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
                logger.warning(f"Soundfile failed for {audio_path}: {sf_error}")

        # Try with audioread for other formats
        try:
            with audioread.audio_open(audio_path) as audio_file:
                sr = audio_file.samplerate
                channels = audio_file.channels
                duration = audio_file.duration

                # Read all samples
                samples = []
                for buf in audio_file:
                    samples.append(buf)

                # Convert to numpy array
                audio_data = b"".join(samples)
                if audio_file.samplerate == 44100:
                    # 16-bit audio
                    y = (
                        np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                        / 32768.0
                    )
                else:
                    # Assume 32-bit float
                    y = np.frombuffer(audio_data, dtype=np.float32)

                # Convert to mono if stereo
                if channels == 2:
                    y = y.reshape(-1, 2).mean(axis=1)

                logger.info(f"Loaded {audio_path} with audioread")
                return y, sr

        except Exception as ar_error:
            logger.warning(f"Audioread failed for {audio_path}: {ar_error}")

        raise ValueError(
            f"Could not load audio file {audio_path} with any available backend"
        )

    except Exception as e:
        logger.error(f"Error loading audio file {audio_path}: {e}")
        raise


def extract_basic_features(y: np.ndarray, sr: int) -> dict:
    """Extract basic audio features using only NumPy and SciPy"""
    try:
        # Ensure mono
        if len(y.shape) > 1:
            y = y.mean(axis=1)

        # Duration
        duration = len(y) / sr

        # RMS Energy
        rms_energy = np.sqrt(np.mean(y**2))

        # Zero Crossing Rate
        zero_crossings = np.sum(np.diff(np.signbit(y))) / (2 * len(y))

        # Spectral features using FFT
        fft_vals = fft(y)
        freqs = fftfreq(len(y), 1 / sr)

        # Only positive frequencies
        pos_freqs = freqs[: len(freqs) // 2]
        pos_fft = np.abs(fft_vals[: len(fft_vals) // 2])

        # Spectral centroid
        spectral_centroid = (
            np.sum(pos_freqs * pos_fft) / np.sum(pos_fft) if np.sum(pos_fft) > 0 else 0
        )

        # Spectral bandwidth
        spectral_bandwidth = (
            np.sqrt(
                np.sum(((pos_freqs - spectral_centroid) ** 2) * pos_fft)
                / np.sum(pos_fft)
            )
            if np.sum(pos_fft) > 0
            else 0
        )

        # Spectral rolloff (85th percentile)
        cumulative_energy = np.cumsum(pos_fft)
        rolloff_threshold = 0.85 * cumulative_energy[-1]
        rolloff_idx = np.where(cumulative_energy >= rolloff_threshold)[0]
        spectral_rolloff = pos_freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0

        # Spectral flatness
        geometric_mean = np.exp(np.mean(np.log(pos_fft + 1e-10)))
        arithmetic_mean = np.mean(pos_fft)
        spectral_flatness = (
            geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0
        )

        # Peak detection
        peaks, _ = scipy_signal.find_peaks(np.abs(y), height=np.std(y) * 2)
        peak_count = len(peaks)

        # Peak intervals
        if len(peaks) > 1:
            peak_intervals = np.diff(peaks) / sr
            peak_interval_mean = np.mean(peak_intervals)
        else:
            peak_interval_mean = 0.0

        # Dominant frequency
        dominant_freq_idx = np.argmax(pos_fft)
        dominant_frequency = pos_freqs[dominant_freq_idx]

        # Frequency range
        frequency_range = pos_freqs[-1] - pos_freqs[0]

        # Basic MFCC-like features (simplified)
        # Use mel-frequency cepstral coefficients approximation
        mel_basis = create_mel_filterbank(sr, len(y), n_mels=13)
        # Ensure dimensions match
        if mel_basis.shape[1] != len(pos_fft):
            # Pad or truncate to match
            if mel_basis.shape[1] > len(pos_fft):
                mel_basis = mel_basis[:, : len(pos_fft)]
            else:
                # Pad pos_fft with zeros
                padded_fft = np.zeros(mel_basis.shape[1])
                padded_fft[: len(pos_fft)] = pos_fft
                pos_fft = padded_fft

        mel_spectrum = np.dot(mel_basis, pos_fft)
        mfccs = np.log(mel_spectrum + 1e-10)
        mfccs_mean = mfccs.tolist()  # Already 1D

        # Chroma features (simplified)
        chroma = create_chroma_features(pos_freqs, pos_fft, sr)
        chroma_mean = chroma.flatten().tolist()

        # Mel spectrogram (simplified)
        mel_spec = mel_spectrum  # Already computed above
        mel_spec_db = 20 * np.log10(mel_spec + 1e-10)

        # Waveform data (downsampled for visualization)
        waveform_data = y[:: max(1, len(y) // 1000)].tolist()

        # Harmonic/Percussive separation (simplified)
        # Use high-pass and low-pass filters
        b_hp, a_hp = scipy_signal.butter(4, 0.1, btype="high")
        b_lp, a_lp = scipy_signal.butter(4, 0.1, btype="low")

        y_harmonic = scipy_signal.filtfilt(b_lp, a_lp, y)
        y_percussive = scipy_signal.filtfilt(b_hp, a_hp, y)

        harmonic_ratio = float(np.sum(y_harmonic**2) / np.sum(y**2))
        percussive_ratio = float(np.sum(y_percussive**2) / np.sum(y**2))

        return {
            "sample_rate": sr,
            "num_samples": len(y),
            "duration": duration,
            "rms_energy": float(rms_energy),
            "zero_crossing_rate": float(zero_crossings),
            "spectral_centroid": float(spectral_centroid),
            "spectral_bandwidth": float(spectral_bandwidth),
            "spectral_rolloff": float(spectral_rolloff),
            "spectral_flatness": float(spectral_flatness),
            "mfccs": mfccs_mean,
            "chroma_stft": chroma_mean,
            "mel_spectrogram": mel_spec_db.tolist(),
            "waveform_data": waveform_data,
            "harmonic_ratio": harmonic_ratio,
            "percussive_ratio": percussive_ratio,
            "peak_count": peak_count,
            "peak_interval_mean": peak_interval_mean,
            "dominant_frequency": dominant_frequency,
            "frequency_range": frequency_range,
        }

    except Exception as e:
        logger.error(f"Error extracting basic features: {e}")
        raise


def create_mel_filterbank(sr: int, n_fft: int, n_mels: int = 13) -> np.ndarray:
    """Create a simplified mel filterbank"""
    # Simplified mel filterbank creation
    f_min = 0
    f_max = sr // 2

    # Create mel frequencies
    mel_min = 2595 * np.log10(1 + f_min / 700)
    mel_max = 2595 * np.log10(1 + f_max / 700)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)

    # Convert back to Hz
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)

    # Create filterbank - ensure correct dimensions
    fft_freqs = np.linspace(0, f_max, n_fft // 2 + 1)
    filterbank = np.zeros((n_mels, len(fft_freqs)))

    for i in range(n_mels):
        left = hz_points[i]
        center = hz_points[i + 1]
        right = hz_points[i + 2]

        # Triangular filter
        for j, freq in enumerate(fft_freqs):
            if left <= freq <= center:
                filterbank[i, j] = (freq - left) / (center - left)
            elif center <= freq <= right:
                filterbank[i, j] = (right - freq) / (right - center)

    return filterbank


def create_chroma_features(
    freqs: np.ndarray, magnitudes: np.ndarray, sr: int
) -> np.ndarray:
    """Create simplified chroma features"""
    # Define chroma frequencies (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
    chroma_freqs = np.array(
        [
            261.63,
            277.18,
            293.66,
            311.13,
            329.63,
            349.23,
            369.99,
            392.00,
            415.30,
            440.00,
            466.16,
            493.88,
        ]
    )

    chroma = np.zeros(12)

    for i, chroma_freq in enumerate(chroma_freqs):
        # Find frequencies close to this chroma frequency
        for octave in range(1, 8):  # 7 octaves
            target_freq = chroma_freq * octave
            if target_freq > sr // 2:
                break

            # Find closest frequency in our spectrum
            idx = np.argmin(np.abs(freqs - target_freq))
            if idx < len(magnitudes):
                chroma[i] += magnitudes[idx]

    return chroma.reshape(-1, 1)


def extract_noise_analysis(y: np.ndarray, sr: int) -> dict:
    """Extract noise analysis features"""
    try:
        # Ensure mono
        if len(y.shape) > 1:
            y = y.mean(axis=1)

        # Convert to dB
        y_db = 20 * np.log10(np.abs(y) + 1e-10)

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
        fft_vals = fft(y)
        freqs = fftfreq(len(y), 1 / sr)

        # Only positive frequencies
        pos_freqs = freqs[: len(freqs) // 2]
        pos_fft = np.abs(fft_vals[: len(fft_vals) // 2])

        # Dominant frequency
        dominant_freq_idx = np.argmax(pos_fft)
        dominant_frequency = pos_freqs[dominant_freq_idx]

        # Frequency range
        frequency_range = pos_freqs[-1] - pos_freqs[0]

        # Spectral centroid
        spectral_centroid = (
            np.sum(pos_freqs * pos_fft) / np.sum(pos_fft) if np.sum(pos_fft) > 0 else 0
        )

        # Spectral bandwidth
        spectral_bandwidth = (
            np.sqrt(
                np.sum(((pos_freqs - spectral_centroid) ** 2) * pos_fft)
                / np.sum(pos_fft)
            )
            if np.sum(pos_fft) > 0
            else 0
        )

        # Spectral rolloff
        cumulative_energy = np.cumsum(pos_fft)
        rolloff_threshold = 0.85 * cumulative_energy[-1]
        rolloff_idx = np.where(cumulative_energy >= rolloff_threshold)[0]
        spectral_rolloff = pos_freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0

        return {
            "mean_db": mean_db,
            "max_db": max_db,
            "min_db": min_db,
            "std_db": std_db,
            "peak_count": peak_count,
            "peak_interval_mean": peak_interval_mean,
            "dominant_frequency": float(dominant_frequency),
            "frequency_range": float(frequency_range),
            "spectral_centroid": float(spectral_centroid),
            "spectral_bandwidth": float(spectral_bandwidth),
            "spectral_rolloff": float(spectral_rolloff),
        }

    except Exception as e:
        logger.error(f"Error extracting noise analysis: {e}")
        raise


def process_audio_file_alternative(audio_path: str, ext: str) -> Tuple[dict, dict]:
    """Process audio file using alternative methods (no librosa)"""
    try:
        # Load audio
        y, sr = load_audio_file(audio_path, ext)

        # Extract features
        audio_features = extract_basic_features(y, sr)
        noise_analysis = extract_noise_analysis(y, sr)

        return audio_features, noise_analysis

    except Exception as e:
        logger.error(f"Error in alternative audio processing: {e}")
        raise
