import librosa
import numpy as np
import json
from scipy import signal
import soundfile as sf
from io import BytesIO
import traceback

def extract_audio_features(audio_file, min_duration=0.1):
    """
    Extract all audio features with comprehensive error handling
    min_duration: minimum audio length in seconds to process (default 0.1s)
    Returns: dict of features if successful, None if failed
    """
    try:
        # Load audio file with multiple fallbacks
        try:
            with audio_file.open('rb') as f:
                audio_data, sr = sf.read(BytesIO(f.read()))
        except Exception as e:
            print(f"Soundfile read failed, trying librosa: {str(e)}")
            audio_data, sr = librosa.load(audio_file.path, sr=None, mono=True)
        
        # Validate audio data
        if audio_data.size == 0:
            return None
            
        if audio_data.ndim > 0:
            audio_data = librosa.to_mono(audio_data)
        
        duration = librosa.get_duration(y=audio_data, sr=sr)
        
        if duration < min_duration:
            return None
        
        # Dynamic FFT size based on audio length
        n_fft = min(2048, len(audio_data))
        hop_length = n_fft // 4
        if n_fft < 0:
            return None
        
        features = {
            'duration': float(duration),
            'sample_rate': int(sr),
            'num_samples': len(audio_data),
        }
        
        # Time-domain features
        features.update(extract_time_domain_features(audio_data))
        
        # Frequency-domain features with dynamic FFT
        features.update(extract_frequency_domain_features(audio_data, sr, n_fft, hop_length))
        
        # Only extract advanced features if audio is long enough
        if duration > 0.1:  # 500ms minimum for these features
            try:
                features.update(extract_mfcc_features(audio_data, sr, n_fft, hop_length))
            except:
                features['mfccs'] = []
            
            try:
                features.update(extract_chroma_features(audio_data, sr, n_fft, hop_length))
            except:
                features['chroma_stft'] = []
            
            try:
                features.update(extract_mel_spectrogram(audio_data, sr, n_fft, hop_length))
            except:
                features['mel_spectrogram'] = '[]'
            
            try:
                features.update(extract_psychoacoustic_features(audio_data, sr))
            except:
                features.update({
                    'harmonic_ratio': 0,
                    'percussive_ratio': 0
                })
        
        # Serialize waveform data last (as it's large)
        try:
            features['waveform_data'] = safe_json_dumps(audio_data.tolist())
        except:
            features['waveform_data'] = '[]'
        
        print(features)
        
        return features
        
    except Exception as e:
        print(f"Audio processing failed: {str(e)}")
        return None

def safe_json_dumps(data):
    """Safe JSON serialization with numpy support"""
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    return json.dumps(data, cls=NumpyEncoder)

def extract_time_domain_features(audio_data):
    """Extract time-domain features with error handling"""
    features = {}
    
    try:
        features['rms_energy'] = float(np.sqrt(np.mean(np.square(audio_data))))
    except:
        features['rms_energy'] = 0.0
    
    try:
        features['zero_crossing_rate'] = float(librosa.feature.zero_crossing_rate(audio_data)[0, 0])
    except:
        features['zero_crossing_rate'] = 0.0
    
    return features

def extract_frequency_domain_features(audio_data, sr, n_fft=2048, hop_length=512):
    """Extract frequency-domain features with dynamic FFT size"""
    features = {}
    
    try:
        stft = np.abs(librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length))
        
        try:
            features['spectral_centroid'] = float(np.mean(librosa.feature.spectral_centroid(S=stft, sr=sr)))
        except:
            features['spectral_centroid'] = 0.0
            
        try:
            features['spectral_bandwidth'] = float(np.mean(librosa.feature.spectral_bandwidth(S=stft, sr=sr)))
        except:
            features['spectral_bandwidth'] = 0.0
            
        try:
            features['spectral_rolloff'] = float(np.mean(librosa.feature.spectral_rolloff(S=stft, sr=sr)))
        except:
            features['spectral_rolloff'] = 0.0
            
        try:
            features['spectral_flatness'] = float(np.mean(librosa.feature.spectral_flatness(y=audio_data)))
        except:
            features['spectral_flatness'] = 0.0
            
    except Exception as e:
        # Set all frequency domain features to 0 if extraction fails
        features.update({
            'spectral_centroid': 0.0,
            'spectral_bandwidth': 0.0,
            'spectral_rolloff': 0.0,
            'spectral_flatness': 0.0
        })
    
    return features

def extract_mfcc_features(audio_data, sr, n_fft=2048, hop_length=512, n_mfcc=13):
    """Extract MFCC features with dynamic FFT"""
    try:
        mfccs = librosa.feature.mfcc(
            y=audio_data, 
            sr=sr, 
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length
        )
        return {'mfccs': np.mean(mfccs, axis=1).tolist()}
    except:
        return {'mfccs': [0]*n_mfcc}

def extract_chroma_features(audio_data, sr, n_fft=2048, hop_length=512):
    """Extract chroma features with dynamic FFT"""
    try:
        chroma = librosa.feature.chroma_stft(
            y=audio_data, 
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length
        )
        return {'chroma_stft': np.mean(chroma, axis=1).tolist()}
    except:
        return {'chroma_stft': [0]*12}

def extract_mel_spectrogram(audio_data, sr, n_fft=2048, hop_length=512, n_mels=128):
    """Extract mel spectrogram with dynamic FFT"""
    try:
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data, 
            sr=sr, 
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return {'mel_spectrogram': mel_spec_db.tolist()}
    except:
        return {'mel_spectrogram': []}

def extract_psychoacoustic_features(audio_data, sr):
    """Extract harmonic/percussive components"""
    try:
        y_harmonic, y_percussive = librosa.effects.hpss(audio_data)
        harmonic_ratio = np.sum(y_harmonic**2) / (np.sum(y_harmonic**2) + np.sum(y_percussive**2))
        return {
            'harmonic_ratio': float(harmonic_ratio),
            'percussive_ratio': float(1 - harmonic_ratio)
        }
    except:
        return {
            'harmonic_ratio': 0.0,
            'percussive_ratio': 0.0
        }

def analyze_noise_characteristics(audio_data, sr):
    """Robust noise analysis with error handling"""
    analysis = {}
    
    try:
        # Calculate dB levels
        rms = np.sqrt(np.mean(np.square(audio_data)))
        db = 20 * np.log10(max(1e-10, rms) / (2e-5))  # Avoid log(0)
        
        analysis.update({
            'mean_db': float(db),
            'max_db': float(db + 10),  # Simplified
            'min_db': float(db - 10)   # Simplified
        })
        
        # Peak detection
        try:
            peaks, _ = signal.find_peaks(audio_data, height=0.5*np.max(audio_data))
            analysis['peak_count'] = int(len(peaks))
            
            if len(peaks) > 1:
                peak_intervals = np.diff(peaks) / sr
                analysis['peak_interval_mean'] = float(np.mean(peak_intervals))
            else:
                analysis['peak_interval_mean'] = 0.0
        except:
            analysis.update({
                'peak_count': 0,
                'peak_interval_mean': 0.0
            })
        
        # Frequency analysis
        try:
            n_fft = min(2048, len(audio_data))
            stft = np.abs(librosa.stft(audio_data, n_fft=n_fft))
            frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            dominant_freq = frequencies[np.argmax(np.mean(stft, axis=1))]
            
            analysis.update({
                'dominant_frequency': float(dominant_freq),
                'frequency_range': {
                    'min': float(frequencies[0]),
                    'max': float(frequencies[-1])
                }
            })
        except:
            analysis.update({
                'dominant_frequency': 0.0,
                'frequency_range': {'min': 0, 'max': 0}
            })
        
        # Psychoacoustic metrics
        try:
            analysis.update({
                'loudness': float(db),
                'sharpness': float(np.mean(librosa.feature.spectral_centroid(S=stft, sr=sr))),
                'roughness': 0.5,  # Placeholder
                'fluctuation_strength': 0.3  # Placeholder
            })
        except:
            analysis.update({
                'loudness': float(db),
                'sharpness': 0.0,
                'roughness': 0.0,
                'fluctuation_strength': 0.0
            })
        
        # Event detection
        analysis.update({
            'event_count': analysis.get('peak_count', 0),
            'event_durations': [0.1]*analysis.get('peak_count', 0)
        })
        
    except Exception as e:
        return None
    
    return analysis