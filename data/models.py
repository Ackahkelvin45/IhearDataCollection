from django.db import models
import hashlib
# Create your models here.
from core.models import Region, Category, Community, Class, Microphone_Type,Time_Of_Day,SubClass
from authentication.models import CustomUser
from django.contrib.postgres.fields import ArrayField
import numpy as np
import json




class NoiseDataset(models.Model):
    name = models.CharField(
        max_length=255,
        null=True,
        help_text="Name of Data set, auto generated"
    )
    collector = models.ForeignKey(
        CustomUser,
        on_delete=models.PROTECT,
        null=True,
        help_text="The person collecting this data"
    )
    description = models.TextField(
        null=True,
        blank=True,
        help_text="Any additional notes about this recording"
    )
    region = models.ForeignKey(
        Region,
        on_delete=models.PROTECT,
        null=True,
        help_text="Select the region where recording was made (Ashanti, Central, Greater Accra, etc.)"
    )
    category = models.ForeignKey(
        Category,
        on_delete=models.PROTECT,
        null=True,
        help_text="Category of the data "
    )

    time_of_day = models.ForeignKey(
        Time_Of_Day,
        on_delete=models.PROTECT,
        null=True,
        help_text="Time of Day (Day, Night, etc.)"
    )
    community = models.ForeignKey(
        Community,
        on_delete=models.PROTECT,
        null=True,
        help_text="Specific community (Kotei, Adum, Ayeduase, etc.)"
    )
    class_name = models.ForeignKey(
        Class,
        on_delete=models.PROTECT,
        null=True,
        help_text="Class of the data"
    )
    subclass = models.ForeignKey(
        SubClass,
        on_delete=models.PROTECT,
        null=True,
        blank=True,
        help_text="Sub Class of the data"
    )

    community = models.ForeignKey(
        Community,
        on_delete=models.PROTECT,
        null=True,
        help_text="Environment Type (Urban, Rural, Coastal, Forested, etc.)"
    )





    microphone_type = models.ForeignKey(
        Microphone_Type,
        on_delete=models.PROTECT,
        null=True,
        help_text="Microphone Type (Omnidirectional, Directional, etc.) - Skip if using mobile phone"
    )
    audio = models.FileField(
        upload_to='files/',
        help_text="Upload audio file"
    )

    duration=models.CharField(
        max_length=255,
        null=True,
        help_text="Duration of the audio in seconds"
    )
    recording_date = models.DateTimeField(
        null=True,
        help_text="Date when recording was made"
    )
    recording_device = models.CharField(
        max_length=255,
        help_text="Recording Device (e.g., iPhone 16, Zoom H4n, etc.)"
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        help_text="When this record was last updated"
    )
    noise_id = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        unique=True,
        help_text="the noise_id,auto generated"
    )
    created_at = models.DateTimeField(
        auto_now_add=True
    )
    def __str__(self):
        return f'{self.name}-{self.noise_id}'
    


    def get_audio_hash(self):
        """
        Generate MD5 hash of the audio file content
        Returns None if no audio file exists
        """
        if not self.audio:
            return None
            
        # For small files (read entire file)
        if self.audio.size < 10 * 1024 * 1024:  # 10MB
            return hashlib.md5(self.audio.read()).hexdigest()
        else:
            # For large files (read in chunks)
            hash_md5 = hashlib.md5()
            for chunk in self.audio.chunks():
                hash_md5.update(chunk)
            return hash_md5.hexdigest()
    
    def __str__(self):
        return f'{self.name}-{self.noise_id}'
    







class AudioFeature(models.Model):
    """
    Stores extracted audio features for analysis and visualization
    """
    noise_dataset = models.OneToOneField(
        NoiseDataset,
        on_delete=models.CASCADE,
        related_name='audio_features'
    )
    
    # Time-domain features
    rms_energy = models.FloatField(help_text="Root Mean Square energy")
    zero_crossing_rate = models.FloatField(help_text="Zero Crossing Rate")
    duration = models.FloatField(help_text="Duration of the audio in seconds",null=True)
    sample_rate = models.IntegerField(null=True)
    num_samples = models.IntegerField(null=True)
    # Frequency-domain features
    spectral_centroid = models.FloatField(help_text="Spectral Centroid")
    spectral_bandwidth = models.FloatField(help_text="Spectral Bandwidth")
    spectral_rolloff = models.FloatField(help_text="Spectral Rolloff")
    spectral_flatness = models.FloatField(help_text="Spectral Flatness")
    
    # MFCCs (Mel-Frequency Cepstral Coefficients)
    mfccs = ArrayField(
        models.FloatField(),
        size=13,  # Typically 13 coefficients
        help_text="Mel-Frequency Cepstral Coefficients"
    )
    
    # Chroma features
    chroma_stft = ArrayField(
        models.FloatField(),
        size=12,
        help_text="Chroma features from Short-Time Fourier Transform"
    )
    
    # Mel-scaled spectrogram
    mel_spectrogram = models.TextField(
        help_text="JSON-serialized mel spectrogram data"
    )
    
    # Time series data for visualization
    waveform_data = models.TextField(
        help_text="JSON-serialized waveform data for visualization"
    )
    
    # Harmonic and percussive components
    harmonic_ratio = models.FloatField(help_text="Harmonic Ratio")
    percussive_ratio = models.FloatField(help_text="Percussive Ratio")
    
    def get_mel_spectrogram_array(self):
        """Convert stored mel spectrogram back to numpy array"""
        return np.array(json.loads(self.mel_spectrogram))
    
    def get_waveform_array(self):
        """Convert stored waveform data back to numpy array"""
        return np.array(json.loads(self.waveform_data))
    
    def __str__(self):
        return f"Audio Features for {self.noise_dataset.name}"


class NoiseAnalysis(models.Model):
    """
    Stores analysis results and statistics for noise data
    """
    noise_dataset = models.OneToOneField(
        NoiseDataset,
        on_delete=models.CASCADE,
        related_name='noise_analysis'
    )
    
    # Statistical features
    mean_db = models.FloatField(help_text="Mean decibel level")
    max_db = models.FloatField(help_text="Maximum decibel level")
    min_db = models.FloatField(help_text="Minimum decibel level")
    std_db = models.FloatField(help_text="Standard deviation of decibel levels")
    
    # Temporal features
    peak_count = models.IntegerField(help_text="Number of significant peaks")
    peak_interval_mean = models.FloatField(help_text="Mean interval between peaks")
    
    # Frequency analysis
    dominant_frequency = models.FloatField(help_text="Dominant frequency in Hz")
    frequency_range = models.CharField(
        max_length=100,
        help_text="Frequency range (low-high)"
    )
    
    # Psychoacoustic metrics
    loudness = models.FloatField(help_text="Perceived loudness")
    sharpness = models.FloatField(help_text="Perceived sharpness")
    roughness = models.FloatField(help_text="Perceived roughness")
    fluctuation_strength = models.FloatField(help_text="Fluctuation strength")
    
    # Event detection
    event_count = models.IntegerField(help_text="Number of distinct noise events")
    event_durations = ArrayField(
        models.FloatField(),
        help_text="Durations of detected events in seconds"
    )
    
    def __str__(self):
        return f"Analysis for {self.noise_dataset.name}"


class NoiseProfile(models.Model):
    """
    Aggregated noise profiles for different locations/categories
    """
    location = models.ForeignKey(
        Community,
        on_delete=models.CASCADE,
        related_name='noise_profiles'
    )
    category = models.ForeignKey(
        Category,
        on_delete=models.CASCADE,
        null=True,
        blank=True
    )
    class_name = models.ForeignKey(
        Class,
        on_delete=models.CASCADE,
        null=True,
        blank=True
    )
    time_of_day = models.ForeignKey(
        Time_Of_Day,
        on_delete=models.CASCADE,
        null=True,
        blank=True
    )
    
    # Aggregated statistics
    avg_db = models.FloatField(help_text="Average decibel level")
    max_db = models.FloatField(help_text="Maximum decibel level")
    min_db = models.FloatField(help_text="Minimum decibel level")
    
    # Frequency profile
    freq_profile = models.TextField(
        help_text="JSON-serialized frequency profile"
    )
    
    # Temporal patterns
    temporal_pattern = models.TextField(
        help_text="JSON-serialized temporal pattern"
    )
    
    # Common features
    common_features = models.TextField(
        help_text="JSON-serialized common audio features"
    )
    
    sample_count = models.IntegerField(
        help_text="Number of samples in this profile"
    )
    
    last_updated = models.DateTimeField(auto_now=True)
    
    class Meta:
        unique_together = ('location', 'category', 'class_name', 'time_of_day')
    
    def get_freq_profile(self):
        return json.loads(self.freq_profile)
    
    def get_temporal_pattern(self):
        return json.loads(self.temporal_pattern)
    
    def get_common_features(self):
        return json.loads(self.common_features)
    
    def __str__(self):
        name_parts = [str(self.location)]
        if self.category:
            name_parts.append(str(self.category))
        if self.class_name:
            name_parts.append(str(self.class_name))
        if self.time_of_day:
            name_parts.append(str(self.time_of_day))
        return " - ".join(name_parts)


class VisualizationPreset(models.Model):
    """
    Predefined visualization configurations for common chart types
    """
    name = models.CharField(max_length=255)
    description = models.TextField()
    
    # Chart configuration
    chart_type = models.CharField(
        max_length=50,
        choices=[
            ('waveform', 'Waveform'),
            ('spectrogram', 'Spectrogram'),
            ('spectrum', 'Frequency Spectrum'),
            ('mfcc', 'MFCCs'),
            ('chroma', 'Chroma Features'),
            ('db_trend', 'Decibel Trend'),
            ('time_analysis', 'Time Analysis'),
            ('feature_heatmap', 'Feature Heatmap'),
        ]
    )
    
    config = models.JSONField(help_text="Chart configuration in JSON format")
    
    # Accessibility settings
    high_contrast = models.BooleanField(default=False)
    alt_text_template = models.TextField(
        help_text="Template for generating alt text"
    )
    
    def __str__(self):
        return f"{self.name} ({self.chart_type})"
    