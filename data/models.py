from django.db import models
import hashlib
from core.models import (
    Region,
    Category,
    Community,
    Class,
    Microphone_Type,
    Time_Of_Day,
    SubClass,
)
from authentication.models import CustomUser
from django.contrib.postgres.fields import ArrayField


class NoiseDataset(models.Model):
    name = models.CharField(
        max_length=255, null=True, help_text="Name of Data set, auto generated"
    )
    collector = models.ForeignKey(
        CustomUser,
        on_delete=models.PROTECT,
        null=True,
        help_text="The person collecting this data",
    )
    description = models.TextField(
        null=True, blank=True, help_text="Any additional notes about this recording"
    )
    region = models.ForeignKey(
        Region,
        on_delete=models.PROTECT,
        null=True,
        help_text="Select the region where recording was made",
    )
    category = models.ForeignKey(
        Category, on_delete=models.PROTECT, null=True, help_text="Category of the data"
    )
    time_of_day = models.ForeignKey(
        Time_Of_Day,
        on_delete=models.PROTECT,
        null=True,
        help_text="Time of Day (Day, Night, etc.)",
    )
    community = models.ForeignKey(
        Community,
        on_delete=models.PROTECT,
        null=True,
        help_text="Specific community where recording was made",
    )
    class_name = models.ForeignKey(
        Class, on_delete=models.PROTECT, null=True, help_text="Class of the data"
    )
    subclass = models.ForeignKey(
        SubClass,
        on_delete=models.PROTECT,
        null=True,
        blank=True,
        help_text="Sub Class of the data",
    )
    microphone_type = models.ForeignKey(
        Microphone_Type,
        on_delete=models.PROTECT,
        null=True,
        help_text="Microphone Type",
    )
    audio = models.FileField(upload_to="files/", help_text="Upload audio file")
    recording_date = models.DateTimeField(
        null=True, help_text="Date when recording was made"
    )
    recording_device = models.CharField(
        max_length=255, help_text="Recording Device (e.g., iPhone 16, Zoom H4n, etc.)"
    )
    updated_at = models.DateTimeField(
        auto_now=True, help_text="When this record was last updated"
    )
    noise_id = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        unique=True,
        help_text="the noise_id,auto generated",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    def get_audio_hash(self):
        """Generate MD5 hash of the audio file content"""
        if not self.audio:
            return None

        if self.audio.size < 10 * 1024 * 1024:  # 10MB
            return hashlib.md5(self.audio.read()).hexdigest()
        else:
            hash_md5 = hashlib.md5()
            for chunk in self.audio.chunks():
                hash_md5.update(chunk)
            return hash_md5.hexdigest()

    def __str__(self):
        return f"{self.name}-{self.noise_id}"


class AudioFeature(models.Model):
    """Stores extracted audio features for analysis and visualization"""

    noise_dataset = models.OneToOneField(
        NoiseDataset, on_delete=models.CASCADE, related_name="audio_features"
    )

    # Basic properties
    sample_rate = models.IntegerField(null=True)
    num_samples = models.IntegerField(null=True)
    duration = models.FloatField(
        help_text="Duration of the audio in seconds", null=True
    )

    # Time-domain features
    rms_energy = models.FloatField(help_text="Root Mean Square energy", null=True)
    zero_crossing_rate = models.FloatField(help_text="Zero Crossing Rate", null=True)

    # Frequency-domain features
    spectral_centroid = models.FloatField(help_text="Spectral Centroid", null=True)
    spectral_bandwidth = models.FloatField(help_text="Spectral Bandwidth", null=True)
    spectral_rolloff = models.FloatField(help_text="Spectral Rolloff", null=True)
    spectral_flatness = models.FloatField(help_text="Spectral Flatness", null=True)

    # Advanced features
    mfccs = ArrayField(
        models.FloatField(),
        size=13,
        help_text="Mel-Frequency Cepstral Coefficients",
        null=True,
    )
    chroma_stft = ArrayField(
        models.FloatField(),
        size=12,
        help_text="Chroma features from Short-Time Fourier Transform",
        null=True,
    )

    # Visualization data
    mel_spectrogram = models.JSONField(help_text="Mel spectrogram data", null=True)
    waveform_data = models.JSONField(
        help_text="Waveform data for visualization", null=True
    )

    # Harmonic/percussive components
    harmonic_ratio = models.FloatField(help_text="Harmonic Ratio", null=True)
    percussive_ratio = models.FloatField(help_text="Percussive Ratio", null=True)

    def __str__(self):
        return f"Audio Features for {self.noise_dataset.name}"


class NoiseAnalysis(models.Model):
    """Stores analysis results and statistics for noise data"""

    noise_dataset = models.OneToOneField(
        NoiseDataset, on_delete=models.CASCADE, related_name="noise_analysis"
    )

    # Statistical features
    mean_db = models.FloatField(help_text="Mean decibel level", null=True)
    max_db = models.FloatField(help_text="Maximum decibel level", null=True)
    min_db = models.FloatField(help_text="Minimum decibel level", null=True)
    std_db = models.FloatField(
        help_text="Standard deviation of decibel levels", null=True
    )

    # Temporal features
    peak_count = models.IntegerField(help_text="Number of significant peaks", null=True)
    peak_interval_mean = models.FloatField(
        help_text="Mean interval between peaks", null=True
    )

    # Frequency analysis
    dominant_frequency = models.FloatField(
        help_text="Dominant frequency in Hz", null=True
    )
    frequency_range = models.CharField(
        max_length=100, help_text="Frequency range (low-high)", null=True
    )

    # Event detection
    event_count = models.IntegerField(help_text="Number of distinct noise events")
    event_durations = ArrayField(
        models.FloatField(),
        help_text="Durations of detected events in seconds",
        null=True,
    )

    def __str__(self):
        return f"Analysis for {self.noise_dataset.name}"


class VisualizationPreset(models.Model):
    """Predefined visualization configurations for common chart types"""

    noise_dataset = models.ForeignKey(
        NoiseDataset,
        on_delete=models.CASCADE,
        related_name="visualization_presets",
        null=True,
        blank=True,
    )
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)

    # Chart configuration
    chart_type = models.CharField(
        max_length=50,
        choices=[
            ("waveform", "Waveform"),
            ("spectrogram", "Spectrogram"),
            ("spectrum", "Frequency Spectrum"),
            ("mfcc", "MFCCs"),
            ("chroma", "Chroma Features"),
            ("db_trend", "Decibel Trend"),
            ("time_analysis", "Time Analysis"),
        ],
    )

    config = models.JSONField(help_text="Chart configuration in JSON format")

    # Accessibility settings
    high_contrast = models.BooleanField(default=False)
    alt_text_template = models.TextField(
        help_text="Template for generating alt text", blank=True
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return f"{self.name} ({self.chart_type})"
