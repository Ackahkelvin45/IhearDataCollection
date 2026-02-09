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
from django.core.exceptions import ValidationError
from dateutil.parser import parse


class Dataset(models.Model):
    DATASET_TYPES = [
        ("noise", "Noise"),
        ("clean_speech", "Clean Speech"),
        ("mixed", "Mixed (Noise and Clean)"),
        ("non_standard_speech", "Non-standard Speech"),
        ("animals", "Animals"),
    ]

    name = models.CharField(max_length=255, choices=DATASET_TYPES, unique=True)
    description = models.TextField(blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.get_name_display()


class NoiseDataset(models.Model):

    dataset_type = models.ForeignKey(Dataset, on_delete=models.CASCADE,null=True,blank=True)

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

    recording = models.ForeignKey('NoiseRecording', on_delete=models.CASCADE, null=True, blank=True, help_text="The recording that was made")
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
        """Generate MD5 hash of the audio file content safely.
        Returns None if file is missing or unreadable.
        """
        try:
            if not self.audio:
                print("[get_audio_hash] No audio attached for dataset:", self.pk)
                return None

            # Accessing size may trigger a storage HEAD call (e.g., S3). Guard it.
            try:
                file_size = self.audio.size
            except Exception as size_exc:
                print(
                    f"[get_audio_hash] Failed to read size for {self.audio.name}: {size_exc}"
                )
                return None

            if file_size < 10 * 1024 * 1024:  # 10MB
                try:
                    # Ensure pointer at start
                    if hasattr(self.audio, "seek"):
                        self.audio.seek(0)
                    content = self.audio.read()
                    if hasattr(self.audio, "seek"):
                        self.audio.seek(0)
                    return hashlib.md5(content).hexdigest()
                except Exception as read_exc:
                    print(
                        f"[get_audio_hash] Failed to read small file {self.audio.name}: {read_exc}"
                    )
                    return None
            else:
                hash_md5 = hashlib.md5()
                try:
                    if hasattr(self.audio, "seek"):
                        self.audio.seek(0)
                    for chunk in self.audio.chunks():
                        hash_md5.update(chunk)
                    if hasattr(self.audio, "seek"):
                        self.audio.seek(0)
                    return hash_md5.hexdigest()
                except Exception as chunk_exc:
                    print(
                        f"[get_audio_hash] Failed to stream chunks for {self.audio.name}: {chunk_exc}"
                    )
                    return None
        except Exception as exc:
            print(f"[get_audio_hash] Unexpected error for dataset {self.pk}: {exc}")
            return None

    def save(self, *args, **kwargs):

        # Ensure dataset type is always "noise"
        if not self.id:
            # create a Dataset of type noise if one doesn't exist
            noise_dataset, _ = Dataset.objects.get_or_create(
                name="noise", defaults={"description": "Noise dataset"}
            )
            self.dataset_type = noise_dataset
        else:
            self.dataset_type.name = "noise"
            self.dataset_type.save()
        super().save(*args, **kwargs)

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


class BulkAudioUpload(models.Model):
    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("processing", "Processing"),
        ("completed", "Completed"),
        ("failed", "Failed"),
        ("completed_with_errors", "Completed With Errors"),
    ]

    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    status = models.CharField(max_length=255, choices=STATUS_CHOICES, default="pending")
    metadata = models.JSONField(null=True)
    total_files = models.IntegerField(default=0)
    processed_files = models.IntegerField(default=0)
    failed_files = models.IntegerField(default=0)

    def __str__(self):
        return f"Bulk Upload {self.id} by {self.user.username}"

    def clean_metadata(self):
        required_fields = [
            "region_id",
            "category_id",
            "time_of_day_id",
            "community_id",
            "class_name_id",
            "recording_date",
            "recording_device",
        ]

        if not self.metadata:
            raise ValidationError("Metadata cannot be empty")

        for field in required_fields:
            if field not in self.metadata:
                raise ValidationError(f"Missing required metadata field: {field}")

        # Validate recording_date format using dateutil
        try:
            parse(self.metadata["recording_date"])
        except (ValueError, TypeError, KeyError):
            raise ValidationError(
                "Invalid recording_date format. Use ISO format (YYYY-MM-DD)"
            )


class BulkReprocessingTask(models.Model):
    """Track bulk reprocessing tasks for progress monitoring"""

    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("processing", "Processing"),
        ("completed", "Completed"),
        ("failed", "Failed"),
        ("completed_with_errors", "Completed With Errors"),
        ("cancelled", "Cancelled"),
    ]

    task_id = models.CharField(max_length=255, unique=True, help_text="Celery task ID")
    user = models.ForeignKey(
        CustomUser, on_delete=models.CASCADE, help_text="User who initiated the task"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    status = models.CharField(max_length=50, choices=STATUS_CHOICES, default="pending")

    # Task details
    total_datasets = models.IntegerField(default=0)
    processed_datasets = models.IntegerField(default=0)
    failed_datasets = models.IntegerField(default=0)

    # Progress tracking
    current_progress = models.IntegerField(default=0)
    progress_percentage = models.FloatField(default=0.0)
    current_status_message = models.TextField(blank=True)

    # Results
    failed_dataset_details = models.JSONField(default=list, blank=True)
    result_summary = models.TextField(blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Bulk Reprocessing Task {self.task_id} - {self.status}"

    @property
    def is_completed(self):
        return self.status in [
            "completed",
            "completed_with_errors",
            "failed",
            "cancelled",
        ]

    @property
    def is_running(self):
        return self.status in ["pending", "processing"]


class NoiseRecording(models.Model):
   

    STATUS_CHOICES = [
        ('pending', 'Pending Processing'),
        ('processed', 'Processed'),
        ('failed', 'Failed'),
    ]

    contributor = models.ForeignKey(
        CustomUser,
        on_delete=models.PROTECT,
        related_name='noise_recordings_contributed',
        help_text="The person who made this recording",
        null=True   
    )

    audio = models.FileField(
        upload_to="recordings/",
        help_text="Raw audio recording file",
         null=True   
    )

    duration = models.FloatField(
        null=True,
        blank=True,
        help_text="Duration of recording in seconds"

    )
    approved = models.BooleanField(default=False, help_text="Whether the recording has been approved")
    approved_by = models.ForeignKey(
        CustomUser,
        on_delete=models.PROTECT,
        null=True,
        blank=True,
        related_name='noise_recordings_approved',
        help_text="The person who approved this recording"
    )
    approved_at = models.DateTimeField(null=True, blank=True)

    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='pending',
        help_text="Processing status of the recording"
        
    )

    # Optional metadata that can be added later
    recording_date = models.DateTimeField(
        auto_now_add=True,
        help_text="When the recording was made"
    )

    device_info = models.JSONField(
        null=True,
        blank=True,
        help_text="Device/browser information (JSON)"
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Noise Recording"
        verbose_name_plural = "Noise Recordings"

    def __str__(self):
        return f"Recording by {self.collector.username} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"

    def get_audio_hash(self):
        """Generate MD5 hash of the audio file content safely."""
        try:
            if self.audio and self.audio.file:
                hash_md5 = hashlib.md5()
                for chunk in self.audio.file.chunks():
                    hash_md5.update(chunk)
                return hash_md5.hexdigest()
        except Exception as e:
            print(f"Error generating hash for {self.audio.name}: {e}")
        return None

    @property
    def audio_size_mb(self):
        """Return audio file size in MB"""
        try:
            return round(self.audio.size / (1024 * 1024), 2)
        except:
            return None
