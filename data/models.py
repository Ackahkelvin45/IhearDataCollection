from django.db import models

# Create your models here.
from core.models import Region, Category, Community, Class, Microphone_Type,Environment_Type,Time_Of_Day,Specific_Mix_Setting
from authentication.models import CustomUser




class NoiseDataset(models.Model):
    name = models.CharField(
        max_length=100,
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
    environment_type = models.ForeignKey(
        Environment_Type,
        on_delete=models.PROTECT,
        null=True,
        help_text="Environment Type (Urban, Rural, Coastal, Forested, etc.)"
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

    specific_mix_setting = models.ForeignKey(
        Specific_Mix_Setting,
        on_delete=models.PROTECT,
        null=True,
        help_text="Specific Mix Setting (Roadside, Market, Residential area, etc.)"
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
        max_length=100,
        null=True,
        help_text="Duration of the audio in seconds"
    )
    recording_date = models.DateTimeField(
        null=True,
        help_text="Date when recording was made"
    )
    recording_device = models.CharField(
        max_length=100,
        help_text="Recording Device (e.g., iPhone 16, Zoom H4n, etc.)"
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        help_text="When this record was last updated"
    )
    noise_id = models.CharField(
        max_length=20,
        blank=True,
        null=True,
        unique=True,
        help_text="the noise_id,auto generated"
    )

    def __str__(self):
        return f'{self.name}-{self.noise_id}'