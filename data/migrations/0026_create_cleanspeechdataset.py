# Manual migration to create CleanSpeechDataset table

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('data', '0025_alter_recording_options_recording_recording_date_and_more'),
        ('core', '0009_create_cleanspeech_models'),
    ]

    operations = [
        migrations.CreateModel(
            name='CleanSpeechDataset',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(help_text='Name of Data set, auto generated', max_length=255, null=True)),
                ('description', models.TextField(blank=True, help_text='Any additional notes about this recording', null=True)),
                ('recording_date', models.DateTimeField(help_text='Date when recording was made', null=True)),
                ('recording_device', models.CharField(help_text='Recording Device (e.g., iPhone 16, Zoom H4n, etc.)', max_length=255)),
                ('updated_at', models.DateTimeField(auto_now=True, help_text='When this record was last updated')),
                ('clean_speech_id', models.CharField(blank=True, help_text='the noise_id,auto generated', max_length=255, null=True, unique=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('audio', models.FileField(blank=True, help_text='Processed clean speech audio file', null=True, upload_to='clean_speech_files/')),
                ('category', models.ForeignKey(help_text='Category of the data', null=True, on_delete=django.db.models.deletion.PROTECT, to='core.cleanspeechcategory')),
                ('class_name', models.ForeignKey(help_text='Class of the data', null=True, on_delete=django.db.models.deletion.PROTECT, to='core.cleanspeechclass')),
                ('collector', models.ForeignKey(help_text='The person collecting this data', null=True, on_delete=django.db.models.deletion.PROTECT, to=settings.AUTH_USER_MODEL)),
                ('community', models.ForeignKey(help_text='Specific community where recording was made', null=True, on_delete=django.db.models.deletion.PROTECT, to='core.community')),
                ('dataset_type', models.ForeignKey(blank=True, help_text='', null=True, on_delete=django.db.models.deletion.CASCADE, to='data.dataset')),
                ('microphone_type', models.ForeignKey(help_text='Microphone Type', null=True, on_delete=django.db.models.deletion.PROTECT, to='core.Microphone_Type')),
                ('recording', models.ForeignKey(blank=True, help_text='The raw recording that was processed into this dataset', null=True, on_delete=django.db.models.deletion.CASCADE, to='data.recording')),
                ('region', models.ForeignKey(help_text='Select the region where recording was made', null=True, on_delete=django.db.models.deletion.PROTECT, to='core.region')),
                ('subclass', models.ForeignKey(blank=True, help_text='Sub Class of the data', null=True, on_delete=django.db.models.deletion.PROTECT, to='core.CleanSpeechSubClass')),
                ('time_of_day', models.ForeignKey(help_text='Time of Day (Day, Night, etc.)', null=True, on_delete=django.db.models.deletion.PROTECT, to='core.Time_Of_Day')),
            ],
        ),
    ]