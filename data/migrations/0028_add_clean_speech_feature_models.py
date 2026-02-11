# Add clean speech feature and analysis models

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('data', '0027_add_recording_fields'),
    ]

    operations = [
        migrations.CreateModel(
            name='CleanSpeechAudioFeature',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('duration', models.FloatField(help_text='Duration of the audio in seconds', null=True)),
                ('sample_rate', models.IntegerField(help_text='Sample rate of the audio', null=True)),
                ('num_channels', models.IntegerField(help_text='Number of audio channels', null=True)),
                ('bit_depth', models.IntegerField(help_text='Bit depth of the audio', null=True)),
                ('rms_energy', models.FloatField(help_text='Root Mean Square energy', null=True)),
                ('zero_crossing_rate', models.FloatField(help_text='Zero crossing rate', null=True)),
                ('spectral_centroid', models.FloatField(help_text='Spectral centroid', null=True)),
                ('spectral_bandwidth', models.FloatField(help_text='Spectral bandwidth', null=True)),
                ('spectral_rolloff', models.FloatField(help_text='Spectral rolloff', null=True)),
                ('mfcc_mean', models.JSONField(blank=True, help_text='Mean MFCC coefficients', null=True)),
                ('mfcc_std', models.JSONField(blank=True, help_text='Standard deviation of MFCC coefficients', null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('clean_speech_dataset', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='audio_features', to='data.cleanspeechdataset')),
            ],
            options={
                'verbose_name': 'Clean Speech Audio Feature',
                'verbose_name_plural': 'Clean Speech Audio Features',
            },
        ),
        migrations.CreateModel(
            name='CleanSpeechAnalysis',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('analysis_type', models.CharField(choices=[('speech_quality', 'Speech Quality Analysis'), ('clarity', 'Clarity Analysis'), ('background_noise', 'Background Noise Analysis')], default='speech_quality', help_text='Type of analysis performed', max_length=50)),
                ('snr', models.FloatField(blank=True, help_text='Signal-to-Noise Ratio', null=True)),
                ('speech_rate', models.FloatField(blank=True, help_text='Speech rate (words per minute)', null=True)),
                ('articulation_clarity', models.FloatField(blank=True, help_text='Articulation clarity score (0-100)', null=True)),
                ('dynamic_range', models.FloatField(blank=True, help_text='Dynamic range in dB', null=True)),
                ('crest_factor', models.FloatField(blank=True, help_text='Crest factor', null=True)),
                ('overall_quality_score', models.FloatField(blank=True, help_text='Overall quality score (0-100)', null=True)),
                ('recommendations', models.TextField(blank=True, help_text='Analysis recommendations')),
                ('analysis_data', models.JSONField(blank=True, help_text='Raw analysis data', null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('clean_speech_dataset', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='analysis', to='data.cleanspeechdataset')),
            ],
            options={
                'verbose_name': 'Clean Speech Analysis',
                'verbose_name_plural': 'Clean Speech Analyses',
                'unique_together': {('clean_speech_dataset', 'analysis_type')},
            },
        ),
    ]