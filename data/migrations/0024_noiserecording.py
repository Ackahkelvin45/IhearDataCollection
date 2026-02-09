# Generated manually for NoiseRecording model

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('data', '0023_noisedataset_has_been_approved_and_more'),
        ('authentication', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='NoiseRecording',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('audio', models.FileField(help_text='Raw audio recording file', upload_to='recordings/')),
                ('duration', models.FloatField(blank=True, help_text='Duration of recording in seconds', null=True)),
                ('status', models.CharField(choices=[('pending', 'Pending Processing'), ('processed', 'Processed'), ('failed', 'Failed')], default='pending', help_text='Processing status of the recording', max_length=20)),
                ('recording_date', models.DateTimeField(help_text='When the recording was made')),
                ('device_info', models.JSONField(blank=True, help_text='Device/browser information (JSON)', null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('collector', models.ForeignKey(help_text='The person who made this recording', on_delete=django.db.models.deletion.PROTECT, to='authentication.customuser')),
            ],
            options={
                'verbose_name': 'Noise Recording',
                'verbose_name_plural': 'Noise Recordings',
                'ordering': ['-created_at'],
            },
        ),
    ]