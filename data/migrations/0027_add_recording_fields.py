# Add missing fields to Recording model

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('data', '0026_create_cleanspeechdataset'),
    ]

    operations = [
        migrations.AddField(
            model_name='recording',
            name='recording_type',
            field=models.CharField(
                choices=[
                    ('clean_speech', 'Clean Speech Recording'),
                    ('english_language', 'English Language'),
                    ('scripted_speech', 'Scripted Speech')
                ],
                default='clean_speech',
                help_text='Type of recording',
                max_length=50
            ),
        ),
        migrations.AddField(
            model_name='recording',
            name='status',
            field=models.CharField(
                choices=[
                    ('pending', 'Pending Processing'),
                    ('processing', 'Processing'),
                    ('completed', 'Completed'),
                    ('failed', 'Failed')
                ],
                default='pending',
                help_text='Processing status of the recording',
                max_length=50
            ),
        ),
        migrations.AddField(
            model_name='recording',
            name='device_info',
            field=models.JSONField(
                blank=True,
                help_text='Device and browser information when recording was made',
                null=True
            ),
        ),
    ]