# Add clean_speech_dataset field to VisualizationPreset

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('data', '0028_add_clean_speech_feature_models'),
    ]

    operations = [
        migrations.AddField(
            model_name='visualizationpreset',
            name='clean_speech_dataset',
            field=models.ForeignKey(
                blank=True,
                help_text='Clean speech dataset this preset belongs to',
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name='visualization_presets',
                to='data.cleanspeechdataset'
            ),
        ),
    ]