# Generated manually to add metadata field to Message model

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('chatbot', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='message',
            name='metadata',
            field=models.JSONField(blank=True, default=dict),
        ),
    ]
