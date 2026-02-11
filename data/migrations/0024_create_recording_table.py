from django.db import migrations, models
import django.db.models.deletion

class Migration(migrations.Migration):

    dependencies = [
        ('data', '0022_remove_recording_device_info'),  # last applied migration
    ]

    operations = [
        migrations.CreateModel(
            name='Recording',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('approved', models.BooleanField(default=False)),
                ('approved_at', models.DateTimeField(null=True, blank=True)),
                ('audio', models.FileField(upload_to='recordings/')),
                ('duration', models.FloatField(null=True, blank=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('contributor', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='authentication.CustomUser')),
                ('approved_by', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='approved_recordings', to='authentication.CustomUser')),
            ],
        ),
    ]
