# Migration to create CleanSpeech models that were missing

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0007_alter_category_name_alter_class_name_and_more'),
        ('data', '0025_alter_recording_options_recording_recording_date_and_more'),
    ]

    operations = [
        # Create CleanSpeechCategory model
        migrations.CreateModel(
            name='CleanSpeechCategory',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255, null=True, unique=True)),
                ('description', models.TextField(blank=True, null=True)),
            ],
        ),

        # Create CleanSpeechClass model
        migrations.CreateModel(
            name='CleanSpeechClass',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255, null=True)),
                ('description', models.TextField(blank=True, null=True)),
                ('category', models.ForeignKey(blank=True, help_text='Category of the data', null=True, on_delete=django.db.models.deletion.PROTECT, related_name='classes', to='core.cleanspeechcategory')),
            ],
            options={
                'unique_together': {('name', 'category')},
            },
        ),

        # Create CleanSpeechSubClass model
        migrations.CreateModel(
            name='CleanSpeechSubClass',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255)),
                ('description', models.TextField(blank=True, null=True)),
                ('parent_class', models.ForeignKey(blank=True, help_text='Sub Class of the data', null=True, on_delete=django.db.models.deletion.PROTECT, related_name='subclasses', to='core.cleanspeechclass')),
            ],
            options={
                'unique_together': {('name', 'parent_class')},
            },
        ),
    ]