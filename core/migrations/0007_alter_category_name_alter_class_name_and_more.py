# Generated by Django 5.2.2 on 2025-06-19 17:47

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0006_community_region_alter_community_description_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='category',
            name='name',
            field=models.CharField(max_length=255, null=True, unique=True),
        ),
        migrations.AlterField(
            model_name='class',
            name='name',
            field=models.CharField(max_length=255, null=True),
        ),
        migrations.AlterField(
            model_name='community',
            name='name',
            field=models.CharField(max_length=255),
        ),
        migrations.AlterField(
            model_name='microphone_type',
            name='name',
            field=models.CharField(max_length=255),
        ),
        migrations.AlterField(
            model_name='region',
            name='name',
            field=models.CharField(max_length=255),
        ),
        migrations.AlterField(
            model_name='subclass',
            name='name',
            field=models.CharField(max_length=255),
        ),
        migrations.AlterField(
            model_name='time_of_day',
            name='name',
            field=models.CharField(max_length=255),
        ),
    ]
