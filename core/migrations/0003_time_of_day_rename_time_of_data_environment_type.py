# Generated by Django 5.2.2 on 2025-06-08 01:29

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0002_specific_mix_setting_time_of_data'),
    ]

    operations = [
        migrations.CreateModel(
            name='Time_Of_Day',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('description', models.TextField()),
            ],
        ),
        migrations.RenameModel(
            old_name='Time_Of_Data',
            new_name='Environment_Type',
        ),
    ]
