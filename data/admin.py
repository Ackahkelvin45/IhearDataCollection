from django.contrib import admin
from .models import NoiseDataset
from unfold.admin import ModelAdmin
import pandas as pd


@admin.register(NoiseDataset)
class NoiseDatasetAdmin(ModelAdmin):
    list_display = ('noise_id', 'name', 'collector',"subclass", 'category', 'region', 'community', 'recording_date', 'updated_at')
  
    list_filter = ('category', 'region', 'community', 'time_of_day', 'class_name', "subclass",'recording_date')
    

    search_fields = ('noise_id', 'name', 'description', 'community__name', 'collector__username')
    
  
    list_editable = ('name',)
    
    list_select_related = ('collector', 'category', 'region', 'community')
    
    list_per_page = 25
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('noise_id', 'name', 'collector', 'description', 'audio', 'duration')
        }),
        ('Location Details', {
            'fields': ('region', 'community', )
        }),
        ('Classification', {
            'fields': ('category', 'class_name','subclass')
        }),
        ('Recording Details', {
            'fields': ('time_of_day', 'recording_device', 'microphone_type', 'recording_date')
        }),
    )
    
    prepopulated_fields = {'name': ('category', 'community')}
    
    readonly_fields = ('noise_id', 'updated_at','duration')
    
    date_hierarchy = 'recording_date'