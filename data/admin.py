from django.contrib import admin
from .models import NoiseDataset
from unfold.admin import ModelAdmin

@admin.register(NoiseDataset)
class NoiseDatasetAdmin(ModelAdmin):
    # Fields to display in the list view
    list_display = ('noise_id', 'name', 'collector', 'category', 'region', 'community', 'recording_date', 'updated_at')
    
    # Enable filtering by these fields
    list_filter = ('category', 'region', 'community', 'environment_type', 'time_of_day', 'class_name', 'recording_date')
    
    # Enable search for these fields
    search_fields = ('noise_id', 'name', 'description', 'community__name', 'collector__username')
    
    # Fields that can be edited directly in the list view
    list_editable = ('name',)
    
    list_select_related = ('collector', 'category', 'region', 'community')
    
    list_per_page = 25
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('noise_id', 'name', 'collector', 'description', 'audio', 'duration')
        }),
        ('Location Details', {
            'fields': ('region', 'community', 'environment_type')
        }),
        ('Classification', {
            'fields': ('category', 'class_name', 'specific_mix_setting')
        }),
        ('Recording Details', {
            'fields': ('time_of_day', 'recording_device', 'microphone_type', 'recording_date')
        }),
    )
    
    prepopulated_fields = {'name': ('category', 'community')}
    
    readonly_fields = ('noise_id', 'updated_at','duration')
    
    date_hierarchy = 'recording_date'