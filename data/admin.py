from django.contrib import admin
from .models import (
    NoiseDataset, 
    AudioFeature, 
    NoiseAnalysis, 

    VisualizationPreset
)
from unfold.admin import ModelAdmin

@admin.register(NoiseDataset)
class NoiseDatasetAdmin(ModelAdmin):
    list_display = ('noise_id', 'name', 'collector', "subclass", 'category', 'region', 'community', 'recording_date', 'updated_at')
    list_filter = ('category', 'region', 'community', 'time_of_day', 'class_name', "subclass", 'recording_date')
    search_fields = ('noise_id', 'name', 'description', 'community__name', 'collector__username')
    list_editable = ('name',)
    list_select_related = ('collector', 'category', 'region', 'community')
    list_per_page = 25
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('noise_id', 'name', 'collector', 'description', 'audio')
        }),
        ('Location Details', {
            'fields': ('region', 'community')
        }),
        ('Classification', {
            'fields': ('category', 'class_name', 'subclass')
        }),
        ('Recording Details', {
            'fields': ('time_of_day', 'recording_device', 'microphone_type', 'recording_date')
        }),
    )
    
    prepopulated_fields = {'name': ('category', 'community')}
    readonly_fields = ('noise_id', 'updated_at')
    date_hierarchy = 'recording_date'

@admin.register(AudioFeature)
class AudioFeatureAdmin(ModelAdmin):
    list_display = ('id', 'noise_dataset', 'duration', 'sample_rate', 'spectral_centroid')
    list_filter = ('noise_dataset__category', 'noise_dataset__class_name')
    search_fields = ('noise_dataset__name', 'noise_dataset__noise_id')
    readonly_fields = ('noise_dataset',)
    list_select_related = ('noise_dataset',)

@admin.register(NoiseAnalysis)
class NoiseAnalysisAdmin(ModelAdmin):
    list_display = ('id', 'noise_dataset', 'mean_db', 'max_db', 'min_db', 'dominant_frequency')
    list_filter = ('noise_dataset__category', 'noise_dataset__class_name')
    search_fields = ('noise_dataset__name', 'noise_dataset__noise_id')
    readonly_fields = ('noise_dataset',)
    list_select_related = ('noise_dataset',)

@admin.register(VisualizationPreset)
class VisualizationPresetAdmin(ModelAdmin):
    list_display = ('name', 'chart_type', 'high_contrast')
    list_filter = ('chart_type', 'high_contrast')
    search_fields = ('name', 'description')
    filter_horizontal = ()