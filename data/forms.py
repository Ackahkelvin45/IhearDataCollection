from django import forms
from django.forms.widgets import DateTimeInput
from .models import NoiseDataset
from core.models import Region, Category, Community, Class, Microphone_Type, Environment_Type, Time_Of_Day, Specific_Mix_Setting


class NoiseDatasetForm(forms.ModelForm):
    class Meta:
        model = NoiseDataset
        fields = [
            'description', 'region', 'category', 'environment_type', 
            'time_of_day', 'community', 'class_name', 'specific_mix_setting', 
            'microphone_type', 'audio', 'recording_date', 'recording_device'
        ]
        
        widgets = {
            'description': forms.Textarea(attrs={
                'class': 'focus:shadow-primary-outline dark:bg-slate-850 dark:text-white text-sm leading-5.6 ease block w-full appearance-none rounded-lg border border-solid border-gray-300 bg-white bg-clip-padding px-3 py-2 font-normal text-gray-700 outline-none transition-all placeholder:text-gray-500 focus:border-blue-500 focus:outline-none',
                'rows': 3,
                'placeholder': 'Any additional notes about this recording'
            }),
            'region': forms.Select(attrs={
                'class': 'focus:shadow-primary-outline dark:bg-slate-850 dark:text-white text-sm leading-5.6 ease block w-full appearance-none rounded-lg border border-solid border-gray-300 bg-white bg-clip-padding px-3 py-2 font-normal text-gray-700 outline-none transition-all placeholder:text-gray-500 focus:border-blue-500 focus:outline-none'
            }),
            'category': forms.Select(attrs={
                'class': 'focus:shadow-primary-outline dark:bg-slate-850 dark:text-white text-sm leading-5.6 ease block w-full appearance-none rounded-lg border border-solid border-gray-300 bg-white bg-clip-padding px-3 py-2 font-normal text-gray-700 outline-none transition-all placeholder:text-gray-500 focus:border-blue-500 focus:outline-none'
            }),
            'environment_type': forms.Select(attrs={
                'class': 'focus:shadow-primary-outline dark:bg-slate-850 dark:text-white text-sm leading-5.6 ease block w-full appearance-none rounded-lg border border-solid border-gray-300 bg-white bg-clip-padding px-3 py-2 font-normal text-gray-700 outline-none transition-all placeholder:text-gray-500 focus:border-blue-500 focus:outline-none'
            }),
            'time_of_day': forms.Select(attrs={
                'class': 'focus:shadow-primary-outline dark:bg-slate-850 dark:text-white text-sm leading-5.6 ease block w-full appearance-none rounded-lg border border-solid border-gray-300 bg-white bg-clip-padding px-3 py-2 font-normal text-gray-700 outline-none transition-all placeholder:text-gray-500 focus:border-blue-500 focus:outline-none'
            }),
            'community': forms.Select(attrs={
                'class': 'focus:shadow-primary-outline dark:bg-slate-850 dark:text-white text-sm leading-5.6 ease block w-full appearance-none rounded-lg border border-solid border-gray-300 bg-white bg-clip-padding px-3 py-2 font-normal text-gray-700 outline-none transition-all placeholder:text-gray-500 focus:border-blue-500 focus:outline-none'
            }),
            'class_name': forms.Select(attrs={
                'class': 'focus:shadow-primary-outline dark:bg-slate-850 dark:text-white text-sm leading-5.6 ease block w-full appearance-none rounded-lg border border-solid border-gray-300 bg-white bg-clip-padding px-3 py-2 font-normal text-gray-700 outline-none transition-all placeholder:text-gray-500 focus:border-blue-500 focus:outline-none'
            }),
            'specific_mix_setting': forms.Select(attrs={
                'class': 'focus:shadow-primary-outline dark:bg-slate-850 dark:text-white text-sm leading-5.6 ease block w-full appearance-none rounded-lg border border-solid border-gray-300 bg-white bg-clip-padding px-3 py-2 font-normal text-gray-700 outline-none transition-all placeholder:text-gray-500 focus:border-blue-500 focus:outline-none'
            }),
            'microphone_type': forms.Select(attrs={
                'class': 'focus:shadow-primary-outline dark:bg-slate-850 dark:text-white text-sm leading-5.6 ease block w-full appearance-none rounded-lg border border-solid border-gray-300 bg-white bg-clip-padding px-3 py-2 font-normal text-gray-700 outline-none transition-all placeholder:text-gray-500 focus:border-blue-500 focus:outline-none'
            }),
            'audio': forms.FileInput(attrs={
                'class': 'focus:shadow-primary-outline dark:bg-slate-850 dark:text-white text-sm leading-5.6 ease block w-full appearance-none rounded-lg border border-solid border-gray-300 bg-white bg-clip-padding px-3 py-2 font-normal text-gray-700 outline-none transition-all placeholder:text-gray-500 focus:border-blue-500 focus:outline-none',
                'accept': 'audio/*'
            }),
            'recording_date': DateTimeInput(attrs={
                'class': 'focus:shadow-primary-outline dark:bg-slate-850 dark:text-white text-sm leading-5.6 ease block w-full appearance-none rounded-lg border border-solid border-gray-300 bg-white bg-clip-padding px-3 py-2 font-normal text-gray-700 outline-none transition-all placeholder:text-gray-500 focus:border-blue-500 focus:outline-none',
                'type': 'datetime-local'
            }),
            'recording_device': forms.TextInput(attrs={
                'class': 'focus:shadow-primary-outline dark:bg-slate-850 dark:text-white text-sm leading-5.6 ease block w-full appearance-none rounded-lg border border-solid border-gray-300 bg-white bg-clip-padding px-3 py-2 font-normal text-gray-700 outline-none transition-all placeholder:text-gray-500 focus:border-blue-500 focus:outline-none',
                'placeholder': 'e.g., iPhone 16, Zoom H4n, etc.'
            }),
        }
        
        labels = {
            'description': 'Description',
            'region': 'Region',
            'category': 'Category',
            'environment_type': 'Environment Type',
            'time_of_day': 'Time of Day',
            'community': 'Community',
            'class_name': 'Class',
            'specific_mix_setting': 'Specific Mix Setting',
            'microphone_type': 'Microphone Type',
            'audio': 'Audio File',
            'recording_date': 'Recording Date',
            'recording_device': 'Recording Device',
        }
        
        help_texts = {
            'description': 'Any additional notes about this recording',
            'region': 'Select the region where recording was made (Ashanti, Central, Greater Accra, etc.)',
            'category': 'Category of the data',
            'environment_type': 'Environment Type (Urban, Rural, Coastal, Forested, etc.)',
            'time_of_day': 'Time of Day (Day, Night, etc.)',
            'community': 'Specific community (Kotei, Adum, Ayeduase, etc.)',
            'class_name': 'Class of the data',
            'specific_mix_setting': 'Specific Mix Setting (Roadside, Market, Residential area, etc.)',
            'microphone_type': 'Microphone Type (Omnidirectional, Directional, etc.) - Skip if using mobile phone',
            'audio': 'Upload audio file',
            'recording_date': 'Date when recording was made',
            'recording_device': 'Recording Device (e.g., iPhone 16, Zoom H4n, etc.)',
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add empty option for select fields
        self.fields['region'].empty_label = "Select Region"
        self.fields['category'].empty_label = "Select Category"
        self.fields['environment_type'].empty_label = "Select Environment Type"
        self.fields['time_of_day'].empty_label = "Select Time of Day"
        self.fields['community'].empty_label = "Select Community"
        self.fields['class_name'].empty_label = "Select Class"
        self.fields['specific_mix_setting'].empty_label = "Select Mix Setting"
        self.fields['microphone_type'].empty_label = "Select Microphone Type (Optional)"
        
        # Make microphone_type not required since it's optional for mobile phones
        self.fields['microphone_type'].required = False