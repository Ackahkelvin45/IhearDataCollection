from django import forms
from django.forms.widgets import DateTimeInput
from .models import NoiseDataset, CleanSpeechDataset
from core.models import (
    Community,
    Class,
    SubClass,
    Time_Of_Day,
    Microphone_Type,
    Category,
    Region,
    CleanSpeechCategory,
    CleanSpeechClass,
)


class NoiseDatasetForm(forms.ModelForm):
    class Meta:
        model = NoiseDataset
        fields = [
            "description",
            "region",
            "category",
            "time_of_day",
            "community",
            "class_name",
            "microphone_type",
            "audio",
            "recording_date",
            "recording_device",
            "subclass",
        ]

        widgets = {
            "description": forms.Textarea(
                attrs={
                    "class": "focus:shadow-primary-outline dark:bg-slate-850 dark:text-white text-sm leading-5.6 ease block w-full appearance-none rounded-lg border border-solid border-gray-300 bg-white bg-clip-padding px-3 py-2 font-normal text-gray-700 outline-none transition-all placeholder:text-gray-500 focus:border-blue-500 focus:outline-none",
                    "rows": 3,
                    "placeholder": "Any additional notes about this recording",
                }
            ),
            "region": forms.Select(
                attrs={
                    "class": "focus:shadow-primary-outline dark:bg-slate-850 dark:text-white text-sm leading-5.6 ease block w-full appearance-none rounded-lg border border-solid border-gray-300 bg-white bg-clip-padding px-3 py-2 font-normal text-gray-700 outline-none transition-all placeholder:text-gray-500 focus:border-blue-500 focus:outline-none"
                }
            ),
            "category": forms.Select(
                attrs={
                    "class": "focus:shadow-primary-outline dark:bg-slate-850 dark:text-white text-sm leading-5.6 ease block w-full appearance-none rounded-lg border border-solid border-gray-300 bg-white bg-clip-padding px-3 py-2 font-normal text-gray-700 outline-none transition-all placeholder:text-gray-500 focus:border-blue-500 focus:outline-none"
                }
            ),
            "time_of_day": forms.Select(
                attrs={
                    "class": "focus:shadow-primary-outline dark:bg-slate-850 dark:text-white text-sm leading-5.6 ease block w-full appearance-none rounded-lg border border-solid border-gray-300 bg-white bg-clip-padding px-3 py-2 font-normal text-gray-700 outline-none transition-all placeholder:text-gray-500 focus:border-blue-500 focus:outline-none"
                }
            ),
            "community": forms.Select(
                attrs={
                    "class": "focus:shadow-primary-outline dark:bg-slate-850 dark:text-white text-sm leading-5.6 ease block w-full appearance-none rounded-lg border border-solid border-gray-300 bg-white bg-clip-padding px-3 py-2 font-normal text-gray-700 outline-none transition-all placeholder:text-gray-500 focus:border-blue-500 focus:outline-none"
                }
            ),
            "class_name": forms.Select(
                attrs={
                    "class": "focus:shadow-primary-outline dark:bg-slate-850 dark:text-white text-sm leading-5.6 ease block w-full appearance-none rounded-lg border border-solid border-gray-300 bg-white bg-clip-padding px-3 py-2 font-normal text-gray-700 outline-none transition-all placeholder:text-gray-500 focus:border-blue-500 focus:outline-none"
                }
            ),
            "subclass": forms.Select(
                attrs={
                    "class": "focus:shadow-primary-outline dark:bg-slate-850 dark:text-white text-sm leading-5.6 ease block w-full appearance-none rounded-lg border border-solid border-gray-300 bg-white bg-clip-padding px-3 py-2 font-normal text-gray-700 outline-none transition-all placeholder:text-gray-500 focus:border-blue-500 focus:outline-none"
                }
            ),
            "microphone_type": forms.Select(
                attrs={
                    "class": "focus:shadow-primary-outline dark:bg-slate-850 dark:text-white text-sm leading-5.6 ease block w-full appearance-none rounded-lg border border-solid border-gray-300 bg-white bg-clip-padding px-3 py-2 font-normal text-gray-700 outline-none transition-all placeholder:text-gray-500 focus:border-blue-500 focus:outline-none"
                }
            ),
            "audio": forms.FileInput(
                attrs={
                    "class": "focus:shadow-primary-outline dark:bg-slate-850 dark:text-white text-sm leading-5.6 ease block w-full appearance-none rounded-lg border border-solid border-gray-300 bg-white bg-clip-padding px-3 py-2 font-normal text-gray-700 outline-none transition-all placeholder:text-gray-500 focus:border-blue-500 focus:outline-none",
                    "accept": "audio/*",
                }
            ),
            "recording_date": DateTimeInput(
                attrs={
                    "class": "focus:shadow-primary-outline dark:bg-slate-850 dark:text-white text-sm leading-5.6 ease block w-full appearance-none rounded-lg border border-solid border-gray-300 bg-white bg-clip-padding px-3 py-2 font-normal text-gray-700 outline-none transition-all placeholder:text-gray-500 focus:border-blue-500 focus:outline-none",
                    "type": "datetime-local",
                }
            ),
            "recording_device": forms.TextInput(
                attrs={
                    "class": "focus:shadow-primary-outline dark:bg-slate-850 dark:text-white text-sm leading-5.6 ease block w-full appearance-none rounded-lg border border-solid border-gray-300 bg-white bg-clip-padding px-3 py-2 font-normal text-gray-700 outline-none transition-all placeholder:text-gray-500 focus:border-blue-500 focus:outline-none",
                    "placeholder": "e.g., iPhone 16, Zoom H4n, etc.",
                }
            ),
        }

        labels = {
            "description": "Description",
            "region": "Region",
            "category": "Category",
            "time_of_day": "Time of Day",
            "community": "Community",
            "class_name": "Class",
            "microphone_type": "Microphone Type",
            "audio": "Audio File",
            "recording_date": "Recording Date",
            "recording_device": "Recording Device",
        }

        help_texts = {
            "description": "Any additional notes about this recording",
            "region": "Select the region where recording was made (Ashanti, Central, Greater Accra, etc.)",
            "category": "Category of the data",
            "time_of_day": "Time of Day (Day, Night, etc.)",
            "community": "Specific community (Kotei, Adum, Ayeduase, etc.)",
            "class_name": "Class of the data",
            "microphone_type": "Microphone Type (Omnidirectional, Directional, etc.)",
            "audio": "Upload audio file",
            "recording_date": "Date when recording was made",
            "recording_device": "Recording Device (e.g., iPhone 16, Zoom H4n, etc.)",
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set empty labels
        self.fields["region"].empty_label = "Select Region"
        self.fields["category"].empty_label = "Select Category"
        self.fields["time_of_day"].empty_label = "Select Time of Day"
        self.fields["community"].empty_label = "Select Community"
        self.fields["class_name"].empty_label = "Select Class"
        self.fields["microphone_type"].empty_label = "Select Microphone Type "

        # Make microphone_type optional
        self.fields["microphone_type"].required = False

        # Handle the region-community relationship
        if "region" in self.data:
            try:
                region_id = int(self.data.get("region"))
                self.fields["community"].queryset = Community.objects.filter(
                    region_id=region_id
                ).order_by("name")
            except (ValueError, TypeError):
                pass  # invalid input from the client; ignore and fallback to empty queryset
        elif self.instance.pk and self.instance.region:
            # If editing an existing instance with a region, show its communities
            self.fields["community"].queryset = (
                self.instance.region.communities.order_by("name")
            )
        else:
            # If new instance or no region, show empty community queryset
            self.fields["community"].queryset = Community.objects.none()

        # Handle the category-class relationship
        if "category" in self.data:
            try:
                category_id = int(self.data.get("category"))
                self.fields["class_name"].queryset = Class.objects.filter(
                    category_id=category_id
                ).order_by("name")
            except (ValueError, TypeError):
                pass  # invalid input from the client; ignore and fallback to empty queryset
        elif self.instance.pk and self.instance.category:
            # If editing an existing instance, show the current related classes
            self.fields["class_name"].queryset = (
                self.instance.category.classes.order_by("name")
            )
        else:
            # If new instance, show empty class_name queryset
            self.fields["class_name"].queryset = Class.objects.none()

        # Add subclass field to the form (if not already in fields)
        self.fields["subclass"] = forms.ModelChoiceField(
            queryset=SubClass.objects.none(),
            required=False,
            label="Sub Class",
            help_text="Sub Class of the data",
            widget=forms.Select(
                attrs={
                    "class": "focus:shadow-primary-outline dark:bg-slate-850 dark:text-white text-sm leading-5.6 ease block w-full appearance-none rounded-lg border border-solid border-gray-300 bg-white bg-clip-padding px-3 py-2 font-normal text-gray-700 outline-none transition-all placeholder:text-gray-500 focus:border-blue-500 focus:outline-none"
                }
            ),
        )

        # Filter subclass queryset based on class_name
        if "class_name" in self.data:
            try:
                class_id = int(self.data.get("class_name"))
                self.fields["subclass"].queryset = SubClass.objects.filter(
                    parent_class_id=class_id
                ).order_by("name")
            except (ValueError, TypeError):
                pass  # invalid input from the client; ignore and fallback to empty queryset
        elif self.instance.pk and self.instance.class_name:
            # If editing an existing instance with a class_name, show its subclasses
            self.fields["subclass"].queryset = (
                self.instance.class_name.subclasses.order_by("name")
            )
        else:
            # If new instance or no class_name, show empty subclass queryset
            self.fields["subclass"].queryset = SubClass.objects.none()


class MultipleFileInput(forms.ClearableFileInput):
    allow_multiple_selected = True


class MultipleFileField(forms.FileField):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("widget", MultipleFileInput())
        super().__init__(*args, **kwargs)

    def clean(self, data, initial=None):
        single_file_clean = super().clean
        if isinstance(data, (list, tuple)):
            return [single_file_clean(d, initial) for d in data]
        return single_file_clean(data, initial)


class BulkAudioUploadForm(forms.Form):
    audio_files = MultipleFileField(
        label="Select multiple audio files",
        help_text="Hold Ctrl/Cmd to select multiple files",
        required=False,
        widget=MultipleFileInput(
            attrs={
                "class": "focus:shadow-primary-outline dark:bg-slate-850 dark:text-white text-sm leading-5.6 ease block w-full appearance-none rounded-lg border border-solid border-gray-300 bg-white bg-clip-padding px-3 py-2 font-normal text-gray-700 outline-none transition-all placeholder:text-gray-500 focus:border-blue-500 focus:outline-none",
                "accept": "audio/*",
            }
        ),
    )
    description = forms.CharField(
        widget=forms.Textarea(
            attrs={
                "class": "focus:shadow-primary-outline dark:bg-slate-850 dark:text-white text-sm leading-5.6 ease block w-full appearance-none rounded-lg border border-solid border-gray-300 bg-white bg-clip-padding px-3 py-2 font-normal text-gray-700 outline-none transition-all placeholder:text-gray-500 focus:border-blue-500 focus:outline-none",
                "rows": 3,
                "placeholder": "Any additional notes about these recordings",
            }
        ),
        required=False,
    )
    region = forms.ModelChoiceField(
        queryset=Region.objects.all(),
        widget=forms.Select(
            attrs={
                "class": "focus:shadow-primary-outline dark:bg-slate-850 dark:text-white text-sm leading-5.6 ease block w-full appearance-none rounded-lg border border-solid border-gray-300 bg-white bg-clip-padding px-3 py-2 font-normal text-gray-700 outline-none transition-all placeholder:text-gray-500 focus:border-blue-500 focus:outline-none"
            }
        ),
        empty_label="Select Region",
    )
    category = forms.ModelChoiceField(
        queryset=Category.objects.all(),
        widget=forms.Select(
            attrs={
                "class": "focus:shadow-primary-outline dark:bg-slate-850 dark:text-white text-sm leading-5.6 ease block w-full appearance-none rounded-lg border border-solid border-gray-300 bg-white bg-clip-padding px-3 py-2 font-normal text-gray-700 outline-none transition-all placeholder:text-gray-500 focus:border-blue-500 focus:outline-none"
            }
        ),
        empty_label="Select Category",
    )
    time_of_day = forms.ModelChoiceField(
        queryset=Time_Of_Day.objects.all(),
        widget=forms.Select(
            attrs={
                "class": "focus:shadow-primary-outline dark:bg-slate-850 dark:text-white text-sm leading-5.6 ease block w-full appearance-none rounded-lg border border-solid border-gray-300 bg-white bg-clip-padding px-3 py-2 font-normal text-gray-700 outline-none transition-all placeholder:text-gray-500 focus:border-blue-500 focus:outline-none"
            }
        ),
        empty_label="Select Time of Day",
    )
    community = forms.ModelChoiceField(
        queryset=Community.objects.none(),
        widget=forms.Select(
            attrs={
                "class": "focus:shadow-primary-outline dark:bg-slate-850 dark:text-white text-sm leading-5.6 ease block w-full appearance-none rounded-lg border border-solid border-gray-300 bg-white bg-clip-padding px-3 py-2 font-normal text-gray-700 outline-none transition-all placeholder:text-gray-500 focus:border-blue-500 focus:outline-none"
            }
        ),
        empty_label="Select Community",
    )
    class_name = forms.ModelChoiceField(
        queryset=Class.objects.none(),
        widget=forms.Select(
            attrs={
                "class": "focus:shadow-primary-outline dark:bg-slate-850 dark:text-white text-sm leading-5.6 ease block w-full appearance-none rounded-lg border border-solid border-gray-300 bg-white bg-clip-padding px-3 py-2 font-normal text-gray-700 outline-none transition-all placeholder:text-gray-500 focus:border-blue-500 focus:outline-none"
            }
        ),
        empty_label="Select Class",
    )
    subclass = forms.ModelChoiceField(
        queryset=SubClass.objects.none(),
        required=False,
        widget=forms.Select(
            attrs={
                "class": "focus:shadow-primary-outline dark:bg-slate-850 dark:text-white text-sm leading-5.6 ease block w-full appearance-none rounded-lg border border-solid border-gray-300 bg-white bg-clip-padding px-3 py-2 font-normal text-gray-700 outline-none transition-all placeholder:text-gray-500 focus:border-blue-500 focus:outline-none"
            }
        ),
        empty_label="Select Sub Class",
    )
    microphone_type = forms.ModelChoiceField(
        queryset=Microphone_Type.objects.all(),
        required=False,
        widget=forms.Select(
            attrs={
                "class": "focus:shadow-primary-outline dark:bg-slate-850 dark:text-white text-sm leading-5.6 ease block w-full appearance-none rounded-lg border border-solid border-gray-300 bg-white bg-clip-padding px-3 py-2 font-normal text-gray-700 outline-none transition-all placeholder:text-gray-500 focus:border-blue-500 focus:outline-none"
            }
        ),
        empty_label="Select Microphone Type",
    )
    recording_date = forms.DateTimeField(
        widget=DateTimeInput(
            attrs={
                "class": "focus:shadow-primary-outline dark:bg-slate-850 dark:text-white text-sm leading-5.6 ease block w-full appearance-none rounded-lg border border-solid border-gray-300 bg-white bg-clip-padding px-3 py-2 font-normal text-gray-700 outline-none transition-all placeholder:text-gray-500 focus:border-blue-500 focus:outline-none",
                "type": "datetime-local",
            }
        )
    )
    recording_device = forms.CharField(
        widget=forms.TextInput(
            attrs={
                "class": "focus:shadow-primary-outline dark:bg-slate-850 dark:text-white text-sm leading-5.6 ease block w-full appearance-none rounded-lg border border-solid border-gray-300 bg-white bg-clip-padding px-3 py-2 font-normal text-gray-700 outline-none transition-all placeholder:text-gray-500 focus:border-blue-500 focus:outline-none",
                "placeholder": "e.g., iPhone 16, Zoom H4n, etc.",
            }
        )
    )

    labels = {
        "description": "Description",
        "region": "Region",
        "category": "Category",
        "time_of_day": "Time of Day",
        "community": "Community",
        "class_name": "Class",
        "microphone_type": "Microphone Type",
        "audio_files": "Audio Files",
        "recording_date": "Recording Date",
        "recording_device": "Recording Device",
    }

    help_texts = {
        "description": "Any additional notes about these recordings",
        "region": "Select the region where recordings were made (Ashanti, Central, Greater Accra, etc.)",
        "category": "Category of the data",
        "time_of_day": "Time of Day (Day, Night, etc.)",
        "community": "Specific community (Kotei, Adum, Ayeduase, etc.)",
        "class_name": "Class of the data",
        "microphone_type": "Microphone Type (Omnidirectional, Directional, etc.)",
        "audio_files": "Upload multiple audio files",
        "recording_date": "Date when recordings were made",
        "recording_device": "Recording Device (e.g., iPhone 16, Zoom H4n, etc.)",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set labels and help texts
        for field_name, label in self.labels.items():
            self.fields[field_name].label = label
        for field_name, help_text in self.help_texts.items():
            self.fields[field_name].help_text = help_text

        # Handle the region-community relationship
        if "region" in self.data:
            try:
                region_id = int(self.data.get("region"))
                self.fields["community"].queryset = Community.objects.filter(
                    region_id=region_id
                ).order_by("name")
            except (ValueError, TypeError):
                pass  # invalid input from the client; ignore and fallback to empty queryset
        elif self.initial.get("region"):
            # If initial region is provided, show its communities
            self.fields["community"].queryset = self.initial[
                "region"
            ].communities.order_by("name")
        else:
            # Otherwise, show empty community queryset
            self.fields["community"].queryset = Community.objects.none()

        # Handle the category-class relationship
        if "category" in self.data:
            try:
                category_id = int(self.data.get("category"))
                self.fields["class_name"].queryset = Class.objects.filter(
                    category_id=category_id
                ).order_by("name")
            except (ValueError, TypeError):
                pass  # invalid input from the client; ignore and fallback to empty queryset
        elif self.initial.get("category"):
            # If initial category is provided, show its classes
            self.fields["class_name"].queryset = self.initial[
                "category"
            ].classes.order_by("name")
        else:
            # Otherwise, show empty class_name queryset
            self.fields["class_name"].queryset = Class.objects.none()

        # Filter subclass queryset based on class_name
        if "class_name" in self.data:
            try:
                class_id = int(self.data.get("class_name"))
                self.fields["subclass"].queryset = SubClass.objects.filter(
                    parent_class_id=class_id
                ).order_by("name")
            except (ValueError, TypeError):
                pass  # invalid input from the client; ignore and fallback to empty queryset
        elif self.initial.get("class_name"):
            # If initial class_name is provided, show its subclasses
            self.fields["subclass"].queryset = self.initial[
                "class_name"
            ].subclasses.order_by("name")
        else:
            # Otherwise, show empty subclass queryset
            self.fields["subclass"].queryset = SubClass.objects.none()


class CleanSpeechDatasetForm(forms.ModelForm):
    class Meta:
        model = CleanSpeechDataset
        fields = [
            "description",
            "region",
            "category",
            "time_of_day",
            "community",
            "class_name",
            "microphone_type",
            "audio",
            "recording_date",
            "recording_device",
        ]

        widgets = {
            "description": forms.Textarea(
                attrs={
                    "class": "focus:shadow-primary-outline dark:bg-slate-850 dark:text-white text-sm leading-5.6 ease block w-full appearance-none rounded-lg border border-solid border-gray-300 bg-white bg-clip-padding px-3 py-2 font-normal text-gray-700 outline-none transition-all placeholder:text-gray-500 focus:border-blue-500 focus:outline-none",
                    "rows": 3,
                    "placeholder": "Any additional notes about this recording",
                }
            ),
            "region": forms.Select(
                attrs={
                    "class": "focus:shadow-primary-outline dark:bg-slate-850 dark:text-white text-sm leading-5.6 ease block w-full appearance-none rounded-lg border border-solid border-gray-300 bg-white bg-clip-padding px-3 py-2 font-normal text-gray-700 outline-none transition-all placeholder:text-gray-500 focus:border-blue-500 focus:outline-none"
                }
            ),
            "category": forms.Select(
                attrs={
                    "class": "focus:shadow-primary-outline dark:bg-slate-850 dark:text-white text-sm leading-5.6 ease block w-full appearance-none rounded-lg border border-solid border-gray-300 bg-white bg-clip-padding px-3 py-2 font-normal text-gray-700 outline-none transition-all placeholder:text-gray-500 focus:border-blue-500 focus:outline-none"
                }
            ),
            "time_of_day": forms.Select(
                attrs={
                    "class": "focus:shadow-primary-outline dark:bg-slate-850 dark:text-white text-sm leading-5.6 ease block w-full appearance-none rounded-lg border border-solid border-gray-300 bg-white bg-clip-padding px-3 py-2 font-normal text-gray-700 outline-none transition-all placeholder:text-gray-500 focus:border-blue-500 focus:outline-none"
                }
            ),
            "community": forms.Select(
                attrs={
                    "class": "focus:shadow-primary-outline dark:bg-slate-850 dark:text-white text-sm leading-5.6 ease block w-full appearance-none rounded-lg border border-solid border-gray-300 bg-white bg-clip-padding px-3 py-2 font-normal text-gray-700 outline-none transition-all placeholder:text-gray-500 focus:border-blue-500 focus:outline-none"
                }
            ),
            "class_name": forms.Select(
                attrs={
                    "class": "focus:shadow-primary-outline dark:bg-slate-850 dark:text-white text-sm leading-5.6 ease block w-full appearance-none rounded-lg border border-solid border-gray-300 bg-white bg-clip-padding px-3 py-2 font-normal text-gray-700 outline-none transition-all placeholder:text-gray-500 focus:border-blue-500 focus:outline-none"
                }
            ),
            "microphone_type": forms.Select(
                attrs={
                    "class": "focus:shadow-primary-outline dark:bg-slate-850 dark:text-white text-sm leading-5.6 ease block w-full appearance-none rounded-lg border border-solid border-gray-300 bg-white bg-clip-padding px-3 py-2 font-normal text-gray-700 outline-none transition-all placeholder:text-gray-500 focus:border-blue-500 focus:outline-none"
                }
            ),
            "recording_date": DateTimeInput(
                attrs={
                    "class": "focus:shadow-primary-outline dark:bg-slate-850 dark:text-white text-sm leading-5.6 ease block w-full appearance-none rounded-lg border border-solid border-gray-300 bg-white bg-clip-padding px-3 py-2 font-normal text-gray-700 outline-none transition-all placeholder:text-gray-500 focus:border-blue-500 focus:outline-none",
                    "type": "datetime-local",
                }
            ),
            "recording_device": forms.TextInput(
                attrs={
                    "class": "focus:shadow-primary-outline dark:bg-slate-850 dark:text-white text-sm leading-5.6 ease block w-full appearance-none rounded-lg border border-solid border-gray-300 bg-white bg-clip-padding px-3 py-2 font-normal text-gray-700 outline-none transition-all placeholder:text-gray-500 focus:border-blue-500 focus:outline-none",
                    "placeholder": "e.g., iPhone 16, Zoom H4n, etc.",
                }
            ),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set initial querysets for ForeignKey fields
        self.fields["region"].queryset = Region.objects.all().order_by("name")
        self.fields["category"].queryset = CleanSpeechCategory.objects.all().order_by("name")
        self.fields["time_of_day"].queryset = Time_Of_Day.objects.all().order_by("name")
        self.fields["community"].queryset = Community.objects.all().order_by("name")
        self.fields["class_name"].queryset = CleanSpeechClass.objects.all().order_by("name")
        self.fields["microphone_type"].queryset = Microphone_Type.objects.all().order_by("name")

        # Handle dynamic filtering for community based on region
        if "region" in self.data:
            try:
                region_id = int(self.data.get("region"))
                self.fields["community"].queryset = Community.objects.filter(
                    region_id=region_id
                ).order_by("name")
            except (ValueError, TypeError):
                pass  # invalid input from the client; ignore and fallback to empty queryset
        elif self.initial.get("region"):
            # If initial region is provided, show its communities
            self.fields["community"].queryset = self.initial[
                "region"
            ].communities.order_by("name")
        else:
            # Otherwise, show empty community queryset
            self.fields["community"].queryset = Community.objects.none()

        # Handle dynamic filtering for class_name based on category
        if "category" in self.data:
            try:
                category_id = int(self.data.get("category"))
                self.fields["class_name"].queryset = CleanSpeechClass.objects.filter(
                    category_id=category_id
                ).order_by("name")
            except (ValueError, TypeError):
                pass  # invalid input from the client; ignore and fallback to empty queryset
        elif self.initial.get("category"):
            # If initial category is provided, show its classes
            self.fields["class_name"].queryset = self.initial[
                "category"
            ].clean_speech_classes.order_by("name")
        else:
            # Otherwise, show empty class queryset
            self.fields["class_name"].queryset = CleanSpeechClass.objects.none()
