from django.contrib import admin
from .models import *
from unfold.admin import ModelAdmin

@admin.register(Region)
class RegionAdmin(ModelAdmin):
    pass

@admin.register(Category)
class CategoryAdmin(ModelAdmin):
    pass

# Repeat for other models or use a loop:
models_to_register = [Community, Class, Microphone_Type, Environment_Type, Time_Of_Day, Specific_Mix_Setting]

for model in models_to_register:
    admin.site.register(model, ModelAdmin)