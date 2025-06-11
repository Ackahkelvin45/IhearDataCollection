from django.contrib import admin
from .models import *
from unfold.admin import ModelAdmin
from django import forms
from django.shortcuts import render, redirect
from django.urls import path
from django.contrib import messages
from django.http import HttpResponseRedirect
import pandas as pd
from core.admin_utils import import_from_excel
import os
import tempfile


@admin.register(Region)
class RegionAdmin(ModelAdmin):
    pass


# Repeat for other models or use a loop:
models_to_register = [Community, Microphone_Type, Time_Of_Day]

for model in models_to_register:
    admin.site.register(model, ModelAdmin)


class CsvImportForm(forms.Form):
    excel_file = forms.FileField(
        label='Excel File',
        help_text='Select an Excel file (.xlsx or .xls)',
        widget=forms.FileInput(attrs={'accept': '.xlsx,.xls'})
    )


@admin.register(Category)
class CategoryAdmin(ModelAdmin):
    list_display = ('name', 'description')
    search_fields = ('name',)


@admin.register(Class)
class ClassAdmin(ModelAdmin):
    list_display = ('name', 'category', 'description')
    list_filter = ('category',)
    search_fields = ('name', 'category__name')


@admin.register(SubClass)
class SubClassAdmin(ModelAdmin):
    list_display = ('name', 'parent_class', 'get_category')
    list_filter = ('parent_class__category', 'parent_class')
    search_fields = ('name', 'parent_class__name')
    
    def get_category(self, obj):
        return obj.parent_class.category
    get_category.short_description = 'Category'
    get_category.admin_order_field = 'parent_class__category'

    # Add custom upload view
    change_list_template = "admin/subclass_changelist.html"
    
    def get_urls(self):
        urls = super().get_urls()
        my_urls = [
            path('import-excel/', self.import_excel, name='subclass_import_excel'),
        ]
        return my_urls + urls
    
    def import_excel(self, request):
        print(f"Request method: {request.method}")
        print(f"Request POST: {request.POST}")
        print(f"Request FILES: {request.FILES}")
        
        if request.method == "POST":
            # Check if file was uploaded
            if 'excel_file' not in request.FILES:
                messages.error(request, "No file was uploaded. Please select a file.")
                return redirect('..')
            
            excel_file = request.FILES["excel_file"]
            
            # Validate file
            if not excel_file:
                messages.error(request, "No file selected.")
                return redirect('..')
            
            # Check file extension
            if not excel_file.name.lower().endswith(('.xlsx', '.xls')):
                messages.error(request, "Please upload an Excel file (.xlsx or .xls)")
                return redirect('..')
            
            try:
                # Use tempfile for better security
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
                    for chunk in excel_file.chunks():
                        tmp_file.write(chunk)
                    tmp_file_path = tmp_file.name
                
                # Import the data
                result = import_from_excel(tmp_file_path)
                
                # Clean up temp file
                os.unlink(tmp_file_path)
                
                messages.success(
                    request,
                    f"Successfully imported: "
                    f"{result['categories_created']} categories, "
                    f"{result['classes_created']} classes, "
                    f"{result['subclasses_created']} subclasses"
                )
                
            except Exception as e:
                # Clean up temp file if it exists
                if 'tmp_file_path' in locals():
                    try:
                        os.unlink(tmp_file_path)
                    except:
                        pass
                
                messages.error(request, f"Error importing data: {str(e)}")
            
            return redirect('..')
        
        # GET request - show the form
        print("Showing form for GET request")
        form = CsvImportForm()
        context = {
            'form': form,
            'title': 'Import Excel File',
            'opts': self.model._meta,
            'has_change_permission': self.has_change_permission(request),
        }
        return render(request, "admin/excel_import_form.html", context)