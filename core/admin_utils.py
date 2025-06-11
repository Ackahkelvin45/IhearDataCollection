# admin_utils.py
import pandas as pd
from django.core.exceptions import ObjectDoesNotExist
from core.models import Category, Class, SubClass

def import_from_excel(file_path):
    # Read the Excel file
    df = pd.read_excel(file_path)
    
    # Initialize counters
    categories_created = 0
    classes_created = 0
    subclasses_created = 0
    
    current_category = None
    current_class = None
    
    for index, row in df.iterrows():
        # Process Category if exists in this row
        if pd.notna(row['Categories']):
            category_name = row['Categories'].strip()
            current_category, created = Category.objects.get_or_create(
                name=category_name
            )
            if created:
                categories_created += 1
        
        # Process Class if exists in this row
        if pd.notna(row['Classes']):
            class_name = row['Classes'].strip()
            if current_category:  # Only create if we have a category
                current_class, created = Class.objects.get_or_create(
                    name=class_name,
                    category=current_category
                )
                if created:
                    classes_created += 1
        
        # Process SubClass if exists in this row
        if pd.notna(row['Sub-Class']):
            subclass_name = row['Sub-Class'].strip()
            if current_class:  # Only create if we have a class
                _, created = SubClass.objects.get_or_create(
                    name=subclass_name,
                    parent_class=current_class
                )
                if created:
                    subclasses_created += 1
    
    return {
        'categories_created': categories_created,
        'classes_created': classes_created,
        'subclasses_created': subclasses_created
    }