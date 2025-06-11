# management/commands/import_categories.py
import os
from django.core.management.base import BaseCommand
from django.conf import settings
from core.admin_utils import import_from_excel

class Command(BaseCommand):
    help = 'Import categories, classes and subclasses from Excel file'

    def add_arguments(self, parser):
        parser.add_argument('file_path', type=str, help='Path to the Excel file')

    def handle(self, *args, **options):
        file_path = options['file_path']
        
        if not os.path.exists(file_path):
            self.stderr.write(self.style.ERROR(f"File not found: {file_path}"))
            return
        
        try:
            result = import_from_excel(file_path)
            self.stdout.write(self.style.SUCCESS(
                f"Successfully imported: "
                f"{result['categories_created']} categories, "
                f"{result['classes_created']} classes, "
                f"{result['subclasses_created']} subclasses"
            ))
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"Error importing data: {str(e)}"))