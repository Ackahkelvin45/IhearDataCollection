"""
Management command to move Generator hum from:
Old: Industrial and construction sounds -> Machinery -> Generator hum
New: Urban Life and Public Spaces -> Generic/Background Events -> Generator Hum
"""

from django.core.management.base import BaseCommand
from core.models import Category, Class, SubClass
from data.models import NoiseDataset
from django.db import transaction
import logging

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Move Generator hum from Industrial/Machinery to Urban Life/Generic Background Events"

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be moved without actually moving",
        )
        parser.add_argument(
            "--create-if-not-exists",
            action="store_true",
            default=True,
            help="Create new category/class/subclass if they do not exist (default: True)",
        )

    def handle(self, *args, **options):
        dry_run = options["dry_run"]
        create_if_not_exists = options["create_if_not_exists"]

        self.stdout.write(self.style.SUCCESS("Starting Generator hum migration..."))

        # Old location
        old_category_name = "Industrial_and_const_sounds"
        old_class_name = "Machinery"
        old_subclass_name = "Generator hum"

        # New location
        new_category_name = "Urban_life_and Public _spaces"
        new_class_name = "Generic/Background Events"
        new_subclass_name = "Generator Hum"

        try:
            with transaction.atomic():
                # Find old category, class, and subclass
                try:
                    old_category = Category.objects.get(name=old_category_name)
                    self.stdout.write(f"✓ Found old category: {old_category_name}")
                except Category.DoesNotExist:
                    self.stdout.write(
                        self.style.ERROR(
                            f"✗ Old category not found: {old_category_name}"
                        )
                    )
                    return

                try:
                    old_class = Class.objects.get(
                        name=old_class_name, category=old_category
                    )
                    self.stdout.write(f"✓ Found old class: {old_class_name}")
                except Class.DoesNotExist:
                    self.stdout.write(
                        self.style.ERROR(
                            f"✗ Old class not found: {old_class_name} in {old_category_name}"
                        )
                    )
                    return

                try:
                    old_subclass = SubClass.objects.get(
                        name=old_subclass_name, parent_class=old_class
                    )
                    self.stdout.write(f"✓ Found old subclass: {old_subclass_name}")
                except SubClass.DoesNotExist:
                    self.stdout.write(
                        self.style.ERROR(
                            f"✗ Old subclass not found: {old_subclass_name} in {old_class_name}"
                        )
                    )
                    return

                # Find or create new category
                new_category, created = Category.objects.get_or_create(
                    name=new_category_name,
                    defaults={"description": "Urban life and public spaces sounds"},
                )
                if created:
                    self.stdout.write(
                        self.style.SUCCESS(
                            f"✓ Created new category: {new_category_name}"
                        )
                    )
                else:
                    self.stdout.write(f"✓ Found existing category: {new_category_name}")

                # Find or create new class
                new_class, created = Class.objects.get_or_create(
                    name=new_class_name,
                    category=new_category,
                    defaults={"description": "Generic and background events"},
                )
                if created:
                    self.stdout.write(
                        self.style.SUCCESS(f"✓ Created new class: {new_class_name}")
                    )
                else:
                    self.stdout.write(f"✓ Found existing class: {new_class_name}")

                # Find or create new subclass
                new_subclass, created = SubClass.objects.get_or_create(
                    name=new_subclass_name,
                    parent_class=new_class,
                    defaults={"description": "Generator hum sound"},
                )
                if created:
                    self.stdout.write(
                        self.style.SUCCESS(
                            f"✓ Created new subclass: {new_subclass_name}"
                        )
                    )
                else:
                    self.stdout.write(f"✓ Found existing subclass: {new_subclass_name}")

                # Find all NoiseDataset records with old category/class/subclass
                datasets_to_move = NoiseDataset.objects.filter(
                    category=old_category, class_name=old_class, subclass=old_subclass
                )

                count = datasets_to_move.count()
                self.stdout.write(
                    self.style.WARNING(f"\nFound {count} NoiseDataset records to move")
                )

                if count == 0:
                    self.stdout.write(
                        self.style.WARNING("No records found to move. Exiting.")
                    )
                    return

                if dry_run:
                    self.stdout.write(self.style.WARNING("\n=== DRY RUN MODE ==="))
                    self.stdout.write(
                        f"Would move {count} records from:\n"
                        f"  Category: {old_category_name}\n"
                        f"  Class: {old_class_name}\n"
                        f"  Subclass: {old_subclass_name}\n"
                        f"\nTo:\n"
                        f"  Category: {new_category_name}\n"
                        f"  Class: {new_class_name}\n"
                        f"  Subclass: {new_subclass_name}"
                    )

                    # Show sample records
                    sample_records = datasets_to_move[:5]
                    self.stdout.write("\nSample records that would be moved:")
                    for record in sample_records:
                        self.stdout.write(
                            f'  - {record.noise_id or record.id}: {record.name or "Unnamed"}'
                        )
                    if count > 5:
                        self.stdout.write(f"  ... and {count - 5} more")
                else:
                    # Perform the move
                    updated_count = datasets_to_move.update(
                        category=new_category,
                        class_name=new_class,
                        subclass=new_subclass,
                    )

                    self.stdout.write(
                        self.style.SUCCESS(
                            f"\n✓ Successfully moved {updated_count} NoiseDataset records!"
                        )
                    )
                    self.stdout.write(
                        f"\nMoved from:\n"
                        f"  {old_category_name} -> {old_class_name} -> {old_subclass_name}\n"
                        f"\nTo:\n"
                        f"  {new_category_name} -> {new_class_name} -> {new_subclass_name}"
                    )

                    # Check if old subclass is now unused
                    remaining_with_old_subclass = NoiseDataset.objects.filter(
                        subclass=old_subclass
                    ).count()

                    if remaining_with_old_subclass == 0:
                        self.stdout.write(
                            self.style.WARNING(
                                f'\nNote: Old subclass "{old_subclass_name}" is now unused.'
                            )
                        )
                        self.stdout.write(
                            "You may want to delete it manually if no longer needed."
                        )

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"\n✗ Error during migration: {str(e)}"))
            logger.error(f"Generator hum migration failed: {str(e)}", exc_info=True)
            raise
