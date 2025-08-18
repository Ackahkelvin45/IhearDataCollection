from django.core.management.base import BaseCommand
from data.models import NoiseDataset
from data.utils import safe_process_audio_file
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Test audio processing to verify numba compatibility fix'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dataset-id',
            type=int,
            help='Specific dataset ID to test',
        )
        parser.add_argument(
            '--limit',
            type=int,
            default=5,
            help='Number of datasets to test (default: 5)',
        )

    def handle(self, *args, **options):
        dataset_id = options.get('dataset_id')
        limit = options.get('limit')
        
        if dataset_id:
            # Test specific dataset
            try:
                dataset = NoiseDataset.objects.get(id=dataset_id)
                self.test_single_dataset(dataset)
            except NoiseDataset.DoesNotExist:
                self.stdout.write(
                    self.style.ERROR(f'Dataset with ID {dataset_id} not found')
                )
        else:
            # Test multiple datasets
            datasets = NoiseDataset.objects.filter(
                audio__isnull=False
            ).order_by('-created_at')[:limit]
            
            if not datasets:
                self.stdout.write(
                    self.style.WARNING('No datasets with audio files found')
                )
                return
            
            self.stdout.write(f'Testing {len(datasets)} datasets...')
            
            success_count = 0
            for dataset in datasets:
                if self.test_single_dataset(dataset):
                    success_count += 1
            
            self.stdout.write(
                self.style.SUCCESS(
                    f'Completed: {success_count}/{len(datasets)} datasets processed successfully'
                )
            )

    def test_single_dataset(self, dataset):
        """Test processing for a single dataset"""
        try:
            self.stdout.write(f'Testing dataset: {dataset.noise_id} (ID: {dataset.id})')
            safe_process_audio_file(dataset)
            self.stdout.write(
                self.style.SUCCESS(f'✅ Successfully processed {dataset.noise_id}')
            )
            return True
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'❌ Failed to process {dataset.noise_id}: {e}')
            )
            return False 