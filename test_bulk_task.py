#!/usr/bin/env python
"""
Simple test script to verify the bulk reprocessing task works correctly
"""
import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'datacollection.settings')
django.setup()

from data.tasks import bulk_reprocess_audio_analysis
from data.models import NoiseDataset
from django.db.models import Q

def test_bulk_task():
    """Test the bulk reprocessing task with a small number of datasets"""
    
    # Get a few datasets that need processing
    datasets_to_process = NoiseDataset.objects.filter(
        Q(audio_features__isnull=True) | Q(noise_analysis__isnull=True),
        audio__isnull=False,  # Only process datasets with audio files
    ).values_list('id', flat=True)[:5]  # Just test with 5 datasets
    
    if not datasets_to_process:
        print("No datasets found that need processing")
        return
    
    dataset_list = list(datasets_to_process)
    print(f"Testing bulk reprocessing with {len(dataset_list)} datasets: {dataset_list}")
    
    # Run the task synchronously for testing
    result = bulk_reprocess_audio_analysis.apply(args=[dataset_list, None])
    
    print(f"Task completed with result: {result.get()}")
    print(f"Task status: {result.status}")
    
    if result.successful():
        print("✅ Task completed successfully!")
    else:
        print("❌ Task failed!")
        print(f"Error: {result.info}")

if __name__ == "__main__":
    test_bulk_task() 