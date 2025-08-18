#!/usr/bin/env python3
"""
Test script to verify the audio processing fix for numba compatibility issues.
This script can be run to test if the audio processing works without the get_call_template error.
"""

import os
import sys
import django

# Add the project directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'datacollection.settings')
django.setup()

from data.models import NoiseDataset
from data.utils import safe_process_audio_file
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_audio_processing():
    """Test the audio processing with the new fix"""
    try:
        # Get a sample dataset with audio
        dataset = NoiseDataset.objects.filter(audio__isnull=False).first()
        
        if not dataset:
            logger.warning("No datasets with audio files found. Please upload some audio files first.")
            return False
            
        logger.info(f"Testing audio processing for dataset: {dataset.noise_id}")
        
        # Test the safe processing function
        safe_process_audio_file(dataset)
        
        logger.info("✅ Audio processing test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Audio processing test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_audio_processing()
    sys.exit(0 if success else 1) 