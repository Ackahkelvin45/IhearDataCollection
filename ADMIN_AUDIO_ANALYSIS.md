# Admin Audio Analysis Management

This document describes the new admin interface for managing audio analysis for noise datasets.

## Overview

The admin interface now includes tools to identify and reprocess noise datasets that are missing audio features or noise analysis. This is particularly useful when:

- Datasets were uploaded before the audio analysis features were implemented
- Audio processing failed for some datasets
- You want to reprocess datasets with updated analysis algorithms

## Features

### 1. Missing Analysis Dashboard

**URL**: `/admin/data/noisedataset/missing-analysis/`

This page shows:
- Statistics cards with counts of datasets missing different types of analysis
- Separate tables for datasets missing:
  - Audio features only
  - Noise analysis only
  - Both audio features and noise analysis
  - Datasets without audio files (cannot be processed)

### 2. Bulk Reprocessing

**URL**: `/admin/data/noisedataset/redo-analysis/`

This page allows you to:
- View all datasets that need reprocessing
- Confirm and start bulk reprocessing
- See warnings about datasets without audio files

### 3. Admin List Enhancements

The main noise dataset admin list now shows:
- Processing status column with visual indicators
- Warning banner when missing analysis is detected
- Quick action buttons to view missing analysis or start reprocessing

### 4. Admin Actions

In the admin list, you can:
- Select specific datasets and use the "🔄 Reprocess audio analysis for selected datasets" action
- This will only process datasets that have audio files

## How to Use

### View Missing Analysis

1. Go to the Django admin
2. Navigate to "Data" → "Noise datasets"
3. If there are datasets with missing analysis, you'll see a warning banner
4. Click "🔍 View Missing Analysis" to see detailed information

### Reprocess All Missing Analysis

1. From the missing analysis page, click "🔄 Redo Noise Analysis for All Missing Datasets"
2. Review the confirmation page showing which datasets will be processed
3. Click "🔄 Confirm and Start Reprocessing"
4. The system will start processing all eligible datasets in the background

### Reprocess Selected Datasets

1. Go to the noise datasets admin list
2. Select the datasets you want to reprocess
3. Choose "🔄 Reprocess audio analysis for selected datasets" from the actions dropdown
4. Click "Go"

## Processing Details

### What Gets Processed

For each dataset, the system will:
1. Extract audio features (MFCCs, spectral features, etc.)
2. Perform noise analysis (dB levels, frequency analysis, etc.)
3. Create visualization presets
4. Store results in the `AudioFeature` and `NoiseAnalysis` models

### Requirements

- Dataset must have an audio file uploaded
- Audio file must be in a supported format (WAV, MP3, FLAC, OGG, AIFF, M4A)
- The Celery worker must be running to process background tasks

### Error Handling

- Datasets without audio files are automatically excluded
- Processing errors are logged but don't stop the entire batch
- You can check the logs for any processing failures

## Technical Implementation

### Models Used

- `NoiseDataset`: Main dataset model
- `AudioFeature`: Stores extracted audio features
- `NoiseAnalysis`: Stores noise analysis results
- `VisualizationPreset`: Stores chart configurations

### Tasks

- `process_audio_task`: Celery task that processes individual datasets
- Uses the existing `process_audio_file` function from `utils.py`

### Templates

- `missing_analysis.html`: Shows datasets with missing analysis
- `redo_analysis_confirm.html`: Confirmation page for bulk processing
- `change_list.html`: Enhanced admin list with warnings and actions

## Monitoring

### Check Processing Status

1. Look at the "Processing Status" column in the admin list
2. Status indicators:
   - ❌ No Audio File: Dataset has no audio file uploaded
   - ⚠️ Missing Audio Features: Audio features not extracted
   - ⚠️ Missing Noise Analysis: Noise analysis not performed
   - ✅ Complete: All analysis completed

### Logs

Check the Django logs for:
- Processing progress
- Error messages
- Task completion status

## Troubleshooting

### Common Issues

1. **No datasets being processed**: Check that datasets have audio files uploaded
2. **Processing fails**: Check that the Celery worker is running
3. **Audio file errors**: Check that audio files are in supported formats

### Debugging

1. Check the Django admin for processing status
2. Review Celery task logs
3. Verify audio file accessibility and format

## Future Enhancements

Potential improvements:
- Progress tracking for bulk operations
- Email notifications when processing completes
- More detailed error reporting
- Support for additional audio formats
- Batch size limits and throttling 