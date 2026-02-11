from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required, user_passes_test
from django.core.paginator import Paginator
from django.db.models import Q
from django.utils import timezone
from datetime import timedelta

from data.models import Recording, CleanSpeechDataset
from data.utils import process_clean_speech_file
from core.models import CleanSpeechCategory, CleanSpeechClass


@login_required
@user_passes_test(lambda u: u.is_staff or u.user_type == "researcher")
def clean_speech_approval_list(request):
    """List all clean speech recordings that haven't been approved yet"""
    # Get all unapproved clean speech recordings
    recordings = Recording.objects.filter(
        recording_type='clean_speech',
        approved=False
    ).select_related('contributor').order_by('-created_at')

    # Apply filters
    search_query = request.GET.get('search', '')
    category_filter = request.GET.get('category', '')
    class_filter = request.GET.get('class_name', '')
    date_range = request.GET.get('date_range', '')
    page_size = request.GET.get('page_size', '50')

    if search_query:
        recordings = recordings.filter(
            Q(contributor__username__icontains=search_query) |
            Q(contributor__email__icontains=search_query) |
            Q(id__icontains=search_query)
        )

    if category_filter:
        # Filter by recordings that have associated clean speech datasets with this category
        recordings = recordings.filter(
            cleanspeechdataset__category__name=category_filter
        ).distinct()

    if class_filter:
        # Filter by recordings that have associated clean speech datasets with this class
        recordings = recordings.filter(
            cleanspeechdataset__class_name__name=class_filter
        ).distinct()

    # Date range filtering
    if date_range:
        now = timezone.now()
        if date_range == 'today':
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
            recordings = recordings.filter(created_at__gte=start_date)
        elif date_range == 'week':
            start_date = now - timedelta(days=7)
            recordings = recordings.filter(created_at__gte=start_date)
        elif date_range == 'month':
            start_date = now - timedelta(days=30)
            recordings = recordings.filter(created_at__gte=start_date)
        elif date_range == 'year':
            start_date = now - timedelta(days=365)
            recordings = recordings.filter(created_at__gte=start_date)

    # Pagination
    paginator = Paginator(recordings, int(page_size))
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    # Get filter options
    categories = CleanSpeechCategory.objects.all().order_by('name')
    classes = CleanSpeechClass.objects.all().order_by('name')

    context = {
        'page_obj': page_obj,
        'paginator': paginator,
        'recordings': page_obj,
        'categories': categories,
        'classes': classes,
        'current_filters': {
            'search': search_query,
            'category': category_filter,
            'class_name': class_filter,
            'date_range': date_range,
            'page_size': page_size,
        },
        'is_paginated': page_obj.has_other_pages(),
    }

    return render(request, 'approval/approvallist.html', context)


@login_required
@user_passes_test(lambda u: u.is_staff or u.user_type == "researcher")
def clean_speech_approval_review(request, recording_id):
    """Review and approve/reject a clean speech recording"""
    recording = get_object_or_404(
        Recording,
        id=recording_id,
        recording_type='clean_speech'
    )

    # Get associated clean speech dataset if it exists
    try:
        clean_speech_dataset = CleanSpeechDataset.objects.get(recording=recording)
    except CleanSpeechDataset.DoesNotExist:
        clean_speech_dataset = None

    if request.method == 'POST':
        action = request.POST.get('action')

        if action == 'approve':
            recording.approved = True
            recording.approved_by = request.user
            recording.approved_at = timezone.now()
            recording.status = 'completed'  # Mark as processed
            recording.save()

            # Process the clean speech dataset if it exists
            if clean_speech_dataset and clean_speech_dataset.audio:
                try:
                    process_clean_speech_file(clean_speech_dataset)
                    messages.success(request, f'Recording #{recording.id} has been approved and processed successfully.')
                except Exception as e:
                    messages.warning(request, f'Recording #{recording.id} was approved but processing failed: {str(e)}')
            else:
                messages.success(request, f'Recording #{recording.id} has been approved.')

        elif action == 'reject':
            # For rejection, we might want to delete the recording or mark it as rejected
            # For now, let's just mark it as not approved and add a note
            recording.approved = False
            recording.save()

            # Also delete associated dataset if it exists
            if clean_speech_dataset:
                clean_speech_dataset.delete()

            messages.success(request, f'Recording #{recording.id} has been rejected and removed.')

        return redirect('approval:clean_speech_approval_list')

    context = {
        'recording': recording,
        'clean_speech_dataset': clean_speech_dataset,
    }

    return render(request, 'approval/approval_review.html', context)
