from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import timedelta
from data_insights.models import ChatMessage

# Default age threshold (minutes). Only messages whose updated_at is older than
# this are considered stuck — so the command can never fail a message that is
# still being processed in-flight.
DEFAULT_STUCK_AGE_MINUTES = 15


class Command(BaseCommand):
    help = "Fix messages that are stuck in processing status"

    def add_arguments(self, parser):
        parser.add_argument(
            "--minutes",
            type=int,
            default=DEFAULT_STUCK_AGE_MINUTES,
            help=(
                "Only reset messages whose updated_at is older than N minutes "
                f"(default: {DEFAULT_STUCK_AGE_MINUTES}). Guards against failing "
                "in-flight messages."
            ),
        )

    def handle(self, *args, **options):
        minutes = options.get("minutes", DEFAULT_STUCK_AGE_MINUTES)
        if minutes < 0:
            minutes = DEFAULT_STUCK_AGE_MINUTES
        cutoff = timezone.now() - timedelta(minutes=minutes)

        # Find messages stuck in processing status AND older than the cutoff, so
        # a message that is still streaming (recently updated) is never touched.
        stuck_messages = ChatMessage.objects.filter(
            status=ChatMessage.MessageStatus.PROCESSING,
            updated_at__lt=cutoff,
        )

        count = stuck_messages.count()
        self.stdout.write(
            f"Found {count} messages stuck in processing status "
            f"(older than {minutes} minutes)"
        )

        if count > 0:
            # Update them to failed status with a helpful message
            updated = stuck_messages.update(
                status=ChatMessage.MessageStatus.FAILED,
                assistant_response="This message was interrupted during processing. Please try sending your question again.",
            )

            self.stdout.write(
                self.style.SUCCESS(f"Successfully fixed {updated} stuck messages")
            )
        else:
            self.stdout.write(self.style.SUCCESS("No stuck messages found"))
