from django.core.management.base import BaseCommand
from data_insights.models import ChatMessage

class Command(BaseCommand):
    help = 'Fix messages that are stuck in processing status'

    def handle(self, *args, **options):
        # Find messages stuck in processing status
        stuck_messages = ChatMessage.objects.filter(
            status=ChatMessage.MessageStatus.PROCESSING
        )
        
        count = stuck_messages.count()
        self.stdout.write(f"Found {count} messages stuck in processing status")
        
        if count > 0:
            # Update them to failed status with a helpful message
            updated = stuck_messages.update(
                status=ChatMessage.MessageStatus.FAILED,
                assistant_response="This message was interrupted during processing. Please try sending your question again."
            )
            
            self.stdout.write(
                self.style.SUCCESS(f'Successfully fixed {updated} stuck messages')
            )
        else:
            self.stdout.write(
                self.style.SUCCESS('No stuck messages found')
            )
