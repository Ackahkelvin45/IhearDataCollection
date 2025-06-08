from django.contrib.auth.models import AbstractUser
from django.db import models
from django.core.mail import send_mail
from django.conf import settings
import random
import string
from django.dispatch import receiver
from django.db.models.signals import post_save
import threading

def random_string(length):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

class CustomUser(AbstractUser):
    phone_number = models.CharField(max_length=20, blank=True, null=True)
    first_name = models.CharField(max_length=30, blank=True, null=True)
    last_name = models.CharField(max_length=30, blank=True, null=True)
    speaker_id = models.CharField(max_length=20, blank=True, null=True, unique=True)
    
    # Make username and password not required at the model level
    username = models.CharField(
        max_length=150,
        unique=True,
        blank=True,
        null=True,
        help_text='Optional. If not provided, one will be generated.',
    )
    password = models.CharField(
        max_length=128,
        blank=True,
        null=True,
        help_text='Optional. If not provided, a random password will be generated.',
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._temp_password = None  # Store temp password as instance variable
    
    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"
    
    def __str__(self):
        return f"{self.first_name} {self.last_name}-{self.speaker_id}"

    def save(self, *args, **kwargs):
        is_new_user = not self.pk
        
        # Generate username if it doesn't exist
        if not self.username:
            if self.email:
                self.username = self.email.split('@')[0]
            else:
                self.username = f"user_{random_string(8)}"
        
        # Ensure username is unique
        if is_new_user:
            original_username = self.username
            counter = 1
            while CustomUser.objects.filter(username=self.username).exists():
                self.username = f"{original_username}_{counter}"
                counter += 1
        
        # Generate speaker_id if it doesn't exist
        if not self.speaker_id:
            initials = ''
            if self.first_name:
                initials += self.first_name[0].upper()
            if self.last_name:
                initials += self.last_name[0].upper()
            
            random_str = random_string(6).upper()
            self.speaker_id = f"{initials}{random_str}"
            
        # Generate random password if it's a new user and password isn't set
        if is_new_user and not self.password:
            temp_password = random_string(12)
            self.set_password(temp_password)
            self._temp_password = temp_password  # Store for signal
            
        super().save(*args, **kwargs)

    class Meta:
        verbose_name = 'Custom User'
        verbose_name_plural = 'Custom Users'






def send_welcome_email_async(user_email, user_first_name, username, speaker_id, password):
    """Function to send welcome email in a separate thread"""
    subject = 'Welcome to Our Platform'
    message = f"""
    Hello {user_first_name or 'User'},

    Your account has been successfully created!

    Here are your login details:
    Username: {username}
    User ID: {speaker_id}
    Password: {password}

    Please change your password after first login.

    Best regards,
    The Team
    """
    
    try:
        send_mail(
            subject=subject,
            message=message,
            from_email=settings.EMAIL_HOST_USER,
            recipient_list=[user_email],
            fail_silently=False,
        )
        print(f"Welcome email sent successfully to {user_email}")
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to send welcome email to {user_email}: {str(e)}")
        print(f"Failed to send email: {str(e)}")

@receiver(post_save, sender=CustomUser)
def user_created_handler(sender, instance, created, **kwargs):
    if created and instance.email and hasattr(instance, '_temp_password') and instance._temp_password:
        print(f"Sending welcome email to {instance.email}")
        
        email_thread = threading.Thread(
            target=send_welcome_email_async,
            args=(
                instance.email,
                instance.first_name,
                instance.username,
                instance.speaker_id,
                instance._temp_password
            ),
            daemon=True
        )
        email_thread.start()
    elif created:
        print(f"User created but email not sent - Email: {instance.email}, Has temp password: {hasattr(instance, '_temp_password')}")