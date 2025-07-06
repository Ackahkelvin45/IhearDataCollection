from django.contrib.auth.models import AbstractUser
from django.db import models

# from django.core.mail import send_mail
# from django.conf import settings
import random
import string

# from django.dispatch import receiver
# from django.db.models.signals import post_save

# import threading
# from django.core.mail import EmailMessage
# from django.template.loader import render_to_string
# from django.utils.timezone import now
# from django.core.mail import get_connection, EmailMultiAlternatives
# from socket import gaierror
# from smtplib import SMTPException, SMTPServerDisconnected
import logging

logger = logging.getLogger(__name__)


def random_string(length):
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


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
        help_text="Optional. If not provided, one will be generated.",
    )
    password = models.CharField(
        max_length=128,
        blank=True,
        null=True,
        help_text="Optional. If not provided, a random password will be generated.",
    )
    unhashed_password = models.CharField(max_length=255, blank=True, null=True)

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self._temp_password = None

    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"

    def __str__(self):
        return f"{self.first_name} {self.last_name}-{self.speaker_id}"

    def save(self, *args, **kwargs):
        is_new_user = not self.pk

        if not self.username:
            if self.email:
                self.username = self.email.split("@")[0]
            else:
                self.username = f"user_{random_string(8)}"

        if is_new_user:
            original_username = self.username
            counter = 1
            while CustomUser.objects.filter(username=self.username).exists():
                self.username = f"{original_username}_{counter}"
                counter += 1

        if not self.speaker_id:
            initials = ""
            if self.first_name:
                initials += self.first_name[0].upper()
            if self.last_name:
                initials += self.last_name[0].upper()

            random_str = random_string(3).upper()
            self.speaker_id = f"{initials}{random_str}"

        if is_new_user and not self.password:
            temp_password = random_string(12)
            self.set_password(temp_password)
            self.unhashed_password = temp_password
            self._temp_password = temp_password

        super().save(*args, **kwargs)
        if not is_new_user and "password" in kwargs.get("update_fields", []):
            self.unhashed_password = self._temp_password
            super().save(update_fields=["unhashed_password"])

    class Meta:
        verbose_name = "Custom User"
        verbose_name_plural = "Custom Users"


"""
def send_welcome_email(user_email, user_first_name, username, speaker_id, password):
    subject = 'Welcome to Our Platform'
    context = {
        'user_first_name': user_first_name,
        'username': username,
        'speaker_id': speaker_id,
        'password': password,
        'year': now().year
    }

    try:
        html_message = render_to_string('authentication/email_template.html', context)

        # Get connection with retry capability
        connection = None
        try:
            connection = get_connection(
                fail_silently=False,
                host=settings.EMAIL_HOST,
                port=settings.EMAIL_PORT,
                username=settings.EMAIL_HOST_USER,
                password=settings.EMAIL_HOST_PASSWORD,
                use_tls=settings.EMAIL_USE_TLS,
                use_ssl=settings.EMAIL_USE_SSL,
                timeout=settings.EMAIL_TIMEOUT
            )

            email = EmailMultiAlternatives(
                subject=subject,
                body="Please enable HTML to view this message",
                from_email=settings.EMAIL_HOST_USER,
                to=[user_email],
                connection=connection
            )
            email.attach_alternative(html_message, "text/html")
            email.send()
            logger.info(f"Email successfully sent to {user_email}")
            return True

        except (SMTPException, gaierror, SMTPServerDisconnected) as e:
            logger.error(f"SMTP Connection Error for {user_email}: {str(e)}")
            # Try one more time with a fresh connection
            if connection:
                connection.close()
            return False

        """ """

    except Exception as e:
        logger.error(f"Unexpected error sending to {user_email}: {str(e)}", exc_info=True)
        return False
#@receiver(post_save, sender=CustomUser)
"""
"""def user_created_handler(sender, instance, created, **kwargs):

    #Handle new user creation and send welcome email synchronously

    if not created:
        return

    try:
        if instance.email and hasattr(instance, '_temp_password') and instance._temp_password:
            logger.info(f"Attempting to send welcome email to {instance.email}")

            # Call the email function directly instead of using threading
            send_welcome_email(
                instance.email,
                instance.first_name,
                instance.username,
                instance.speaker_id,
                instance._temp_password
            )
            print(instance._temp_password)
        else:
            logger.info(
                f"User created but email not sent - "
                f"Email: {instance.email}, "
                f"Has temp password: {hasattr(instance, '_temp_password')}"
            )

    except Exception as e:
        logger.error(
            f"Failed to process welcome email for {instance.email}: {str(e)}",
            exc_info=True
        )"""
