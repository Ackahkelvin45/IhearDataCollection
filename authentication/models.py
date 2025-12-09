from django.contrib.auth.models import AbstractUser
from django.db import models
import random
import string
from datetime import timedelta
from django.utils import timezone

import logging

logger = logging.getLogger(__name__)


def random_string(length):
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


class CustomUser(AbstractUser):
    class  UserType(models.TextChoices):
        CONTRIBUTOR = 'contributor'
        RESEARCHER = 'researcher'
        SUPER_ADMIN = 'super_admin'

    
    phone_number = models.CharField(max_length=20, blank=True, null=True)
    first_name = models.CharField(max_length=30, blank=True, null=True)
    last_name = models.CharField(max_length=30, blank=True, null=True)
    speaker_id = models.CharField(max_length=20, blank=True, null=True, unique=True)
    user_type = models.CharField(max_length=20, blank=True, null=True, choices=UserType.choices)
    is_verified = models.BooleanField(default=False)
    

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


class UserOTP(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    otp = models.CharField(max_length=6)
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()

    def __str__(self):
        return f"{self.user.email} - {self.otp}"
    
    def save(self, *args, **kwargs):
        if not self.expires_at:
            self.expires_at = timezone.now() + timedelta(minutes=5)
        super().save(*args, **kwargs)