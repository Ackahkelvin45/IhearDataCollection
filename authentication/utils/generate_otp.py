import random
import string
from .send_email import generic_send_mail
from django.utils import timezone
from datetime import timedelta
from ..models import CustomUser, UserOTP


def generate_otp():
    """Generate a 6-digit OTP"""
    return "".join(random.choices(string.digits, k=6))


def send_otp(user):
    """Generate and send OTP to user via email"""
    otp = generate_otp()
    expires_at = timezone.now() + timedelta(minutes=5)

    # Delete any existing OTPs for this user first
    UserOTP.objects.filter(user=user).delete()

    # Create new UserOTP record
    user_otp = UserOTP.objects.create(user=user, otp=otp, expires_at=expires_at)

    # Explicitly save to ensure it's persisted
    user_otp.save()

    # Prepare email payload
    user_name = user.get_full_name() or user.first_name or user.email.split("@")[0]
    payload = {
        "otp": otp,
        "user_name": user_name,
    }

    # Send email
    result = generic_send_mail(user.email, "OTP Verification - I Hear Dataset", payload)

    return otp, result


def verify_otp(user, otp_code):
    try:
        user_otp = UserOTP.objects.filter(user=user, otp=otp_code).latest("created_at")

        # Check if OTP has expired
        if user_otp.expires_at < timezone.now():
            return False, "OTP has expired. Please request a new one."

        user.is_verified = True
        user.save()

        # OTP is valid
        return True, "OTP verified successfully"
    except UserOTP.DoesNotExist:
        return False, "Invalid OTP. Please check and try again."
