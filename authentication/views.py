from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.views.decorators.csrf import csrf_protect
from .utils.generate_otp import send_otp, verify_otp
from .models import CustomUser
from data.models import NoiseDataset

# Create your views here.


@csrf_protect
def login_view(request):
    if request.user.is_authenticated:
        return redirect("data:dashboard")

    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(request, username=username, password=password)

        if user is not None:

            if not user.is_verified:

                otp, email_result = send_otp(user)
                if email_result:
                    request.session["otp_user_id"] = user.id
                    messages.success(
                        request,
                        "An OTP has been sent to your email. Please verify to continue.",
                    )
                    return redirect("auth:verify_otp")
                else:
                    messages.error(request, "Failed to send OTP. Please try again.")
            else:

                login(request, user)
                messages.success(
                    request, "Welcome back! You have been logged in successfully."
                )
                return redirect("data:dashboard")
        else:
            messages.error(request, "Invalid username or password.")

    return render(request, "authentication/authentication.html")


@login_required
def logout_view(request):
    logout(request)
    messages.success(request, "You have been logged out successfully.")
    return redirect("auth:login")


def resend_otp(request):
    user_id = request.session.get("otp_user_id")
    if user_id:
        user = CustomUser.objects.get(id=user_id)
        otp, email_result = send_otp(user)
        if email_result:
            messages.success(
                request, "OTP has been resent to your email. Please verify to continue."
            )
            return redirect("auth:verify_otp")
    else:
        messages.error(request, "User not found.")
        return redirect("auth:login")


def signup_view(request):
    if request.method == "POST":
        email = request.POST.get("email")
        password = request.POST.get("password")
        confirm_password = request.POST.get("confirm_password")
        first_name = request.POST.get("first_name", "")
        last_name = request.POST.get("last_name", "")
        username = request.POST.get("username", "")

        if password != confirm_password:
            messages.error(request, "Passwords do not match.")
            return redirect("auth:signup")
        if CustomUser.objects.filter(email=email).exists():
            messages.error(request, "Email already exists.")
            return redirect("auth:signup")
        if username and CustomUser.objects.filter(username=username).exists():
            messages.error(request, "Username already exists.")
            return redirect("auth:signup")

        if len(password) < 8:
            messages.error(request, "Password must be at least 8 characters long.")
            return redirect("auth:signup")
        if not any(char.isdigit() for char in password):
            messages.error(request, "Password must contain at least one number.")
            return redirect("auth:signup")
        if not any(char.isalpha() for char in password):
            messages.error(request, "Password must contain at least one letter.")
            return redirect("auth:signup")

        try:
            user = CustomUser.objects.create_user(
                email=email,
                username=username if username else None,
                first_name=first_name,
                last_name=last_name,
            )
            user.set_password(password)
            user.user_type = "contributor"
            user.is_verified = False
            user.save()
            otp, email_result = send_otp(user)
            if email_result:
                request.session["otp_user_id"] = user.id
                messages.success(
                    request,
                    "Account created! An OTP has been sent to your email. Please verify to continue.",
                )
                return redirect("auth:verify_otp")
            else:
                messages.error(
                    request,
                    "Account created but failed to send OTP. Please contact support.",
                )
                return redirect("auth:login")
        except Exception as e:
            messages.error(request, f"Error creating account: {str(e)}")
            print(f"Error creating account: {str(e)}")

    return render(request, "authentication/signup.html")


def verify_userotp(request):
    user_id = request.session.get("otp_user_id")
    if user_id:
        try:
            user = CustomUser.objects.get(id=user_id)
        except CustomUser.DoesNotExist:
            messages.error(request, "Session expired. Please login again.")
            return redirect("auth:login")
    elif request.user.is_authenticated:
        user = request.user
    else:
        messages.error(request, "Please login first.")
        return redirect("auth:login")

    if request.method == "POST":
        otp_code = request.POST.get("otp")
        if not otp_code:
            messages.error(request, "Please enter the OTP.")
            return render(request, "authentication/verify_otp.html")

        is_valid, message = verify_otp(user, otp_code)

        if is_valid:
            # Clear the session
            if "otp_user_id" in request.session:
                del request.session["otp_user_id"]
            messages.success(
                request,
                "OTP verified successfully! Please login with your credentials.",
            )
            return redirect("auth:login")
        else:
            messages.error(request, message)

    return render(request, "authentication/verify_otp.html")


@login_required
def profile_view(request):
    user = request.user

    if request.method == "POST":
        # Get form data
        username = request.POST.get("username", "").strip()
        email = request.POST.get("email", "").strip()
        first_name = request.POST.get("first_name", "").strip()
        last_name = request.POST.get("last_name", "").strip()
        phone_number = request.POST.get("phone_number", "").strip()

        # Validate email
        if email and email != user.email:
            if CustomUser.objects.filter(email=email).exclude(id=user.id).exists():
                messages.error(request, "Email already exists.")
                return render(request, "authentication/profile.html", {"user": user})
            user.email = email

        # Validate username
        if username and username != user.username:
            if (
                CustomUser.objects.filter(username=username)
                .exclude(id=user.id)
                .exists()
            ):
                messages.error(request, "Username already exists.")
                return render(request, "authentication/profile.html", {"user": user})
            user.username = username

        # Update other fields
        user.first_name = first_name
        user.last_name = last_name
        user.phone_number = phone_number

        try:
            user.save()
            messages.success(request, "Profile updated successfully!")
            return redirect("auth:profile")
        except Exception as e:
            messages.error(request, f"Error updating profile: {str(e)}")

    # Get total noise data count for the user
    total_noise_data = NoiseDataset.objects.filter(collector=user).count()

    return render(
        request,
        "authentication/profile.html",
        {"user": user, "total_noise_data": total_noise_data},
    )
