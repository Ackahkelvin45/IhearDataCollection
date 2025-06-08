from django.shortcuts import render
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.views.decorators.csrf import csrf_protect

# Create your views here.





@csrf_protect
def login_view(request):
    if request.user.is_authenticated:
        return redirect('data:dashboard')  
    
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            return redirect('data:dashboard')  
        else:
            messages.error(request, 'Invalid username or password.')
    
    return render(request, 'authentication/authentication.html')  

@login_required
def logout_view(request):
    logout(request)
    messages.success(request, 'You have been logged out successfully.')
    return redirect('auth:login')  