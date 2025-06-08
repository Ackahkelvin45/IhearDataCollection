# admin.py
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.contrib.auth.forms import UserChangeForm
from django import forms
from .models import CustomUser
import random
import string

def random_string(length):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

class CustomUserCreationForm(forms.ModelForm):
    class Meta:
        model = CustomUser
        fields = ('email', 'first_name', 'last_name', 'phone_number')
    
    def save(self, commit=True):
        user = super().save(commit=False)
        # Password will be set in save_model
        if commit:
            user.save()
        return user

class CustomUserChangeForm(UserChangeForm):
    class Meta(UserChangeForm.Meta):
        model = CustomUser
        fields = '__all__'
from unfold.admin import ModelAdmin

@admin.register(CustomUser)
class CustomUserAdmin(ModelAdmin,UserAdmin):
    form = CustomUserChangeForm
    add_form = CustomUserCreationForm
    
    list_display = ('email', 'first_name', 'last_name', 'speaker_id', 'is_staff')
    list_filter = ('is_staff', 'is_superuser', 'is_active', 'groups')
    search_fields = ('email', 'first_name', 'last_name', 'speaker_id')
    ordering = ('email',)
    
    fieldsets = (
        (None, {'fields': ('email', 'username')}),
        ('Personal info', {'fields': ('first_name', 'last_name', 'phone_number', 'speaker_id',"password")}),
        ('Permissions', {
            'fields': ('is_active', 'is_staff', 'is_superuser',),
        }),
        ('Important dates', {'fields': ('last_login', 'date_joined')}),
    )
    
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('email', 'first_name', 'last_name', 'phone_number'),
        }),
    )

    readonly_fields = ('last_login', 'date_joined', 'speaker_id')
    
    def get_form(self, request, obj=None, **kwargs):
        """
        Use special form during user creation
        """
        defaults = {}
        if obj is None:
            defaults['form'] = self.add_form
        defaults.update(kwargs)
        return super().get_form(request, obj, **defaults)
    
    def save_model(self, request, obj, form, change):
        if not change:  # Only for new users
            # Generate random password
            temp_password = random_string(12)
            obj.set_password(temp_password)
            obj._temp_password = temp_password  # This will be used by the post_save signal
        super().save_model(request, obj, form, change)