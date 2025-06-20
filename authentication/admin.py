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
    password = forms.CharField(
        label="Password",
        widget=forms.PasswordInput(attrs={
            'autocomplete': 'new-password',
            'class': 'vTextField'  # Add Unfold's text field class
        }),
        strip=False,
        required=False,
        help_text="Raw passwords are not stored, so there is no way to see this user's password, but you can change the password using <a href=\"../password/\">this form</a> or by entering a new password here."
    )

    class Meta(UserChangeForm.Meta):
        model = CustomUser
        fields = '__all__'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        password = self.fields.get('password')
        if password:
            password.help_text = password.help_text.format('../password/')
            # Ensure the password field has the proper Unfold styling
            password.widget.attrs.update({
                'class': 'vTextField',
                'placeholder': 'Enter new password'
            })

from unfold.admin import ModelAdmin

@admin.register(CustomUser)
class CustomUserAdmin(ModelAdmin, UserAdmin):
    form = CustomUserChangeForm
    add_form = CustomUserCreationForm
    
    list_display = ('email', 'first_name', 'last_name', 'speaker_id', 'is_staff')
    list_filter = ('is_staff', 'is_superuser', 'is_active', 'groups')
    search_fields = ('email', 'first_name', 'last_name', 'speaker_id')
    ordering = ('email',)
    
    fieldsets = (
        (None, {'fields': ('email', 'username', 'password')}),
        ('Personal info', {'fields': ('first_name', 'last_name', 'phone_number', 'speaker_id', 'unhashed_password')}),
        ('Permissions', {
            'fields': ('is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions'),
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
        defaults = {}
        if obj is None:
            defaults['form'] = self.add_form
        defaults.update(kwargs)
        return super().get_form(request, obj, **defaults)
    
    def save_model(self, request, obj, form, change):
        if not change:  # Only for new users
            temp_password = random_string(12)
            obj.unhashed_password = temp_password
            obj.set_password(temp_password)
            obj._temp_password = temp_password
        else:
            # Handle password change for existing users
            if 'password' in form.changed_data and form.cleaned_data['password']:
                obj.set_password(form.cleaned_data['password'])
                obj.unhashed_password = form.cleaned_data['password']
        super().save_model(request, obj, form, change)
    
    def get_urls(self):
        from django.urls import path
        urls = super().get_urls()
        custom_urls = [
            path(
                '<id>/password/',
                self.admin_site.admin_view(self.user_change_password),
                name='auth_user_password_change',
            ),
        ]
        return custom_urls + urls
    
    def user_change_password(self, request, id, form_url=''):
        from django.contrib.auth.forms import AdminPasswordChangeForm
        user = self.get_object(request, id)
        if request.method == 'POST':
            form = AdminPasswordChangeForm(user, request.POST)
            if form.is_valid():
                form.save()
                # Update the unhashed_password field
                user.unhashed_password = form.cleaned_data['password1']
                user.save()
                from django.contrib import messages
                messages.success(request, 'Password changed successfully')
                from django.urls import reverse
                from django.http import HttpResponseRedirect
                return HttpResponseRedirect(
                    reverse(
                        '%s:%s_%s_change' % (
                            self.admin_site.name,
                            user._meta.app_label,
                            user._meta.model_name,
                        ),
                        args=(user.pk,),
                    )
                )
        else:
            form = AdminPasswordChangeForm(user)
        
        context = {
            'title': 'Change password',
            'form': form,
            'form_url': form_url,
            'user': user,
            'opts': self.model._meta,
            'has_permission': True,
        }
        return self.render_change_password_form(request, context, form)
        
    def render_change_password_form(self, request, context, form):
        from django.template.response import TemplateResponse
        # Use Unfold's base template
        return TemplateResponse(
            request,
            'admin/auth/user/change_password.html',
            context,
        )