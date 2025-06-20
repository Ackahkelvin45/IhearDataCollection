"""
Django settings for datacollection project.

Generated by 'django-admin startproject' using Django 4.2.20.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/4.2/ref/settings/
"""

from pathlib import Path
from dotenv import load_dotenv

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
import os
load_dotenv()
import logging

logger = logging.getLogger(__name__)



# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/4.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-*_^$!#mp28jhe25iq6dok5suz(_!529k-c2hj#wnd!7@6@p%ri'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True
ALLOWED_HOSTS = ['localhost','178.128.165.1', '127.0.0.1','www.ihearandsee-at-rail.com','ihearandsee-at-rail.com','https://www.ihearandsee-at-rail.com','www.google.com']
 

# Application definition

INSTALLED_APPS = [
      # optional, if special form elements are needed
      'storages',
       "unfold",  # before django.contrib.admin
    "unfold.contrib.filters",  # optional, if special filters are needed
    "unfold.contrib.forms",  # optional, if special form elements are needed
    "unfold.contrib.inlines",  # optional, if special inlines are needed
    "unfold.contrib.import_export",  # optional, if django-import-export package is used
    "unfold.contrib.guardian",  # optional, if django-guardian package is used
    "unfold.contrib.simple_history",  # optional, if django-simple-history package is used
    "django.contrib.admin",
     'tailwind',
  'theme',
    
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    "data",
     'core',
    'authentication',
      "datacollection",
        

    


   
    
]
TAILWIND_APP_NAME = 'theme'


MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]





ROOT_URLCONF = 'datacollection.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'datacollection.wsgi.application'


# Database
# https://docs.djangoproject.com/en/4.2/ref/settings/#databases

# settings.py
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.getenv('POSTGRES_DB'),
        'USER': os.getenv('POSTGRES_USER'),
        'PASSWORD': os.getenv('POSTGRES_PASSWORD'),
        'HOST': 'db',  # This matches the service name in docker-compose
        'PORT': '5432',
    }
}
# Password validation
# https://docs.djangoproject.com/en/4.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/4.2/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/4.2/howto/static-files/


# Default primary key field type
# https://docs.djangoproject.com/en/4.2/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

AUTH_USER_MODEL = 'authentication.CustomUser'




def as_bool(value: str):
    if value is None:
        return False
    return value.lower() in ["true", "yes", "1", "y"]

# Email Configuration
EMAIL_BACKEND = os.getenv("EMAIL_BACKEND", "django.core.mail.backends.smtp.EmailBackend")
EMAIL_HOST = os.getenv("EMAIL_HOST", "smtp.gmail.com")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", 587))
EMAIL_USE_TLS = os.getenv("EMAIL_USE_TLS", "True") == "True"
EMAIL_USE_SSL = os.getenv("EMAIL_USE_SSL", "False") == "True"
EMAIL_HOST_USER = os.getenv("EMAIL_HOST_USER")
EMAIL_HOST_PASSWORD = os.getenv("EMAIL_HOST_PASSWORD")


# Ensure only one of USE_TLS or USE_SSL is True
if EMAIL_USE_TLS and EMAIL_USE_SSL:
    EMAIL_USE_SSL = False
    logger.warning("Both EMAIL_USE_TLS and EMAIL_USE_SSL were True. Disabling SSL in favor of TLS.")

# Temporarily add this before your email settings to debug

# Authentication settings
LOGIN_URL = '/auth/login/'  # URL to redirect to for login
LOGIN_REDIRECT_URL = '/'  # Where to redirect after login
LOGOUT_REDIRECT_URL = '/auth/login/'





# At the top of your settings.py file, add this import
from django.templatetags.static import static


# Alternative approach without lambda (simpler):
UNFOLD = {
     "SITE_TITLE": "I hear Dataset Admin Portal",
     "SITE_HEADER": "I hear Dataset",
     
     "DARK_MODE": True,
     "LOGIN": {
         "image": "/static/assets/img/image.png",
         "title": "Welcome to I hear Dataset Admin Portal",
       "description": "Please enter your credentials",
     },
     "SIDEBAR": {
         "show_search": True,
         "show_all_applications": False,
     },
     
 }

CORS_ALLOWED_ORIGINS = [
    "http://localhost",  # Your frontend URL
    "http://127.0.0.10",
    "http://0.0.0.0",
    "https://shalom-enterprise-1.onrender.com",
    'iheardatacollection.onrender.com',
  'https://iheardatacollection.onrender.com',
'178.128.165.1',
'www.ihearandsee-at-rail.com',
'ihearandsee-at-rail.com',
'https://www.ihearandsee-at-rail.com',
'www.google.com'

    ]


CORS_ALLOW_CREDENTIALS = True  # To allow cookies


CSRF_TRUSTED_ORIGINS = [
  'https://iheardatacollection.onrender.com',    # Add other trusted origins if needed
    'http://127.0.0.1',
    "http://127.0.0.1:8000",
    'http://localhost',
    'http://0.0.0.0',
'http://178.128.165.1',
'https://www.ihearandsee-at-rail.com',
'https://https//www.google.com/'


        ]




# DigitalOcean Spaces Settings
AWS_ACCESS_KEY_ID = os.getenv('DO_SPACES_KEY')
AWS_SECRET_ACCESS_KEY = os.getenv('DO_SPACES_SECRET')
AWS_STORAGE_BUCKET_NAME = os.getenv('DO_SPACES_BUCKET')
AWS_S3_ENDPOINT_URL = 'https://lon1.digitaloceanspaces.com'

STORAGES = {
    "default": {
        "BACKEND": "storages.backends.s3boto3.S3Boto3Storage",
        "OPTIONS": {
            "access_key": AWS_ACCESS_KEY_ID,
            "secret_key": AWS_SECRET_ACCESS_KEY,
            "bucket_name": AWS_STORAGE_BUCKET_NAME,
            "endpoint_url": AWS_S3_ENDPOINT_URL,
            "location": "media",
            "default_acl": "public-read",
            "object_parameters": {"CacheControl": "max-age=86400"},
        },
    },
    "staticfiles": {
        "BACKEND": "storages.backends.s3boto3.S3Boto3Storage",
        "OPTIONS": {
            "access_key": AWS_ACCESS_KEY_ID,
            "secret_key": AWS_SECRET_ACCESS_KEY,
            "bucket_name": AWS_STORAGE_BUCKET_NAME,
            "endpoint_url": AWS_S3_ENDPOINT_URL,
            "location": "static",
            "default_acl": "public-read",
            "object_parameters": {"CacheControl": "max-age=86400"},
        },
    },
}

STATIC_URL = f"{AWS_S3_ENDPOINT_URL}/{AWS_STORAGE_BUCKET_NAME}/static/"
MEDIA_URL = f"{AWS_S3_ENDPOINT_URL}/{AWS_STORAGE_BUCKET_NAME}/media/"
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, 'static'),
]