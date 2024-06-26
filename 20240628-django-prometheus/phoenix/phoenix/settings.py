"""
Django settings for phoenix project.

Generated by 'django-admin startproject' using Django 5.0.4.

For more information on this file, see
https://docs.djangoproject.com/en/5.0/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/5.0/ref/settings/
"""

import os
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/5.0/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = "django-insecure-@y@-65-qy+mnfl15g$ic_+@gthzmd7eqt_1&s*506m&g9&8s8o"

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []


# Application definition

INSTALLED_APPS = [
    # Own app
    "tasks.apps.TasksConfig",
    # Celery
    # use django ORM for results. Define a single model:
    # django_celery_results.models.TaskResult
    "django_celery_results",
    # monitor tasks in django admin interface
    "django_celery_beat",
    #
    "channels",
    # Defaults
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django_prometheus",  # Prometheus
]

MIDDLEWARE = [
    "django_prometheus.middleware.PrometheusBeforeMiddleware",  # Prometheus
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "django_prometheus.middleware.PrometheusAfterMiddleware",  # Prometheus
]

ROOT_URLCONF = "phoenix.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "phoenix.wsgi.application"


# Database
# https://docs.djangoproject.com/en/5.0/ref/settings/#databases

DATABASES = {
    "default": {
        # "ENGINE": "django.db.backends.sqlite3",
        # Use django_prometheus to add metrics to the db
        "ENGINE": "django_prometheus.db.backends.sqlite3",
        # Use on-disk db
        "NAME": BASE_DIR / "db.sqlite3",
        # Only works in Django 5.1+, scheduled for August 2024, meanwhile run
        # those separately:
        # sqlite3 db.sqlite3 'PRAGMA journal_mode = WAL;'
        # sqlite3 db.sqlite3 'PRAGMA synchronous = normal;'
        # sqlite3 db.sqlite3 'PRAGMA journal_size_limit = 6144000;'
        # sqlite3 db.sqlite3 'PRAGMA page_size = 32768;'
        # https://phiresky.github.io/blog/2020/sqlite-performance-tuning/
        # https://blog.pecar.me/django-sqlite-benchmark
        # https://old.reddit.com/r/django/comments/1alxsjb/benchmarking_different_sqlite_options_with_django/
        # "OPTIONS": {
        #     "init_command": "PRAGMA journal_mode=WAL;",
        #     "transaction_mode": "IMMEDIATE",
        # },
        #
        # https://docs.djangoproject.com/en/5.0/ref/databases/#general-notes
        # Keep persistent connextions to the db, instead of starting it on each request. This isn't
        # the default for historical reasons apparently.
        "CONN_MAX_AGE": None,
        "CONN_HEALTH_CHECKS": True,
    }
}


# Password validation
# https://docs.djangoproject.com/en/5.0/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]


# Internationalization
# https://docs.djangoproject.com/en/5.0/topics/i18n/

LANGUAGE_CODE = "en-us"

TIME_ZONE = "UTC"

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/5.0/howto/static-files/

STATIC_URL = "static/"

# Default primary key field type
# https://docs.djangoproject.com/en/5.0/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# For celery, starts with "CELERY_" and then the uppercase version of those options:
# https://docs.celeryq.dev/en/stable/userguide/configuration.html
# CELERY_TIMEZONE = "Australia/Tasmania"
# CELERY_TASK_TRACK_STARTED = True
# CELERY_TASK_TIME_LIMIT = 30 * 60
# CELERY_BROKER_URL = "amqp://guest:guest@localhost:5672//"
# CELERY_RESULT_BACKEND = "rpc://"
CELERY_BROKER_URL = "redis://localhost:6379/0"
CELERY_RESULT_BACKEND = "redis://localhost:6379/0"
# Default to 4, this is just nicer for testing
CELERY_WORKER_PREFETCH_MULTIPLIER = 1
# Use django ORM for results
# https://docs.celeryq.dev/en/stable/django/first-steps-with-django.html#django-celery-results-using-the-django-orm-cache-as-a-result-backend
CELERY_RESULT_BACKEND = "django-db"
CELERY_CACHE_BACKEND = "django-cache"
# Use django specific scheduler
CELERY_BEAT_SCHEDULER = "django_celery_beat.schedulers:DatabaseScheduler"

# Logging
# https://docs.djangoproject.com/en/5.0/topics/logging/#configuring-logging
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
        },
        # "celery": {
        #     "level": "WARNING",
        #     "class": "logging.StreamHandler",
        # },
    },
    "loggers": {
        "django": {
            "handlers": ["console"],
            "level": os.getenv("DJANGO_LOG_LEVEL", "WARNING").upper(),
            "propagate": False,
        },
        "phoenix": {
            "handlers": ["console"],
            "level": os.getenv("PHOENIX_LOG_LEVEL", "DEBUG").upper(),
            "propagate": True,
        },
        ###
        # "celery": {
        #     "handlers": ["celery", "console"],
        #     "level": "WARNING",
        #     "propagate": False,
        # },
        ##
    },
}

# Redis channels
CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels_redis.core.RedisChannelLayer",
        "CONFIG": {
            "hosts": [("localhost", 6379)],
        },
    },
}
ASGI_APPLICATION = "phoenix.asgi.application"

# Common prefix for the metrics
PROMETHEUS_METRIC_NAMESPACE = "phoenix"
