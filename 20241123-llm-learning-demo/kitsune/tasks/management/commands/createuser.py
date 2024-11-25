from django.contrib.auth.models import User
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Creates a test user for the Kitsune application"

    def handle(self, *args, **kwargs):
        username = "fox"
        email = "fox@foxy.com"
        password = "fox"

        if User.objects.filter(username=username).exists():
            self.stdout.write(self.style.WARNING(f'User "{username}" already exists'))
        else:
            User.objects.create_user(username=username, email=email, password=password)
            self.stdout.write(self.style.SUCCESS(f'Successfully created user "{username}"'))
