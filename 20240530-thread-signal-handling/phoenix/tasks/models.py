from django.db import models


class Birds(models.Model):
    name = models.CharField(max_length=128, primary_key=True, unique=True)
    date_added = models.DateTimeField(auto_now_add=True)
    flying_speed = models.IntegerField(default=30)
    emoji = models.CharField(max_length=16, default="🪿")

    def __str__(self):
        return f"{self.emoji} {self.name} {self.flying_speed} km/h"
