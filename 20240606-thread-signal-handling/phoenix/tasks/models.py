from django.db import models


class Birds(models.Model):
    name = models.CharField(max_length=128, primary_key=True, unique=True)
    date_added = models.DateTimeField(auto_now_add=True)
    speed = models.IntegerField(default=30)
    emoji = models.CharField(max_length=16, null=True)

    def __str__(self):
        return f"{self.emoji} {self.name:<8} (flying at {self.speed:>3} km/h)"

    class Meta:
        db_table = "birds"
        get_latest_by = "date_added"
