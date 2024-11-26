from django.contrib.auth.models import User
from django.db import models


class Lesson(models.Model):
    title = models.CharField(max_length=512)
    summary = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]


class Chapter(models.Model):
    lesson = models.ForeignKey(Lesson, related_name="chapters", on_delete=models.CASCADE)
    title = models.CharField(max_length=512)
    content = models.TextField()
    order = models.IntegerField(default=0)

    class Meta:
        ordering = ["order"]


class UserProgress(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    lesson = models.ForeignKey(Lesson, on_delete=models.CASCADE)
    current_chapter = models.ForeignKey(Chapter, on_delete=models.SET_NULL, null=True)
    completed = models.BooleanField(default=False)
    last_accessed = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ["user", "lesson"]


class Question(models.Model):
    lesson = models.ForeignKey(Lesson, related_name="questions", on_delete=models.CASCADE)
    text = models.TextField()
    answer = models.TextField()
    order = models.IntegerField(default=0)

    class Meta:
        ordering = ["order"]


class LessonRequest(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    description = models.TextField()
    title = models.CharField(max_length=512, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(
        max_length=20,
        choices=[
            ("pending", "Pending"),
            ("creating", "Creating"),
            ("approved", "Approved"),
            ("rejected", "Rejected"),
        ],
        default="pending",
    )
