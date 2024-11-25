from django.contrib import admin

from tasks.models import Chapter, Lesson, LessonRequest, Question, UserProgress

admin.site.register(Lesson)
admin.site.register(Chapter)
admin.site.register(UserProgress)
admin.site.register(Question)
admin.site.register(LessonRequest)

# @admin.register(LessonRequest)
# class LessonRequestAdmin(admin.ModelAdmin):
#     readonly_fields = ["created_at", "title"]
