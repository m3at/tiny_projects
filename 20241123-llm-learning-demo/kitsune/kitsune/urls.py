from django.contrib import admin
from django.urls import path
from tasks import views

urlpatterns = [
    path("admin/", admin.site.urls),
    # path("favicon.ico", views.favicon),
    path("", views.home, name="home"),
    path("request-lesson/", views.request_lesson, name="request_lesson"),
    path("lesson/<int:lesson_id>/", views.lesson_detail, name="lesson_detail"),
    path("lesson/<int:lesson_id>/quiz/", views.lesson_quiz, name="lesson_quiz"),
    path("get-pending-lessons/", views.get_pending_lessons, name="get_pending_lessons"),
]
