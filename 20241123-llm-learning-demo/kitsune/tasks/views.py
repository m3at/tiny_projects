from django.contrib.auth import login
from django.contrib.auth.models import User
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render

from .celery_tasks import create_lesson_task
from .models import Lesson, LessonRequest, UserProgress


def auto_login(view_func):
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            # Automatically log in as testuser
            user = User.objects.get(username="fox")
            login(request, user)
        return view_func(request, *args, **kwargs)

    return wrapper


@auto_login
def home(request):
    user_progress = UserProgress.objects.filter(user=request.user).select_related("lesson")
    ongoing_lessons = [up for up in user_progress if not up.completed]
    completed_lessons = [up for up in user_progress if up.completed]
    available_lessons = Lesson.objects.exclude(id__in=[up.lesson_id for up in user_progress])
    pending_lessons = LessonRequest.objects.filter(user=request.user).exclude(status="approved")

    return render(
        request,
        "home.html",
        {
            "ongoing_lessons": ongoing_lessons,
            "completed_lessons": completed_lessons,
            "available_lessons": available_lessons,
            "pending_lessons": pending_lessons,
        },
    )


@auto_login
def lesson_detail(request, lesson_id):
    lesson = get_object_or_404(Lesson.objects.prefetch_related("chapters"), id=lesson_id)
    user_progress, created = UserProgress.objects.get_or_create(user=request.user, lesson=lesson)

    if created or not user_progress.current_chapter:
        current_chapter = lesson.chapters.first()
    else:
        current_chapter = user_progress.current_chapter

    chapter_id = request.GET.get("chapter")
    if chapter_id:
        current_chapter = get_object_or_404(lesson.chapters.all(), id=chapter_id)

    user_progress.current_chapter = current_chapter
    user_progress.save()

    next_chapter = lesson.chapters.filter(order__gt=current_chapter.order).first()
    prev_chapter = lesson.chapters.filter(order__lt=current_chapter.order).last()

    return render(
        request,
        "lesson.html",
        {
            "lesson": lesson,
            "current_chapter": current_chapter,
            "next_chapter": next_chapter,
            "prev_chapter": prev_chapter,
            "total_chapters": lesson.chapters.count(),
            "progress": (current_chapter.order / lesson.chapters.count()) * 100,
        },
    )


@auto_login
def lesson_quiz(request, lesson_id):
    lesson = get_object_or_404(Lesson.objects.prefetch_related("questions"), id=lesson_id)
    questions = lesson.questions.all()

    if request.method == "POST":
        # Process quiz answers here if needed
        user_progress = UserProgress.objects.get(user=request.user, lesson=lesson)
        user_progress.completed = True
        user_progress.save()
        return redirect("home")

    return render(request, "quiz.html", {"lesson": lesson, "questions": questions})


@auto_login
def request_lesson(request):
    if request.method == "POST":
        title = request.POST.get("title")
        description = request.POST.get("description")
        if title and description:
            lesson_request = LessonRequest.objects.create(
                user=request.user, title=title, description=description, status="pending"
            )
            create_lesson_task.delay(lesson_request.id)
    return redirect("home")


@auto_login
def get_pending_lessons(request):
    pending_lessons = LessonRequest.objects.filter(user=request.user).exclude(status="approved")
    lessons_data = [
        {
            "id": lesson.id,
            "title": lesson.title,
            "description": lesson.description,
            "status": lesson.get_status_display(),
        }
        for lesson in pending_lessons
    ]
    return JsonResponse({"pending_lessons": lessons_data})


# Handled through WHITENOISE_ROOT instead
# @require_GET
# @cache_control(max_age=60 * 60 * 24, immutable=True, public=True)
# def favicon(request):
#     file = (settings.STATIC_ROOT.parent / "fox.png").open("rb")
#     return FileResponse(file)
