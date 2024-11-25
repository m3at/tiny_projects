import logging

from celery import shared_task

from .content_generator import generate_lesson
from .models import LessonRequest

logger = logging.getLogger("kitsune." + __file__)


@shared_task
def debug_task():
    logger.debug("this is a dummy task")


@shared_task
def create_lesson_task(lesson_request_id):
    try:
        lesson_request = LessonRequest.objects.get(id=lesson_request_id)
        lesson_request.status = "creating"
        lesson_request.save()

        generate_lesson(title=lesson_request.title, description=lesson_request.description)

        lesson_request.status = "approved"
        lesson_request.save()
        logger.info(f"Lesson created successfully: {lesson_request.title}")
    except Exception as e:
        logger.error(f"Error creating lesson: {str(e)}")
        lesson_request.status = "rejected"
        lesson_request.save()
