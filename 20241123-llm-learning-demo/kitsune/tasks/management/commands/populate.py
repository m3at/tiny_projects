from django.core.management.base import BaseCommand
from tasks.models import Chapter, Lesson, Question

from .content_samples import LESSON_CONTENT


class Command(BaseCommand):
    help = "Populates the database with test lessons from content_samples.py"

    def handle(self, *args, **kwargs):
        for lesson_idx, lesson_data in enumerate(LESSON_CONTENT.values()):
            # Create the lesson
            lesson = Lesson.objects.create(
                title=lesson_data["title"],
                summary=lesson_data["summary"],
            )

            # Create the chapters
            chapters = [
                Chapter(
                    lesson=lesson,
                    order=chapter_idx + 1,
                    title=chapter_data["title"],
                    content=chapter_data["content"],
                )
                for chapter_idx, chapter_data in enumerate(lesson_data["chapters"])
            ]
            Chapter.objects.bulk_create(chapters)

            # Create the questions
            questions = [
                Question(
                    lesson=lesson,
                    order=question_idx + 1,
                    text=question_data["text"],
                )
                for question_idx, question_data in enumerate(lesson_data["questions"])
            ]
            Question.objects.bulk_create(questions)

        self.stdout.write("Database populated with lessons, chapters, and questions from content_samples.py")
