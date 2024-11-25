import logging
from random import randint
from time import sleep
from typing import Literal

from django.template import Context, Template

from tasks.models import Chapter, Lesson, Question

logger = logging.getLogger("kitsune." + __file__)


def get_lorem(count: int = 3, random: bool = False, method: Literal["p", "w"] = "p") -> str:
    """Generate words or html paragraphs of lorem text."""
    t = Template(f"{{% lorem {count} {method} {'random' if random else ''} %}}")
    return t.render(Context({}))


def generate_lesson(
    title: str,
    description: str,
    nb_chapters: int | None = None,
    nb_questions: int | None = None,
) -> None:
    """Generate a complete lesson.

    Dummy content during developement, will be replaced with real lessons later, but the interface and i/o will be kept.
    """
    logger.debug(f"Strating lesson creation for: {title}")

    # TODO: call a LLM to generate lessons instead. Look at `content_samples.py` for some examples.

    python_lesson = Lesson.objects.create(title=title, summary=description)

    # If not provided, random number of questions
    nb_chapters = nb_chapters or randint(3, 6)
    nb_questions = nb_questions or randint(2, 3)

    # Common dummy content
    fixed_content = get_lorem(5, False, "p")

    chapters = []
    questions = []

    for i in range(nb_chapters):
        chapter_title = get_lorem(4, True, "w")
        chapters.append(
            Chapter(
                lesson=python_lesson,
                order=i + 1,
                title=f"[Chapter {i + 1}] {chapter_title}",
                content=fixed_content,
            )
        )

    for i in range(nb_questions):
        chapter_title = get_lorem(4, True, "w")
        questions.append(
            Question(
                lesson=python_lesson,
                order=i + 1,
                text=get_lorem(12, True, "w"),
            )
        )

    # Random sleep to simulate a time consuming process
    sleep(randint(5, 30))

    Chapter.objects.bulk_create(chapters)
    Question.objects.bulk_create(questions)

    logger.info(f"Created lessons with lorem ipsum content, {nb_chapters=}, {nb_questions=}")
