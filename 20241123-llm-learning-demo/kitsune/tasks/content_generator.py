import logging
from random import randint
from time import sleep
from typing import Literal

from django.template import Context, Template
from openai import OpenAI
from pydantic import BaseModel

from kitsune.settings import p_env
from tasks.models import Chapter, Lesson, Question

logger = logging.getLogger("kitsune." + __file__)


def get_lorem(count: int = 3, random: bool = False, method: Literal["p", "w"] = "p") -> str:
    """Generate words or html paragraphs of lorem text."""
    t = Template(f"{{% lorem {count} {method} {'random' if random else ''} %}}")
    return t.render(Context({}))


def generate_lesson(
    description: str,
    title: str | None,
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
    # sleep(randint(5, 30))
    sleep(randint(1, 3))

    Chapter.objects.bulk_create(chapters)
    Question.objects.bulk_create(questions)

    logger.info(f"Created lessons with lorem ipsum content, {nb_chapters=}, {nb_questions=}")


class ResponseChapter(BaseModel):
    title: str
    html_content: str


class ResponseQuestion(BaseModel):
    text: str
    example_answer: str


class ExpectedResponse(BaseModel):
    short_title: str
    summary: str
    chapters: list[ResponseChapter]
    questions: list[ResponseQuestion]


def generate_lesson_openai(
    description: str,
    title: str | None,
    nb_chapters: int | None = None,
    nb_questions: int | None = None,
) -> None:
    logger.debug(f"Strating lesson creation for: {title}")

    client = OpenAI(api_key=p_env.OPENAI_API_KEY.get_secret_value())

    # TODO: filter the initial request first to assert if it's likely to pass the safety filters

    if len(description) > 1024:
        raise ValueError(f"Too long: {len(description)}")

    # _min_c, _max_c = 2, 4
    _min_c, _max_c = 3, 6

    # description = "I want to learn about the end of the Tokugawa shogunate, especially right before, during, and right after the Perry expedition!"

    _model = "gpt-4o-2024-08-06"
    logger.debug(f"Calling model: {_model}")

    completion = client.beta.chat.completions.parse(
        model=_model,
        messages=[
            {
                "role": "system",
                "content": (
                    f"""\
    You are a dedicated teacher, providing lessons on demand on subjects requested by the user. The target audience are educated adults.
    For the request below, you will write at least {_min_c} chapters, and no more than {_max_c}. Aim for about 12 paragraphs per chapter.

    You will also prepare:
    * a summary of the content, sparking the student's interest
    * 3 quiz questions to help the student self-check their understanding

    For the lesson content, use HTML markup like <blockquote>, <strong>, <em> or <ruby> when appropriate. For foreign language words, \
    show them in their original language when relevant, at least the first time they are seen.
    """
                ),
            },
            {"role": "user", "content": f"[<USER REQUESTED LESSON>]\n{description}"},
        ],
        response_format=ExpectedResponse,
        timeout=60,
    )

    r: ExpectedResponse | None = completion.choices[0].message.parsed
    if r is None:
        logger.error("OpenAI returned None. This is a sad day.")
        return

    # TODO: keep track of cost and other misc response infos like timing
    logger.debug("Successful response. Trying to coerce into db objects.")

    # Save to the db
    lesson = Lesson.objects.create(
        # Use the user privded title if it exists, otherwise the LLM generated one
        # title=title or r.short_title,
        title=r.short_title,
        summary=r.summary,
    )

    chapters = []
    questions = []
    for i, c in enumerate(r.chapters):
        chapters.append(
            Chapter(
                lesson=lesson,
                order=i + 1,
                title=c.title,
                content=c.html_content,
            )
        )

    for i, q in enumerate(r.questions):
        questions.append(
            Question(
                lesson=lesson,
                order=i + 1,
                text=q.text,
                answer=q.example_answer,
            )
        )

    Chapter.objects.bulk_create(chapters)
    Question.objects.bulk_create(questions)

    logger.info(f"Created lessons, {len(r.chapters)=}, {len(r.questions)=}")
