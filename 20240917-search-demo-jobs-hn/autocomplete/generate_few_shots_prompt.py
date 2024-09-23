#!/usr/bin/env python3

import argparse
import logging
import sys

from transformers import AutoTokenizer

logger = logging.getLogger("base")


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")

    sample_completions = """\
Software engineer jobs in New York
New York software engineer job openings
software development positions in NYC
software engineer roles in New York City
NYC tech jobs for software engineers

Remote marketing positions
remote marketing job opportunities
work from home marketing roles
digital marketing positions remote
virtual marketing jobs

data sciens job entry level
entry-level data science jobs
junior data scientist positions
data analyst internships
data sciensce entry positions

senior devops engineer san francisco
senior DevOps positions in San Francisco
San Francisco DevOps engineer jobs
lead DevOps roles in SF
senior cloud engineer jobs SF

part time IT jobs near me
part-time IT support roles nearby
IT technician part-time jobs
flexible IT jobs in my area
local part-time technology positions

internships for computer science students
computer science internships available
paid internships for CS students
tech internships for college students
software internships for beginners

jobs in AI and machine learning
AI job opportunities in tech
machine learning engineer positions
artificial intelligence career openings
jobs in AI research and development

hiring ux designers now
urgent UX designer positions
UX/UI design jobs hiring immediately
user experience designer job openings
remote UX design jobs available

java developer work from home
remote Java developer positions
Java programming jobs from home
work from home Java software engineer
telecommute Java developer roles

project manager tech startup
project management jobs in tech startups
startup project manager openings
agile project manager positions in tech
tech startup PM job opportunities"""

    # 5 examples of 3 augmentations each seems enough (form a feeling)
    nb_augs = 3
    nb_examples = 5

    messages = [
        {
            "role": "system",
            "content": f"You are a query augmentation engine, part of a job board. You augment queries to increase the chances of a search match. You write exactly {nb_augs} augmentations.",
        },
    ]

    for row in sample_completions.split("\n\n")[:nb_examples]:
        query, *rest = row.split("\n")
        rest = rest[:nb_augs]
        messages.extend(
            [
                {"role": "user", "content": query},
                {"role": "assistant", "content": "\n".join(rest)},
            ]
        )

    messages.append({"role": "user", "content": "'$CONTENT'"})

    print(
        tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        .encode("unicode_escape")
        .decode("utf-8"),
        end="",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="generate few shots prompt",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="DEBUG",
        choices=["debug", "info", "warning", "error"],
        help="Logging level",
    )
    args = vars(parser.parse_args())
    log_level = getattr(logging, args.pop("log_level").upper())

    logger.setLevel(log_level)
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(
        logging.Formatter(
            "{asctime} {levelname} {name:>16}│ {message}", datefmt="%H:%M:%S", style="{"
        )
    )
    logger.addHandler(ch)

    if sys.stdout.isatty():
        _m = logging.getLevelNamesMapping()
        for c, lvl in [[226, "DEBUG"], [148, "INFO"], [208, "WARNING"], [197, "ERROR"]]:
            logging.addLevelName(_m[lvl], f"\x1b[38;5;{c}m{lvl:<7}\x1b[0m")

    main()
