#!/usr/bin/env python3

import argparse
import concurrent.futures
import json
import logging
import os
import re
import sys
from collections.abc import Generator
from pathlib import Path
from typing import Final, Tuple

from openai import OpenAI
from openai.types import CompletionUsage
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from tenacity import retry, stop_after_attempt, wait_random_exponential

from static import PLACES, SYSTEM_PROMPT

logger = logging.getLogger("base")

RE_FILENAME = re.compile(r"[^a-zA-Z_-]+")

# As of 2024/06/02
# https://openai.com/api/pricing/
_PRICES = {
    # Input, output
    "gpt-4o": (5, 15),
    "gpt-3.5-turbo-0125": (0.5, 1.5),
    "text-embedding-3-small": (0.02, 0.02),
}
# Prices are per 1M tokens
PRICES: Final[dict[str, Tuple[int, int]]] = {
    k: (v[0] / 1e6, v[1] / 1e6) for k, v in _PRICES.items()
}


# MODEL: Final = "gpt-3.5-turbo-0125"
MODEL: Final = "gpt-4o"


def get_cost(usage: CompletionUsage, *, prices=PRICES[MODEL]) -> float:
    a, b = prices
    return (a * usage.prompt_tokens) + (b * usage.completion_tokens)


def threads_progress(executor, fn, *iterables) -> Generator[float, None, None]:
    futures_list = []

    for iterable in iterables:
        futures_list += [executor.submit(fn, i) for i in iterable]

    # tqdm.contrib.concurrent.thread_map is also nice, and a single line
    c = 0
    with Progress(
        TextColumn("total: {task.description} USD"),
        BarColumn(bar_width=None),
        TimeRemainingColumn(elapsed_when_finished=True),
    ) as progress:
        task = progress.add_task(" " * 5, total=len(futures_list))

        for f in concurrent.futures.as_completed(futures_list):
            c += f.result()
            progress.update(task, description=f"{c:>5.2f}", advance=1)
            yield c


class Process:
    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY", None)
        assert api_key is not None, "Please set the env OPENAI_API_KEY"
        self.client = OpenAI(api_key=api_key)

    @retry(wait=wait_random_exponential(min=2, max=20), stop=stop_after_attempt(3))
    def __call__(self, place: str) -> float:
        response = self.client.chat.completions.create(
            model=MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"place: {place}"},
            ],
            timeout=30,
        )

        rez = response.choices[0].message.content

        if rez is None or response.usage is None:
            logger.error(f"Model says no :( {place=}")
            return 0

        cost = get_cost(response.usage)

        try:
            j = json.loads(rez)
        except Exception:
            logger.error(f"Not a valid json. Bad model. Bad! {place=}")
            return cost

        p = RE_FILENAME.sub("_", place).lower()
        Path(f"data/{p}.json").write_text(json.dumps(j))

        return cost


def main() -> None:
    logger.debug(f"Starting, using model: {MODEL}")

    Path("data").mkdir(exist_ok=True)
    process = Process()

    # Cost about 15$ for 64 places
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        costs = list(threads_progress(executor, process, PLACES))

    logger.info(f"Total cost: {sum(costs):.2f} USD")


if __name__ == "__main__":
    # Get system arguments
    parser = argparse.ArgumentParser(
        description="Generate a fake dataset of tourism tours using OpenAI's api",
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

    # Setup logging
    logger.setLevel(log_level)
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(
        logging.Formatter(
            "{asctime} {levelname}│ {message}", datefmt="%H:%M:%S", style="{"
        )
    )
    logger.addHandler(ch)

    # Add colors if stdout is not piped
    if sys.stdout.isatty():
        _levels = [[226, "DEBUG"], [148, "INFO"], [208, "WARNING"], [197, "ERROR"]]
        for color, lvl in _levels:
            _l = getattr(logging, lvl)
            logging.addLevelName(
                _l, "\x1b[38;5;{}m{:<7}\x1b[0m".format(color, logging.getLevelName(_l))
            )

    main()
