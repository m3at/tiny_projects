#!/usr/bin/env python3

import argparse
import concurrent.futures
import json
import logging
import sys
from pathlib import Path

from lib.common import MODEL, RE_FILENAME, ProcessBase, get_cost, threads_progress
from lib.static import PLACES, SYSTEM_PROMPT

logger = logging.getLogger("base")


class Process(ProcessBase):
    def process(self, place: str) -> float:
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
