#!/usr/bin/env python3

import argparse
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path

import httpx
from openai import OpenAI
from tqdm.contrib.concurrent import thread_map

logger = logging.getLogger("base")


def strip_accents(s: str) -> str:
    """adélie -> adelie"""
    # return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    return unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode()


def get_img(client, p, place_name_org):
    response = client.images.generate(
        # model="dall-e-2", size="256x256",  # $0.016 / image, 9s
        model="dall-e-3",
        size="1024x1024",  # $0.040 / image, 17s
        # prompt=f"a wonderful toursim magazine cover of {place_name_org}",
        # prompt=f"an award winning landscape photography of {place_name_org}",
        prompt=f"high quality picture, award winning photography of {place_name_org}, detailed, daytime, aesthetic, magazine cover, 8k",
        quality="standard",
        n=1,
    )

    image_url = response.data[0].url
    assert image_url is not None
    r = httpx.get(image_url)
    b = r.content
    p.write_bytes(b)


def main() -> None:
    client = OpenAI()

    cache_dir = Path.home() / ".cache" / "m3at" / "mock_cdn"
    cache_dir.mkdir(exist_ok=True, parents=True)

    buff = []
    for f in Path("./data").glob("*.json"):
        j = json.loads(f.read_text())
        try:
            _, country = j["place"].split(", ", maxsplit=1)
        except ValueError:
            country = ""

        for listing in j["listings"]:
            place_name_org = "{}, {}".format(listing["location"], country)

            place_name = strip_accents(place_name_org.lower())
            place_name = re.sub(r"[^a-z]+", "_", place_name)
            p = cache_dir / f"{place_name}.png"
            if p.exists():
                continue
            buff.append((place_name, p))

    def fn(x):
        get_img(client, x[1], x[0])

    thread_map(fn, buff, max_workers=8)


if __name__ == "__main__":
    # Get system arguments
    parser = argparse.ArgumentParser(
        description="replace_me",
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
            "{asctime} {levelname}│ {message}",
            datefmt="%H:%M:%S",
            style="{",
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
