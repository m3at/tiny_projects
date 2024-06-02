#!/usr/bin/env python3

import argparse
import json
import logging
import re
import sys
from pathlib import Path

from qdrant_client import QdrantClient

logger = logging.getLogger("base")


def extract_float(sentence, *, re_prices=re.compile(r"\b\d+\.\d+|\b\d+\b")):
    # TODO: show example of using LLM to extract consistent price format
    sentence = sentence.lower()
    if "free" in sentence:
        return 0
    match = re_prices.search(sentence)
    return 0 if not match else float(match.group())


def main() -> None:
    logger.info("Loading data")

    docs = []
    metadata = []
    for f in Path("./data").glob("*.json"):
        j = json.loads(f.read_text())
        try:
            area, country = j["place"].split(", ", maxsplit=1)
        except ValueError:
            logger.warning("Couldn't split: {}".format(j["place"]))
            area = j["place"]
            country = ""

        for listing in j["listings"]:
            price = extract_float(listing["price"])

            metadata.append(
                dict(
                    country=country,
                    area=area,
                    price=price,
                    title=listing["title"],
                    location=listing["location"],
                    access=listing["access"],
                    duration=listing["duration"],
                    requirements=listing["requirements"],
                    highlights=listing["highlights"],
                )
            )
            docs.append(
                listing["what to expect"],
            )

    # client = QdrantClient(":memory:")
    client = QdrantClient(path="db.qdrant")

    # client.set_model()

    logger.info(f"Generating embeddings and upserting data for {len(docs)} entries...")
    client.add(
        collection_name="demo_collection",
        documents=docs,
        metadata=metadata,
        # ids=list(range(len(docs))),
    )
    logger.info("Done")


if __name__ == "__main__":
    # Get system arguments
    parser = argparse.ArgumentParser(
        description="Generate embeddings",
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
