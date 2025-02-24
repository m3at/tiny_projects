#!/usr/bin/env python3


import argparse
import logging
import platform
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Callable

import torch
from google.cloud import storage

logger = logging.getLogger("base")

DELAY = 0.02


def p(message: str, log: Callable = logger.info) -> None:
    # print(message, flush=True)
    log(message)


def check_exist(path: str):
    p = Path(path)
    logger.debug(f"{p.exists()} {p}")


def main() -> None:
    ####
    # Check GPU
    ####
    has_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if has_cuda else "N/A"
    p(
        "[Pre-flight check] CUDA: {} , GPU count: {}, device: {}".format(
            has_cuda, torch.cuda.device_count(), device_name
        )
    )

    p("[Liftoff]", logger.debug)

    # logger.warning(pwd.getpwuid(os.getuid())[0])

    ####
    # Read from GCP
    ####
    storage_client = storage.Client(project="CHANGEME")
    blobs = list(storage_client.list_blobs("BUCKET", prefix="test-job-k8s"))
    p(f"[Max Q] List blobs: {[x.name for x in blobs]}")

    bucket = storage_client.bucket("BUCKET")
    blob: storage.Blob = bucket.get_blob("test-job-k8s/content.txt")  # type:ignore

    with blob.open() as f:
        content = f.readlines()

    p(f"[MECO] Read content: {content}")

    ####
    # Write to GCP
    ####
    # "blob" create a new blob
    blob: storage.Blob = bucket.blob("test-job-k8s/write_payload.txt")
    payload = "{}-{}-{}".format(platform.node(), platform.machine(), datetime.now())
    # blob.upload_from_string("", content_type="application/x-www-form-urlencoded;charset=UTF-8")
    blob.upload_from_string(payload)

    p(f"[Payload deployment] Wrote content into: {blob.name}")
    # gsutil cat gs://BUCKET/test-job-k8s/write_payload.txt

    sleep(DELAY)
    p("[Mission success]", logger.debug)


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
            "{asctime} {levelname}│ {message}", datefmt="%H:%M:%S", style="{"
        )
    )
    logger.addHandler(ch)

    # Add colors
    _levels = [[226, "DEBUG"], [148, "INFO"], [208, "WARNING"], [197, "ERROR"]]
    for color, lvl in _levels:
        _l = getattr(logging, lvl)
        logging.addLevelName(
            _l, "\x1b[38;5;{}m{:<7}\x1b[0m".format(color, logging.getLevelName(_l))
        )

    main()
