#!/usr/bin/env python3

import argparse
import logging
import os
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from random import random
from time import sleep

logger = logging.getLogger("base")


class ShouldRestartException(Exception):
    pass


def receive_usr1(signum, stack):
    logger.warning(f"Received: {signum=}, from: {stack=}")


def receive_usr1_send_usr2(signum, _, target_pid: int, delay: float):
    # Send USR2 to current program after some time
    logger.warning(f"Received: {signum=}, will send SIGUSR2 to myself in {delay}s")
    sleep(delay)
    os.kill(target_pid, signal.SIGUSR2)


def receive_usr2(signum, _):
    logger.error(f"Received: {signum=}")
    raise ShouldRestartException()


def worker(param):
    logger.debug(f"worker: {param}")
    sleep(0.5 + random())


def busy_loop():
    c = 0
    with ThreadPoolExecutor(max_workers=3) as executor:
        while True:
            logger.info(f"Sending work, step {c}")
            for i in range(3):
                executor.submit(worker, dict(c=c, i=i))
            c += 1
            sleep(0.5 + random())


def main() -> None:
    pid = os.getpid()
    logger.info(f"Program's PID: {pid}")

    # signal.signal(signal.SIGUSR1, receive_usr1)
    signal.signal(
        signal.SIGUSR1, partial(receive_usr1_send_usr2, target_pid=pid, delay=5.0)
    )
    signal.signal(signal.SIGUSR2, receive_usr2)

    try:
        busy_loop()
    except ShouldRestartException:
        logger.info("Would restart")
    except KeyboardInterrupt:
        logger.info("Exit on KeyboardInterrupt")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test of signal handling with threading. Usage: kill -USR1 PID; kill -USR2 PID",
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
            "{asctime} {levelname}│ {message}", datefmt="%H:%M:%S", style="{"
        )
    )
    logger.addHandler(ch)

    if sys.stdout.isatty():
        _levels = [[226, "DEBUG"], [148, "INFO"], [208, "WARNING"], [197, "ERROR"]]
        for color, lvl in _levels:
            _l = getattr(logging, lvl)
            logging.addLevelName(
                _l, "\x1b[38;5;{}m{:<7}\x1b[0m".format(color, logging.getLevelName(_l))
            )

    main()
