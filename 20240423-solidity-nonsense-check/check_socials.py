#!/usr/bin/env python3

import argparse
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def reach_url(url: str) -> int | None:
    try:
        req = Request(url)
        # Set a user agent to be less likely to be rejected, especially by twitter
        req.add_header(
            "User-Agent",
            "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
        )
        return urlopen(req).getcode()
    except HTTPError:
        return


def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc]) and result.scheme in (
            "http",
            "https",
        )
    except ValueError:
        return False


def main(filename: Path, maxlines: int):
    lines = filename.read_text().splitlines()

    pattern = re.compile(r"(https?://[^\s]+)")

    buff = []
    for line in lines[:maxlines]:
        buff.extend(pattern.findall(line))

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {}
        for url in buff:
            if not is_valid_url(url):
                print(f"Invalid URL {url=}")
                continue
            futures[executor.submit(reach_url, url)] = url

        for f in futures:
            url = futures[f]
            print(f"{url=} {f.result()=}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Regex for URLs in a text file, and check if each are valid and reachable",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("filename", type=Path)
    parser.add_argument("--maxlines", type=int, default=50)
    args = parser.parse_args()
    main(args.filename, args.maxlines)
