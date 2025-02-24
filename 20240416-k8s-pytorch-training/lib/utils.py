import hashlib
import io
import logging
import os
import re
import unicodedata
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from itertools import islice
from pathlib import Path
from time import perf_counter
from typing import Generator, Iterable, List, Tuple, TypeVar

import numpy as np
from google.cloud import storage

logger = logging.getLogger("base")

# Generic type
T = TypeVar("T")


def str_to_uuid(value):
    # _id = uuid.UUID(hex=hashlib.md5(_id.encode("utf-8")).hexdigest())
    return hashlib.md5(value.encode("utf-8")).hexdigest()


@contextmanager
def timer(text, *, use_ms=True):
    start = perf_counter()
    try:
        yield
    finally:
        _diff = perf_counter() - start
        if use_ms:
            _diff *= 1000
            logger.info(f"{text}: {_diff:.1f}ms")
        else:
            logger.info(f"{text}: {_diff:.2f}s")


class Chrono:
    """Convenience elapsed time printing."""

    def __init__(self):
        self.delta = perf_counter()

    def __call__(self) -> float:
        new = perf_counter()
        diff = new - self.delta
        self.delta = new
        return diff

    def __repr__(self, pad=7, precision=1) -> str:
        diff = self()
        return f"{diff:>{pad}.{precision}f}s"


def check_pod_parallelism() -> Tuple[int, int]:
    """Use environment and kubernetes indexed job capability to get the number of sibling pods.

    https://kubernetes.io/blog/2021/04/19/introducing-indexed-jobs/
    """

    pod_parallelism_n = int(os.environ.get("POD_PARALLELISM_N", 1))

    if "JOB_COMPLETION_INDEX" not in os.environ:
        logger.info("Env JOB_COMPLETION_INDEX not found, using local_rank=0")
        if pod_parallelism_n < 2:
            logger.warning(
                "No JOB_COMPLETION_INDEX found, forced to disable parallelism"
            )
            pod_parallelism_n = 0

    local_rank = int(os.environ.get("JOB_COMPLETION_INDEX", 0))

    logger.debug(f"Parallelized across N={pod_parallelism_n}, local_rank={local_rank}")

    return pod_parallelism_n, local_rank


def batched(iterable: Iterable[T], batch_size: int) -> Generator[List[T], None, None]:
    assert batch_size > 1, f"Batch size need to be at least 2, got: {batch_size}"

    # it = iter(iterable)
    while batch := list(islice(iterable, batch_size)):
        yield batch


@contextmanager
def quiet(logging):
    level = logging.getLogger().getEffectiveLevel()
    logging.getLogger().setLevel(logging.ERROR)
    try:
        with (
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore")
            yield
    finally:
        logging.getLogger().setLevel(level)


REGEX_CHAR_REMOVE = re.compile(
    r"[［］\[\]\-「」【】（）()\"#/@;:<>{}=~|.?,～]|:\/\/.*?[\r\n]|  "
)


def clean_text(text: str, *, to_sub: re.Pattern[str] = REGEX_CHAR_REMOVE) -> str:
    """Lowercase, normalize unicode and prune some punctuations."""

    text = text.lower()
    text = unicodedata.normalize("NFKC", text)
    text = to_sub.sub("", text)

    # Remove leading and trailing characters
    text = text.strip()

    return text


clean_text_vectorized = np.vectorize(clean_text)


def clean_text_list(texts):
    return clean_text_vectorized(texts).tolist()


def gcp_download_if_missing(
    gcp_path: str,
    local_path: Path = Path("./models_weight"),
    bucket_name: str = "BUCKET_NAME",
):
    if not local_path.exists():
        logger.info(f"Downloading from gcp: {local_path}")
        local_path.parent.mkdir(exist_ok=True)

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        bucket.blob(gcp_path).download_to_filename(local_path)


def gcp_check_file_exists(gcp_path, bucket_name: str = "BUCKET_NAME") -> bool:
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    return bucket.blob(gcp_path).exists()


def upload_to_gcp(
    files: List[Tuple[Path, str | None]],
    *,
    gcp_dir: str,
    bucket_name: str = "BUCKET_NAME",
):
    logger.info("Uploading to GCP")

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    for f, subdir in files:
        if subdir is None:
            subdir = ""
        else:
            subdir = subdir if subdir.endswith("/") else subdir + "/"

        _name = f.name
        _name = _name.replace(" ", "_")

        bucket.blob(f"{gcp_dir}/{subdir}{_name}").upload_from_filename(str(f))

    logging.info(f"Uploaded models to gcp in: {gcp_dir}")
