import concurrent.futures
import logging
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Final, Tuple

from openai import OpenAI
from openai.types import CompletionUsage
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from tenacity import retry, stop_after_attempt, wait_random_exponential

logger = logging.getLogger("base")

RE_FILENAME = re.compile(r"[^a-zA-Z_-]+")

# As of 2024/06/02
# https://openai.com/api/pricing/
_PRICES = {
    # Input, output
    "gpt-4o": (5, 15),
    "gpt-3.5-turbo-0125": (0.5, 1.5),
    "text-embedding-3-small": (0.02, 0.02),
    "text-embedding-3-large": (0.13, 0.13),
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


class ProcessBase(ABC):
    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY", None)
        assert api_key is not None, "Please set the env OPENAI_API_KEY"
        self.client = OpenAI(api_key=api_key)

    @abstractmethod
    def process(self, *args, **kwargs) -> float:
        pass

    @retry(wait=wait_random_exponential(min=2, max=20), stop=stop_after_attempt(3))
    def __call__(self, *args, **kwargs) -> float:
        return self.process(*args, **kwargs)
