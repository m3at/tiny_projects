#!/usr/bin/env python3

import re
import unicodedata
from collections.abc import Generator
from textwrap import shorten
from time import perf_counter
from typing import Final, Tuple

import httpx
import numpy as np
import onnxruntime
from numpy.typing import NDArray
from qdrant_client import QdrantClient
from qdrant_client.conversions.common_types import ScoredPoint
from qdrant_client.fastembed_common import QueryResponse
from rich.console import Console
from rich.prompt import Prompt
from tokenizers import Tokenizer

DOCS_MAX_TOKENS_RERANK: Final[int] = 1024
PRINT_MAX_CHAR: Final[int] = 768


class Reranker:
    def __init__(self):
        self._query_max_chars = 512

        self.tokenizer = Tokenizer.from_pretrained(
            "Alibaba-NLP/gte-multilingual-reranker-base"
        )
        self.tokenizer.enable_padding()
        self.tokenizer.enable_truncation(max_length=DOCS_MAX_TOKENS_RERANK)

        model_path = "./onnx-gte-multilingual-reranker-base/model.onnx"

        # _ep = ["CoreMLExecutionProvider", 'CPUExecutionProvider']
        _ep = ["CPUExecutionProvider"]
        opt = onnxruntime.SessionOptions()
        opt.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.ort_sess = onnxruntime.InferenceSession(
            model_path, providers=_ep, sess_options=opt
        )

        self.regex_char_remove = re.compile(r"[\[\]\-#/@;{}=~|～]|:\/\/.*?[\r\n]|  ")

        # Warmup
        pairs = [["query", d] for d in ["first doc", "second doc"]]
        _ = self.ort_sess.run(["logits"], self._prepare_inputs(pairs))[0]

    def _prepare_inputs(self, pairs: list[list[str]]):
        encoded = self.tokenizer.encode_batch(pairs)

        _input_ids = np.stack([x.ids for x in encoded])

        return {
            "input_ids": _input_ids,
            "attention_mask": np.stack([x.attention_mask for x in encoded]),
            # just a guess from checking random source code for `token_type_ids`
            # https://github.com/search?q=repo%3AFlagOpen%2FFlagEmbedding%20token_type_ids&type=code
            "token_type_ids": np.zeros_like(_input_ids),
        }

    def clean_text(self, text: str) -> str:
        """Lowercase, normalize unicode and prune some punctuations."""
        text = text.lower()
        text = unicodedata.normalize("NFKC", text)
        text = self.regex_char_remove.sub("", text)
        text = text.strip()
        return text

    def __call__(self, query: str, docs: list[str]) -> NDArray[np.float32]:
        """Higher score is better. Not in a defined range."""

        query = self.clean_text(query)[: self._query_max_chars]
        pairs = [[query, d] for d in docs]
        inputs = self._prepare_inputs(pairs)
        outputs: NDArray[np.float32] = self.ort_sess.run(["logits"], inputs)[0][:, 0]
        return outputs


def loop(console: Console) -> Generator[str, None, None]:
    try:
        while True:
            console.print()
            query = Prompt.ask(
                "[green]Query[/]",
                console=console,
                default="java dev wfh",
                show_default=False,
            )

            if query.lower() == "exit" or query.lower() == ":q":
                exit(0)

            yield query

    except KeyboardInterrupt:
        exit(0)


def extract_numbers(
    s, maxint: int, *, check=re.compile(r"^[\s+\-0-9]*$"), parse=re.compile(r"[+-]?\d+")
) -> Tuple[list | None, list | None]:
    """Find integers if any, separate into positive and negative"""

    if not check.match(s):
        return None, None

    numbers = parse.findall(s)

    pos = [int(x) for x in numbers if x.startswith("+")]
    neg = [int(x) for x in numbers if x.startswith("-")]

    # Assume that no sign is positive
    for num in numbers:
        if not num.startswith("+") and not num.startswith("-"):
            pos.append(int(num))

    # Ignore anything out of bounds
    pos = [x for x in pos if x <= maxint]
    neg = [x for x in neg if x <= maxint]

    return pos, neg


def augment_query(
    client: httpx.Client, content: str, *, temperature=1.2, port: int = 8082
) -> list[str]:
    _pre = """\
<|im_start|>system\nYou are a query augmentation engine, part of a job board. You augment queries to increase the chances of a search match. You write exactly 3 augmentations.<|im_end|>\n<|im_start|>user\nSoftware engineer jobs in New York<|im_end|>\n<|im_start|>assistant\nNew York software engineer job openings\nsoftware development positions in NYC\nsoftware engineer roles in New York City<|im_end|>\n<|im_start|>user\nRemote marketing positions<|im_end|>\n<|im_start|>assistant\nremote marketing job opportunities\nwork from home marketing roles\ndigital marketing positions remote<|im_end|>\n<|im_start|>user\ndata sciens job entry level<|im_end|>\n<|im_start|>assistant\nentry-level data science jobs\njunior data scientist positions\ndata analyst internships<|im_end|>\n<|im_start|>user\nsenior devops engineer san francisco<|im_end|>\n<|im_start|>assistant\nsenior DevOps positions in San Francisco\nSan Francisco DevOps engineer jobs\nlead DevOps roles in SF<|im_end|>\n<|im_start|>user\npart time IT jobs near me<|im_end|>\n<|im_start|>assistant\npart-time IT support roles nearby\nIT technician part-time jobs\nflexible IT jobs in my area<|im_end|>\n<|im_start|>user\n"""
    _post = """<|im_end|>\n<|im_start|>assistant\n"""

    response = client.post(
        f"http://localhost:{port}/completion",
        json={
            "temp": temperature,
            "n_predict": 32,
            "cache_prompt": True,
            "stop": ["<|im_end|>"],
            "prompt": _pre + content + _post,
        },
        headers={"Content-Type": "application/json"},
    )

    if response.status_code == 200:
        augmented = response.json().get("content")
        augmented = augmented.split("\n")
        return augmented

    print(f"Request failed with status code {response.status_code}")
    raise ValueError("bad engine, bad")


def get_embedding(
    client: httpx.Client, content: str, *, port: int = 8989
) -> list[float]:
    response = client.post(
        f"http://localhost:{port}/embedding",
        json={"content": content},
        headers={"Content-Type": "application/json"},
    )

    if response.status_code == 200:
        embedding = response.json().get("embedding")
        return embedding

    print(f"Request failed with status code {response.status_code}")
    raise ValueError("bad kitty, bad")


def print_rez(console, r: list[QueryResponse] | list[ScoredPoint]):
    for _, row in enumerate(r):
        if isinstance(row, QueryResponse):
            m = row.metadata
        else:
            m = row.payload
            if m is None:
                console.log("Panic, no payload")
                return

        loc = "?" if len(ml := m["locations"]) < 2 else ml[0]
        doc = shorten(m["hn_text"], width=PRINT_MAX_CHAR)
        company = m["company"]
        console.print(
            # f"[{idx+1}] [yellow]{row.score:.3f} [/][underline]{company}[/] ({loc})\n{doc}\n"
            f"[yellow]{row.score:.3f} [/][underline]{company}[/] ({loc})\n{doc}"
        )


def print_rez_rr(console, rr):
    for s, row in rr:
        m = row.payload
        if m is None:
            continue
        loc = "?" if len(ml := m["locations"]) < 2 else ml[0]
        doc = shorten(m["hn_text"], width=PRINT_MAX_CHAR)
        company = m["company"]
        console.print(
            f"[green][{s:>6.3f}][/] [yellow]{row.score:.3f}[/] [underline]{company}[/] ({loc})\n{doc}"
        )


def mainloop(http_client: httpx.Client, reranker: Reranker) -> None:
    client = QdrantClient(path="db.qdrant")
    # highlight automatically color some stuff like numbers and urls
    console = Console(highlight=False)

    collection_name = "hn_jobs"
    c = client.count(collection_name)

    limit = 16
    show_top = 8

    console.log(
        f"Starting search demo, on {c.count} entries. Retrieval+rerank N={limit}. Showing top={show_top}"
    )
    console.print("Use natural language query.")

    r = None

    for query in loop(console):
        # TODO: use model to intelligently parse the query, extract filter from it
        # Use LLM to augment query
        # User re-ranker

        # Query augmentation
        _t = perf_counter()
        # _query_vec = get_embedding(http_client, query)
        augmented_queries = augment_query(http_client, query)
        # TODO: do in parallel
        _q = np.stack(
            [np.asarray(get_embedding(http_client, q)) for q in augmented_queries]
        )
        _query_vec = np.mean(_q, axis=0)
        _delta = perf_counter() - _t
        console.print(f"[cornflower_blue]QUERY AUGMENTATION took {_delta:.2f}s[/]")
        console.print("Query augmented to:\n{}".format("\n".join(augmented_queries)))

        # Vector search
        _t = perf_counter()
        r = client.query_points(
            collection_name=collection_name,
            query=_query_vec,
            query_filter=None,
            limit=limit,
        ).points

        _delta = perf_counter() - _t
        console.print(f"[cornflower_blue]VECTOR SEARCH took {_delta:.2f}s[/]")
        print_rez(console, r[:show_top])

        # Reranking
        _t = perf_counter()
        _docs = [x.payload["hn_text"] for x in r if x.payload is not None]
        scores = reranker(query, _docs)
        rr = sorted(list(zip(scores, r)), key=lambda x: x[0], reverse=True)

        _delta = perf_counter() - _t
        console.print(f"[cornflower_blue]RERANKED took {_delta:.2f}s[/]")
        print_rez_rr(console, rr[:show_top])


def main() -> None:
    reranker = Reranker()
    with httpx.Client() as http_client:
        mainloop(http_client, reranker)


if __name__ == "__main__":
    main()
