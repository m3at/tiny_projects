#!/usr/bin/env python3

import re
import unicodedata
from collections.abc import Generator
from textwrap import shorten
from time import perf_counter
from typing import Final, Tuple

import bm25s
import httpx
import numpy as np
import onnxruntime
import Stemmer
from numpy.typing import NDArray
from qdrant_client import QdrantClient
from qdrant_client.conversions.common_types import ScoredPoint
from rich.console import Console
from rich.prompt import Prompt
from tokenizers import Tokenizer

# DOCS_MAX_TOKENS_RERANK: Final[int] = 256
DOCS_MAX_TOKENS_RERANK: Final[int] = 512  # 2.6s
# DOCS_MAX_TOKENS_RERANK: Final[int] = 1024  # 2.6s. Docs are too short?
PRINT_MAX_CHAR: Final[int] = 500


class Reranker:
    def __init__(self):
        self._query_max_chars = 512

        self.tokenizer = Tokenizer.from_pretrained(
            # "Alibaba-NLP/gte-multilingual-reranker-base"
            "jinaai/jina-reranker-v2-base-multilingual"
        )
        self.tokenizer.enable_padding()
        self.tokenizer.enable_truncation(max_length=DOCS_MAX_TOKENS_RERANK)

        # model_path = "./models/gte-multilingual-reranker-base.onnx"
        # TODO: try https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual
        # Should be almost the same quality but a bit faster
        model_path = "./models/jina-reranker-v2-base-multilingual_quantized.onnx"
        # Try:
        # https://github.com/ggerganov/llama.cpp/pull/9510

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
            # "token_type_ids": np.zeros_like(_input_ids),
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


class RerankerExt:
    def __init__(self):
        self._query_max_chars = 512
        # self._docs_max_chars = 10_000
        self._docs_max_chars = 1100
        self.regex_char_remove = re.compile(r"[\[\]\-#/@;{}=~|～]|:\/\/.*?[\r\n]|  ")

        # hf's text-embeddings-inference , using BAAI/bge-reranker-v2-m3
        self.port = 8084
        self.host = "127.0.0.1"

    def clean_text(self, text: str) -> str:
        """Lowercase, normalize unicode and prune some punctuations."""
        text = text.lower()
        text = unicodedata.normalize("NFKC", text)
        text = self.regex_char_remove.sub("", text)
        text = text.strip()
        return text

    def __call__(self, query: str, docs: list[str]) -> NDArray[np.float32]:
        payload = {
            "query": self.clean_text(query)[: self._query_max_chars],
            "texts": [self.clean_text(x)[: self._docs_max_chars] for x in docs],
        }

        response = httpx.post(f"http://{self.host}:{self.port}/rerank", json=payload, timeout=20)
        response.raise_for_status()
        # [{"index":2,"score":0.9976495},{"index":1,"score":0.12721826},{"index":0,"score":0.000035081117}]
        results = response.json()

        # The score is already sorted in decreasing order, but we'll revert that
        # return np.array([result['score'] for result in results], dtype=np.float32)

        oredered_rez = sorted(results, key=lambda x: x["index"], reverse=False)
        return np.array([result["score"] for result in oredered_rez], dtype=np.float32)


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
    # Qwen2
    #     _pre = """\
    # <|im_start|>system\nYou are a query augmentation engine, part of a job board. You augment queries to increase the chances of a search match. You write exactly 3 augmentations.<|im_end|>\n<|im_start|>user\nSoftware engineer jobs in New York<|im_end|>\n<|im_start|>assistant\nNew York software engineer job openings\nsoftware development positions in NYC\nsoftware engineer roles in New York City<|im_end|>\n<|im_start|>user\nRemote marketing positions<|im_end|>\n<|im_start|>assistant\nremote marketing job opportunities\nwork from home marketing roles\ndigital marketing positions remote<|im_end|>\n<|im_start|>user\ndata sciens job entry level<|im_end|>\n<|im_start|>assistant\nentry-level data science jobs\njunior data scientist positions\ndata analyst internships<|im_end|>\n<|im_start|>user\nsenior devops engineer san francisco<|im_end|>\n<|im_start|>assistant\nsenior DevOps positions in San Francisco\nSan Francisco DevOps engineer jobs\nlead DevOps roles in SF<|im_end|>\n<|im_start|>user\npart time IT jobs near me<|im_end|>\n<|im_start|>assistant\npart-time IT support roles nearby\nIT technician part-time jobs\nflexible IT jobs in my area<|im_end|>\n<|im_start|>user\n"""
    #     _post = """<|im_end|>\n<|im_start|>assistant\n"""

    # Llama 3.2, 3 lines, 5 shots
    #     _pre = """\
    # <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nYou are a query augmentation engine, part of a job board. You augment queries to increase the chances of a search match. You write exactly 3 augmentations.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nSoftware engineer jobs in New York<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nNew York software engineer job openings\nsoftware development positions in NYC\nsoftware engineer roles in New York City<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nRemote marketing positions<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nremote marketing job opportunities\nwork from home marketing roles\ndigital marketing positions remote<|eot_id|><|start_header_id|>user<|end_header_id|>\n\ndata sciens job entry level<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nentry-level data science jobs\njunior data scientist positions\ndata analyst internships<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nsenior devops engineer san francisco<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nsenior DevOps positions in San Francisco\nSan Francisco DevOps engineer jobs\nlead DevOps roles in SF<|eot_id|><|start_header_id|>user<|end_header_id|>\n\npart time IT jobs near me<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\npart-time IT support roles nearby\nIT technician part-time jobs\nflexible IT jobs in my area<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"""
    #     _post = """<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""

    # Llama 3.2, 4 lines, 10 shots
    _pre = """\
<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nYou are a query augmentation engine, part of a job board. You augment queries to increase the chances of a search match. You write exactly 4 augmentations.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nSoftware engineer jobs in New York<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nNew York software engineer job openings\nsoftware development positions in NYC\nsoftware engineer roles in New York City\nNYC tech jobs for software engineers<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nRemote marketing positions<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nremote marketing job opportunities\nwork from home marketing roles\ndigital marketing positions remote\nvirtual marketing jobs<|eot_id|><|start_header_id|>user<|end_header_id|>\n\ndata sciens job entry level<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nentry-level data science jobs\njunior data scientist positions\ndata analyst internships\ndata sciensce entry positions<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nsenior devops engineer san francisco<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nsenior DevOps positions in San Francisco\nSan Francisco DevOps engineer jobs\nlead DevOps roles in SF\nsenior cloud engineer jobs SF<|eot_id|><|start_header_id|>user<|end_header_id|>\n\npart time IT jobs near me<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\npart-time IT support roles nearby\nIT technician part-time jobs\nflexible IT jobs in my area\nlocal part-time technology positions<|eot_id|><|start_header_id|>user<|end_header_id|>\n\ninternships for computer science students<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\ncomputer science internships available\npaid internships for CS students\ntech internships for college students\nsoftware internships for beginners<|eot_id|><|start_header_id|>user<|end_header_id|>\n\njobs in AI and machine learning<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nAI job opportunities in tech\nmachine learning engineer positions\nartificial intelligence career openings\njobs in AI research and development<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nhiring ux designers now<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nurgent UX designer positions\nUX/UI design jobs hiring immediately\nuser experience designer job openings\nremote UX design jobs available<|eot_id|><|start_header_id|>user<|end_header_id|>\n\njava developer work from home<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nremote Java developer positions\nJava programming jobs from home\nwork from home Java software engineer\ntelecommute Java developer roles<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nproject manager tech startup<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nproject management jobs in tech startups\nstartup project manager openings\nagile project manager positions in tech\ntech startup PM job opportunities<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"""
    _post = """<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""

    response = client.post(
        f"http://localhost:{port}/completion",
        json={
            "temp": temperature,
            "n_predict": 32,
            "cache_prompt": True,
            # "stop": ["<|im_end|>"],
            "stop": ["<|eot_id|>"],
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


def print_rez(console, r: list[ScoredPoint]):
    for _, row in enumerate(r):
        # if isinstance(row, QueryResponse):
        #     m = row.metadata
        # else:
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


def mainloop(http_client: httpx.Client, reranker: Reranker | RerankerExt) -> None:
    print("Setup...")
    client = QdrantClient(path="db.qdrant")
    # highlight automatically color some stuff like numbers and urls
    console = Console(highlight=False)

    retriever_bm25 = bm25s.BM25.load("bm25_index", load_corpus=False)
    # TODO: this is using HF tokenizer under the hood. Prevent double loading
    stemmer = Stemmer.Stemmer("english")

    collection_name = "hn_jobs"
    c = client.count(collection_name)

    # limit = 16  # 2.6s at 1024
    # limit = 16  # 2.7s at 512 ?
    limit_vector = 12
    limit_bm25 = 4
    limit = limit_vector + limit_bm25
    # limit = 32  # 8.0s at 1024
    # limit = 32  # 3.3s at 256
    # limit = 32  # 5.5s at 512
    show_top = 5

    # senior ml eng medical domain

    console.log(
        f"Starting search demo, on {c.count} entries. Retrieval & rerank N={limit}. Showing top={show_top}"
    )
    console.print(
        "Use natural language query. Example `java dev wfh`, `senior ml eng medicl domain`"
    )

    r = None

    for query in loop(console):
        # TODO: use model to intelligently parse the query, extract filter from it
        # Use LLM to augment query
        # User re-ranker

        # ╔════════════════════╗
        # ║ Query augmentation ║
        # ╚════════════════════╝
        _t = perf_counter()
        # _query_vec = get_embedding(http_client, query)
        augmented_queries = augment_query(http_client, query)
        # TODO: do in parallel
        _q = np.stack(
            [np.asarray(get_embedding(http_client, q)) for q in augmented_queries]
        )
        # _query_vec = np.mean(_q, axis=0)
        _delta = perf_counter() - _t
        console.print(f"[cornflower_blue]QUERY AUGMENTATION took {_delta:.2f}s[/]")
        console.print(
            "Query '{}' augmented to:\n{}".format(query, "\n".join(augmented_queries))
        )

        # ╔═════════════╗
        # ║ BM25 search ║
        # ╚═════════════╝
        query_tokens = bm25s.tokenize(augmented_queries[0], stemmer=stemmer)
        # Scores are sorted from higher (most relevant) to lover (least relevant)
        # Shape: indexes: list[list[int]] , scores: list[list[float]]
        # idx_bm25, scores_bm25 = retriever_bm25.retrieve(query_tokens, k=limit_bm25)
        idx_bm25, _ = retriever_bm25.retrieve(query_tokens, k=limit_bm25)
        # TODO: track global indexes better
        _r = client.retrieve(collection_name, idx_bm25[0].tolist())
        # assert _r is not None
        _docs_bm25 = [x.payload["hn_text"] for x in _r]  # type:ignore
        # TODO: print BM25 results

        # ╔═══════════════╗
        # ║ Vector search ║
        # ╚═══════════════╝
        _t = perf_counter()
        # r = client.query_points(
        #     collection_name=collection_name,
        #     query=_query_vec,
        #     query_filter=None,
        #     limit=limit,
        # ).points

        # Look at limit * 3 for each augmented queries, concat, and take the top
        _mixed_vec = []
        for qv in _q:
            _r = client.query_points(
                collection_name=collection_name,
                query=qv,
                query_filter=None,
                limit=limit_vector * 3,
            ).points
            _mixed_vec.extend(_r)

        # Get unique results
        _filt = []
        _s = set()
        for x in sorted(_mixed_vec, key=lambda x: x.score, reverse=True):
            if x.id not in _s:
                _filt.append(x)
                _s.add(x.id)
        r = _filt[:limit_vector]

        _delta = perf_counter() - _t
        console.print(f"[cornflower_blue]VECTOR SEARCH took {_delta:.4f}s[/]")
        print_rez(console, r[:show_top])

        # ╔═══════════╗
        # ║ Reranking ║
        # ╚═══════════╝
        _t = perf_counter()
        # _docs = [x.payload["hn_text"] for x in r if x.payload is not None]
        _docs = _docs_bm25 + [x.payload["hn_text"] for x in r if x.payload is not None]
        scores = reranker(query, _docs)
        rr = sorted(list(zip(scores, r)), key=lambda x: x[0], reverse=True)

        _delta = perf_counter() - _t
        console.print(f"[cornflower_blue]RERANKING took {_delta:.2f}s[/]")
        print_rez_rr(console, rr[:show_top])


def main() -> None:
    reranker = Reranker()
    # reranker = RerankerExt()  # is it really faster? unclear as we can't run the same models
    with httpx.Client() as http_client:
        mainloop(http_client, reranker)


if __name__ == "__main__":
    main()
