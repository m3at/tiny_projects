#!/usr/bin/env python3

import argparse
import json
import logging
import re
import sys
import tarfile
from itertools import islice
from pathlib import Path
from typing import Literal

import bm25s
import httpx
import Stemmer
from pydantic import BaseModel
from qdrant_client import QdrantClient, models
from rich.progress import track

logger = logging.getLogger("base")


class Row(BaseModel):
    hn_text: str
    company: str
    locations: list[str | None]
    visa: bool | None
    remote: Literal["onsite", "hybrid", "remote", "world-wide remote"] | None
    application: str | None
    sector: str | None
    technologies: list[str | None]
    job_title: str | None
    seniority: str | None
    salary: float | None
    currency: str | None
    incentives: str | None
    employment: Literal["full-time", "part-time", "contractor", "internship"] | None
    salary_usd: float | None


def get_data() -> list[Row]:
    p = Path("./combined_processed.json.tar.gz")
    assert p.exists(), f"Could not find file {p}"

    with tarfile.open(p, "r:gz") as tar:
        t = tar.extractfile(tar.getmembers()[0])
        assert t is not None
        j = json.loads(t.read())
        rows = []
        for x in track(j, "decoding", transient=True):
            rows.append(Row(**x))
        return rows


def extract_float(sentence, *, re_prices=re.compile(r"\b\d+\.\d+|\b\d+\b")):
    # TODO: show example of using LLM to extract consistent price format
    sentence = sentence.lower()
    if "free" in sentence:
        return 0
    match = re_prices.search(sentence)
    return 0 if not match else float(match.group())


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

    logger.error(f"Request failed with status code {response.status_code}")
    raise ValueError("bad kitty, bad")


def batched(iterable, n):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch


def generate_qdrant() -> None:
    logger.info("Loading data")

    _metadata = get_data()
    docs = [r.hn_text for r in _metadata]
    metadata = [m.model_dump() for m in _metadata]

    # client = QdrantClient(":memory:")
    client = QdrantClient(path="db.qdrant")
    collection_name = "hn_jobs"

    logger.info("Pinging embedding endpoint")
    with httpx.Client() as http_client:
        t = get_embedding(http_client, "All work and no play makes Jack a dull boy")
        _size = len(t)

    try:
        client.get_collection(collection_name)
    except ValueError:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=_size,
                # distance=models.Distance.COSINE,
                distance=models.Distance.DOT,
            ),
        )

    # logger.info(f"Generating embeddings and upserting data for {len(docs)} entries...")
    logger.info("Generating embeddings")
    # client.add(
    #     collection_name="demo_collection",
    #     documents=docs,
    #     metadata=metadata,
    #     # ids=list(range(len(docs))),
    # )

    _max_docs = min(3000, len(docs))
    points = []
    # TODO: parallelize calls
    with httpx.Client() as http_client:
        for idx, (_doc, _meta) in track(
            enumerate(zip(docs, metadata[:_max_docs])), "embedding", total=_max_docs
        ):
            t = get_embedding(http_client, _doc[:512])
            points.append(models.PointStruct(id=idx, vector=t, payload=_meta))

    logger.info(f"Upserting data for {len(points)} entries...")
    client.upload_points(
        collection_name=collection_name,
        points=points,
    )

    logger.info("Trying a query")
    with httpx.Client() as http_client:
        hits = client.query_points(
            collection_name=collection_name,
            query=get_embedding(http_client, "alien invasion"),
            limit=3,
        ).points

    for hit in hits:
        # logger.debug(hit.payload, "score:", hit.score)
        if (p := hit.payload) is None:
            continue
        s = "{}\n{}".format(p["company"], p["hn_text"][:128])
        logger.debug(f"score: {hit.score:.4f}\n{s}")

    logger.info("Done")


def generate_bm25() -> None:
    # TODO: load only once
    logger.info("Loading data")
    _metadata = get_data()
    docs = [r.hn_text for r in _metadata]
    # metadata = [m.model_dump() for m in _metadata]

    logger.info("Stem + BM25")
    # TODO: pre-process text to prevent warnings like:
    # SyntaxWarning: invalid escape sequence '\w'
    stemmer = Stemmer.Stemmer("english")
    corpus_tokens = bm25s.tokenize(docs, stopwords="en", stemmer=stemmer)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    logger.info("Test query")
    query = "alien invasion"
    query_tokens = bm25s.tokenize(query, stemmer=stemmer)

    # Get top-k results as a tuple of (doc ids, scores). Both are arrays of shape (n_queries, k)
    results, scores = retriever.retrieve(query_tokens, corpus=docs, k=2)

    for i in range(results.shape[1]):
        doc, score = results[0, i], scores[0, i]
        print(f"Rank {i+1} (score: {score:.2f}): {doc}")

    # Save the arrays to a directory
    retriever.save("bm25_index")


def infer_bm25() -> None:
    logger.info("Checking BM25 inference")
    # At inference:
    # import bm25s
    # import Stemmer
    retriever = bm25s.BM25.load("bm25_index", load_corpus=False)
    stemmer = Stemmer.Stemmer("english")

    query = "alien invasion"
    query_tokens = bm25s.tokenize(query, stemmer=stemmer)
    # Scores are sorted from higher (most relevant) to lover (least relevant)
    # Shape: indexes: list[list[int]] , scores: list[list[float]]
    indexes, scores = retriever.retrieve(query_tokens, k=5)
    logger.debug(f"Indexes: {indexes[0]}")
    logger.debug(f"Scores: {scores[0].round(3)}")

    # DEBUG
    # client = QdrantClient(path="db.qdrant")
    # collection_name = "hn_jobs"
    # rrr = client.retrieve(collection_name, indexes[0].tolist())
    # # print(rrr)
    # print([x.payload["hn_text"] for x in rrr])


def main() -> None:
    # generate_qdrant()
    # generate_bm25()
    infer_bm25()


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
        _m = logging.getLevelNamesMapping()
        for c, lvl in [[226, "DEBUG"], [148, "INFO"], [208, "WARNING"], [197, "ERROR"]]:
            logging.addLevelName(_m[lvl], f"\x1b[38;5;{c}m{lvl:<7}\x1b[0m")

    main()
