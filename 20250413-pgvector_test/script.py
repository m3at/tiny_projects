#!/usr/bin/env python3

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Final

import httpx
import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import TupleRow

logger = logging.getLogger("base")


try:
    TOKEN_JINA = os.environ["TOKEN_JINA"]
except KeyError:
    logger.warning("Please set your jina token in `.env`, example: TOKEN_JINA=1234 ")
    # exit(1)

TABLE_NAME: Final[str] = "wonderland"
DB_NAME: Final[str] = "vecdemo"
DIMS: Final[int] = 128


def get_embedding_local(client: httpx.Client, content: str, *, port: int = 8989) -> list[float]:
    """Local embeddings with llama.cpp."""

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


def get_embedding_jina(client: httpx.Client, content: list[str], *, dims: int = DIMS) -> list[list[float]]:
    """Embeddings from jina's api."""

    assert 32 <= dims <= 1024, f"jina-embeddings-v3 support dims in 32-1024, got {dims=}"

    response = client.post(
        "https://api.jina.ai/v1/embeddings",
        # Note: retrieval.query + retrieval.passage would give better embedding for retrieval, skipping for simplicity (aka laziness)
        json={"model": "jina-embeddings-v3", "task": "text-matching", "dimensions": dims, "input": content},
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {TOKEN_JINA}",
        },
        timeout=15,
    )

    if response.status_code == 200:
        embeddings = [x["embedding"] for x in response.json().get("data")]
        return embeddings

    logger.error(f"Request failed with status code {response.status_code}")
    raise ValueError("bad kitty, bad")


def get_sample_text_embeddings(*, dims: int = DIMS) -> list[tuple[str, list[float]]]:
    #     hwl = """\
    # I shot an arrow into the air,
    # It fell to earth, I knew not where;
    # For, so swiftly it flew, the sight
    # Could not follow it in its flight.
    # I breathed a song into the air,
    # It fell to earth, I knew not where;
    # For who has sight so keen and strong,
    # That it can follow the flight of song?
    # Long, long afterward, in an oak
    # I found the arrow, still unbroke;
    # And the song, from beginning to end,
    # I found again in the heart of a friend."""
    #
    #     # Cropped embeddings from jina-embeddings-v2-base-en, first 8
    #     vec = [
    #         [0.012, -0.049, 0.023, -0.018, -0.022, 0.017, 0.037, -0.014],
    #         [-0.034, -0.050, 0.079, 0.003, -0.014, -0.002, 0.058, 0.019],
    #         [-0.006, -0.025, 0.064, 0.048, -0.031, -0.011, 0.034, 0.009],
    #         [0.003, -0.028, 0.025, 0.036, -0.033, 0.031, 0.023, -0.001],
    #         [-0.019, -0.014, 0.024, 0.056, -0.027, 0.008, 0.010, 0.004],
    #         [-0.034, -0.050, 0.079, 0.003, -0.014, -0.002, 0.058, 0.019],
    #         [-0.034, 0.008, 0.068, 0.025, -0.031, -0.014, 0.006, -0.049],
    #         [-0.015, -0.003, 0.017, 0.062, -0.017, 0.013, 0.020, -0.036],
    #         [-0.004, -0.043, 0.073, 0.035, -0.032, 0.022, 0.009, -0.061],
    #         [0.001, -0.022, 0.019, 0.009, -0.054, 0.033, 0.030, 0.008],
    #         [-0.041, -0.011, 0.042, 0.051, -0.021, -0.007, -0.003, -0.027],
    #         [-0.010, -0.022, 0.074, 0.008, -0.011, 0.001, 0.016, -0.009],
    #     ]
    #
    #    return [(text, vec) for text, vec in zip(hwl.split("\n"), vec)]

    # jina-embeddings-v3 vectors, can use 32 to 1024 dims
    text = json.loads(Path("./sample.json").read_text())["input"]
    vec = [x["embedding"][:dims] for x in json.loads(Path("./embeddings.json").read_text())["data"]]

    return list(zip(text, vec))


def _connect() -> psycopg.Connection[TupleRow]:
    return psycopg.connect(dbname=DB_NAME, autocommit=True, user="alice", port=5432, host="localhost")


def prepare() -> None:
    logger.debug("Connecting and setting up extension")
    conn = _connect()
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)

    logger.debug("Creating table and adding vectors")
    conn.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
    conn.execute(
        f"CREATE TABLE {TABLE_NAME} (id bigserial PRIMARY KEY, content text, embedding vector({DIMS}), starts_with_i boolean)"  # pyright:ignore[reportArgumentType]
    )

    cur = conn.cursor()
    with cur.copy(f"COPY {TABLE_NAME} (content, embedding, starts_with_i) FROM STDIN WITH (FORMAT BINARY)") as copy:
        copy.set_types(["text", "vector", "boolean"])

        # Add all except the last few that we'll use as queries
        for content, embedding in get_sample_text_embeddings()[:-3]:
            starts_with_i = content.lower().startswith("i")
            copy.write_row([content, embedding, starts_with_i])

    logger.debug(
        "Creating an index to navigate a hierachy of navigable small words"
    )  # everyone knows what HNSW stands for, right?
    # conn.execute("SET maintenance_work_mem = '2GB'")
    # conn.execute("SET max_parallel_maintenance_workers = 7")
    # _operator = "vector_cosine_ops"
    _operator = "vector_ip_ops"
    conn.execute(f"CREATE INDEX ON {TABLE_NAME} USING hnsw (embedding {_operator})")

    logger.info("Prepared db")


def query() -> None:
    logger.debug("Trying some queries")
    conn = _connect()
    register_vector(conn)

    # First 8 floats of embedding for the word flying:
    # _flying_emb = [
    #     0.004221153,
    #     -0.024645567,
    #     0.030678282,
    #     0.02042206,
    #     -0.007014929,
    #     0.03867173,
    #     0.016879236,
    #     -0.008619699,
    # ]

    # Comparison operants:
    # <#> negative inner product
    # <-> L2
    # <=> cosine

    # Compare embeddings to the query vector (the %s that'll be replaced), and take the 5 closest
    sql_query = f"""\
    SELECT
        (embedding <#> %s) * -1 AS inner_product
      , content
    FROM {TABLE_NAME}
    ORDER BY inner_product DESC
    LIMIT 5
    """

    for content, embedding in get_sample_text_embeddings()[-3:]:
        result = conn.execute(sql_query, (np.array(embedding),)).fetchall()

        logger.info(f"Closest sentences to '{content}':\n" + "\n".join([f"{d:>7.5f} {t}" for d, t in result]) + "\n")

    logger.debug('Query with filtering, only lines starting with "i"')
    # Compare embeddings to the query vector (the %s that'll be replaced), and take the 5 closest
    sql_query = f"""\
    SELECT
        (embedding <#> %s) * -1 AS inner_product
      , content
    FROM {TABLE_NAME}
    WHERE starts_with_i
    ORDER BY inner_product DESC
    LIMIT 5
    """

    for content, embedding in get_sample_text_embeddings()[-3:-2]:
        result = conn.execute(sql_query, (np.array(embedding),)).fetchall()

        logger.info(f"Closest sentences to '{content}':\n" + "\n".join([f"{d:>7.5f} {t}" for d, t in result]) + "\n")


def main() -> None:
    prepare()
    query()


if __name__ == "__main__":
    # Get system arguments
    parser = argparse.ArgumentParser(
        description="Prepare db, add some vectors and try some queries",
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
    ch.setFormatter(logging.Formatter("{asctime} {levelname}│ {message}", datefmt="%H:%M:%S", style="{"))
    logger.addHandler(ch)

    # Add colors if stdout is not piped
    if sys.stdout.isatty():
        _m = logging.getLevelNamesMapping()
        for c, lvl in [[226, "DEBUG"], [148, "INFO"], [208, "WARNING"], [197, "ERROR"]]:
            logging.addLevelName(_m[lvl], f"\x1b[38;5;{c}m{lvl:<7}\x1b[0m")

    main()
