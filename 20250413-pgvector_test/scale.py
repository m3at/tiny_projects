#!/usr/bin/env python3
import argparse
import logging
import random
import sys
import time
from typing import Final

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import TupleRow

DB_NAME: Final[str] = "vecdemo"
TABLE_NAME: Final[str] = "wonderland_scale"
DIMS: Final[int] = 128
N_VECTORS: Final[int] = 100_000
COPY_BATCH: Final[int] = 1_000  # rows streamed per COPY batch

logger = logging.getLogger("scale_test")


def _connect() -> psycopg.Connection[TupleRow]:
    return psycopg.connect(
        dbname=DB_NAME,
        autocommit=True,
        user="alice",
        host="localhost",
        port=5432,
    )


def prepare() -> None:
    t0 = time.perf_counter()

    conn = _connect()
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)

    logger.debug("Dropping & creating table …")
    conn.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
    conn.execute(
        f"""
        CREATE TABLE {TABLE_NAME} (
            id          BIGSERIAL PRIMARY KEY,
            content     TEXT,
            embedding   VECTOR({DIMS}),
            starts_with_i BOOLEAN
        )
        """  # pyright:ignore
    )

    # Bulk-insert random vectors
    rng = np.random.default_rng(42)
    total_written = 0
    cur = conn.cursor()
    logger.debug("Streaming %d rows via binary COPY …", N_VECTORS)
    with cur.copy(f"COPY {TABLE_NAME} (content, embedding, starts_with_i) FROM STDIN WITH (FORMAT BINARY)") as cp:
        cp.set_types(["text", "vector", "boolean"])

        buf_content: list[str] = []
        buf_emb: list[list[float]] = []
        buf_i: list[bool] = []

        for i in range(N_VECTORS):
            # Generate data
            content = f"dummy sentence {i}"
            emb = rng.random(DIMS, dtype=np.float32).tolist()
            starts_i = content.startswith("i")  # always False for our dummy strings

            buf_content.append(content)
            buf_emb.append(emb)
            buf_i.append(starts_i)

            if (i + 1) % COPY_BATCH == 0:
                for row in zip(buf_content, buf_emb, buf_i):
                    cp.write_row(row)
                total_written += len(buf_content)
                _ = buf_content.clear(), buf_emb.clear(), buf_i.clear()

        # flush remainder
        if buf_content:
            for row in zip(buf_content, buf_emb, buf_i):
                cp.write_row(row)
            total_written += len(buf_content)

    logger.debug("Inserted %d rows in %.2fs", total_written, time.perf_counter() - t0)

    # Index
    logger.info("Building HNSW index (inner-product opclass) …")
    t1 = time.perf_counter()
    conn.execute(f"CREATE INDEX ON {TABLE_NAME} USING hnsw (embedding vector_ip_ops)")
    logger.debug("Index built in %.2fs", time.perf_counter() - t1)

    conn.close()


def run_queries(sample_k: int = 5) -> None:
    conn = _connect()
    register_vector(conn)

    # pick K random rows to use their embeddings as queries
    ids = random.sample(range(1, N_VECTORS + 1), sample_k)
    rows = conn.execute(f"SELECT embedding, content FROM {TABLE_NAME} WHERE id = ANY(%s)", (ids,)).fetchall()

    sql = f"""
    SELECT (embedding <#> %s) * -1 AS score, content
    FROM {TABLE_NAME}
    ORDER BY score DESC
    LIMIT 5
    """

    logger.info("Running %d similarity lookups …", sample_k)
    for emb, original_text in rows:
        t0 = time.perf_counter()
        nearest = conn.execute(sql, (np.array(emb),)).fetchall()
        took = (time.perf_counter() - t0) * 1_000

        logger.info("Query for “%s…” (%.1f ms)", original_text[:20], took)
        for s, txt in nearest:
            logger.debug("  %7.3f  %s", s, txt)

    conn.close()


def main() -> None:
    prepare()
    run_queries()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bulk-load random vectors & test pgvector at scale",
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
    ch.setFormatter(logging.Formatter("{asctime} {levelname}│ {message}", datefmt="%H:%M:%S", style="{"))
    logger.addHandler(ch)

    if sys.stdout.isatty():
        _m = logging.getLevelNamesMapping()
        for c, lvl in [[226, "DEBUG"], [148, "INFO"], [208, "WARNING"], [197, "ERROR"]]:
            logging.addLevelName(_m[lvl], f"\x1b[38;5;{c}m{lvl:<7}\x1b[0m")

    main()
