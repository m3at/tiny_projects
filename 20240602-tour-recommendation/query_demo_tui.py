#!/usr/bin/env python3

import re
from collections.abc import Generator
from typing import Tuple

from qdrant_client import QdrantClient, models
from qdrant_client.conversions.common_types import ScoredPoint
from qdrant_client.fastembed_common import QueryResponse
from rich.console import Console
from rich.prompt import Prompt


def loop(console: Console) -> Generator[str, None, None]:
    try:
        while True:
            console.print()
            query = Prompt.ask(
                "[green]Query[/]",
                console=console,
                default="horse riding",
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


def print_rez(console, r: list[QueryResponse] | list[ScoredPoint]):
    for idx, row in enumerate(r):
        if isinstance(row, QueryResponse):
            m = row.metadata
        else:
            m = row.payload
            if m is None:
                console.log("Panic, no payload")
                return

        loc = m["location"]
        doc = m["document"]
        title = m["title"]
        console.print(
            f"[{idx+1}] [yellow]{row.score:.2f} [/][underline]{title}[/] ({loc})\n{doc}"
        )


def main() -> None:
    client = QdrantClient(path="db.qdrant")
    console = Console()

    c = client.count("demo_collection")

    console.log(f"Starting search demo, on {c.count} entries")
    console.print(
        "Use natural language query. Or signed numbers to get recommendations based on similarity to previous results."
    )

    limit = 5
    embedding_model_name = "fast-bge-small-en"

    r = None

    for query in loop(console):
        # First pass
        if r is None:
            r = client.query(
                collection_name="demo_collection",
                query_text=query,
                query_filter=None,
                limit=limit,
            )

            print_rez(console, r)
            continue

        # TODO: use model to intelligently parse the query, extract filter from it

        # Either refine...
        pos, neg = extract_numbers(query, maxint=limit)

        if pos is not None and neg is not None:
            ids = [row.id for row in r]
            _pos = [ids[x - 1] for x in pos]
            _neg = [ids[abs(x) - 1] for x in neg]

            r = client.recommend(
                collection_name="demo_collection",
                positive=_pos,
                negative=_neg,
                # strategy=models.RecommendStrategy.AVERAGE_VECTOR,
                strategy=models.RecommendStrategy.BEST_SCORE,
                # query_filter=models.Filter(
                #     must=[
                #         models.FieldCondition(
                #             key="city",
                #             match=models.MatchValue(
                #                 value="London",
                #             ),
                #         )
                #     ]
                # ),
                limit=limit,
                # Will abort otherwise, bad defaults in the library
                using=embedding_model_name,
            )
            print_rez(console, r)
            continue

        # ... or process the query
        r = client.query(
            collection_name="demo_collection",
            query_text=query,
            query_filter=None,
            limit=limit,
            # with_vector=True,
        )
        print_rez(console, r)

        # name = Prompt.ask("Enter your name", choices=["Jessica", "Duncan"], default="Jessica", console=console)


if __name__ == "__main__":
    main()
