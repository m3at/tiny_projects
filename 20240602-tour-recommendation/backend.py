#!/usr/bin/env python

import re
import unicodedata
from copy import deepcopy
from pathlib import Path
from random import randint, shuffle
from time import perf_counter

import httpx
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI
from pydantic import BaseModel
from qdrant_client import QdrantClient, models
from qdrant_client.conversions.common_types import ScoredPoint
from qdrant_client.fastembed_common import QueryResponse

client = OpenAI()

qclient = QdrantClient(path="db.qdrant")
c = qclient.count("demo_collection")
embedding_model_name = "fast-bge-small-en"

app = FastAPI()

origins = [
    "*",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Item(BaseModel):
    id: str
    # text: str
    state: int


class Query(BaseModel):
    query: str


class QueryWithItems(BaseModel):
    query: str
    items: list[Item] = []


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/mockapi")
def mockapi(items: list[Item]):
    print(items)
    return {"ok": ", ".join(str(x.state) for x in items)}


def _make_prompts():
    messages = [
        {
            "role": "system",
            "content": "You are a search auto-completion engine for a travel agency, making suggestions and correcting typographical errors. You return 5 suggestions to complete the user's input. Only suggestions, nothing else. One suggestion per line.",
        },
    ]
    t = """\
paris bus to
Paris bus tour with Eiffel Tower visit
Paris bus tour to Versailles Palace
Paris hop-on hop-off bus tour
Paris bus tour with Seine River cruise
Paris night bus tour with light show

ny pizza
New York pizza
New York pizza tasting
New York pizza making class
New York pizza walking tour
New York pizza and brewery tour

bangkok temple
Bangkok temple tour
Bangkok temple and market tour
Bangkok temple and canal tour
Bangkok temple and cultural show
Bangkok temple and cooking class

suba divin class
Scuba diving class
Scuba diving class in the Great Barrier Reef
Scuba diving class in Hawaii
Scuba diving experience in the Bahamas
Scuba diving tour

kyoto tori
Kyoto torii gate tour
Kyoto torii shrine visit
Kyoto torii gates photography tour
Kyoto torii gates hike
Kyoto torii temples and shrines tour

gibli musem
Ghibli museum in Tokyo
Ghibli museum tickets
Ghibli museum guided tour
Ghibli museum family visit
Ghibli museum and Mitaka walk"""
    for chunk in t.split("\n\n"):
        lines = chunk.splitlines()
        messages.extend(
            [
                {
                    "role": "user",
                    "content": lines[0],
                },
                {
                    "role": "assistant",
                    "content": "\n".join(lines[1:]),
                },
            ]
        )

    return {
        "model": "Qwen2-1.5B-Instruct",
        "max_tokens": 128,
        # "temperature": 0.3,
        "temperature": 0.0,
        "cache_prompt": True,
        "messages": messages,
    }


d = _make_prompts()


# d = {
#     "model": "Qwen2-1.5B-Instruct",
#     "max_tokens": 128,
#     "temperature": 0.3,
#     "cache_prompt": True,
#     "messages": [
#         {
#             "role": "system",
#             "content": "You are a search auto-completion engine for an online marketplace, handling suggestions and typographical errors. You return 5 suggestions to complete the user's input text. Only suggestions, nothing else. One suggestion per line.",
#         },
#         {"role": "user", "content": "men's raz"},
#         {
#             "role": "assistant",
#             "content": """men's razors
# men's razor blades
# men's razors for sensitive skin
# men's razor subscription
# men's razor set""",
#         },
#         {"role": "user", "content": "2L sham"},
#         {
#             "role": "assistant",
#             "content": """2L shampoo
# 2L shampoo and conditioner
# 2L shampoo refill
# 2L shampoo for hair loss
# 2L shampoo organic""",
#         },
#         {"role": "user", "content": "charziard"},
#         {
#             "role": "assistant",
#             "content": """charizard action figure
# charizard trading card
# charizard plush toy
# charizard hoodie
# charizard phone case""",
#         },
#         {"role": "user", "content": "high gain mi"},
#         {
#             "role": "assistant",
#             "content": """high gain microphone
# high gain mic preamp
# high gain mini wifi adapter
# high gain midi interface
# high gain microphone for streaming""",
#         },
#         {"role": "user", "content": "pika tr"},
#         {
#             "role": "assistant",
#             "content": """pikachu trading card
# pikachu transformer toy
# pikachu travel mug
# pikachu tracksuit
# pikachu train set""",
#         },
#     ],
# }


@app.post("/autocomplete")
def autocomplete(q: Query) -> list[str]:
    # Serve separately with:
    # llamafile --model Qwen2-1.5B-Instruct.Q6_K.gguf --server --nobrowser --n-gpu-layers 999 --gpu APPLE -c 0 --parallel 1 --port 8080 --fast

    query = q.query
    d2 = deepcopy(d)
    d2["messages"] = d2["messages"] + [
        {"role": "user", "content": query},
    ]

    r = httpx.post("http://localhost:8080/v1/chat/completions", json=d2)
    data = r.json()

    suggestions = data["choices"][0]["message"]["content"].splitlines()
    # print(suggestions)
    return suggestions


def strip_accents(s: str) -> str:
    """adélie -> adelie"""
    # return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    return unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode()


answers = [
    "Rome, Italy",
    "Paris, France",
    "Tokyo, Japan",
    "Jersey Island, United Kingdom",
    "Guernsey Island, United Kingdom",
    "Kerguelen Islands, France",
    "Terrer Adélie, Antarctica",
    "Saint Pierre and Miquelon, France",
    "Clipperton Island, France",
    "Saint Barthélemy, France",
    "Wallis and Futuna, France",
]
ids = list(range(len(answers)))

PLACEHOLDER_IMAGE_URL = "https://placehold.co/320x180?text=16x9"


@app.post("/search")
# def search(q: QueryWithItems) -> list[dict[str, int | str]]:
def search(q: QueryWithItems) -> list[dict]:
    # Serve separately with:
    # llamafile --model Qwen2-1.5B-Instruct.Q6_K.gguf --server --nobrowser --n-gpu-layers 999 --gpu APPLE -c 0 --parallel 1 --port 8080 --fast

    query = q.query
    print(query)
    print(q.items)

    limit = 6

    # voted items
    voted_items = [i for i in q.items if i.state != 0]

    if len(voted_items) == 0:
        # Normal search
        r = qclient.query(
            collection_name="demo_collection",
            query_text=query,
            query_filter=None,
            limit=limit,
        )
    else:
        # Recommendation
        _pos = [i.id for i in voted_items if i.state > 0]
        _neg = [i.id for i in voted_items if i.state < 0]

        r = qclient.recommend(
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

    buff = []
    for idx, row in enumerate(r):
        if isinstance(row, QueryResponse):
            m = row.metadata
        else:
            m = row.payload
            assert m is not None

        # doc = m["document"]
        # i = m["idx"]

        place_name_org = "{}, {}".format(m["location"], m["country"])
        place_name = strip_accents(place_name_org.lower())
        place_name = re.sub(r"[^a-z]+", "_", place_name)
        # print(place_name)

        print(row.id)

        buff.append(
            {
                "id": row.id,
                "title": m["title"],
                "location": m["location"],
                # "highlights": m["highlights"],
                # "highlights": m["duration"],
                "highlights": m["document"],
                # "image_url": f"http://127.0.0.1:8000/cdn/{i % len(answers)}",
                "image_url": f"http://127.0.0.1:8000/cdn/{place_name}",
            }
        )

    # print(m)

    # r = list(range(len(answers)))
    # shuffle(r)
    # r = r[:4]
    # return [{"id": i, "text": t, "image_url": PLACEHOLDER_IMAGE_URL} for i, t in zip(r, a[:4])]
    # return [{"id": ids[i], "text": answers[i], "image_url": f"http://127.0.0.1:8000/cdn/{i}"} for i in r]

    return buff


_IMG_TYPE = "png"
# _IMG_TYPE = "jpg"

_IMG = Path(
    "/Users/meat/.cache/m3at/mock_cdn_prev/_danube_river_hungary.png"
).read_bytes()


@app.get(
    "/cdn/{image_name}",
    responses={200: {"content": {f"image/{_IMG_TYPE}": {}}}},
)
def image_cdn(image_name: str):
    # DEBUG
    # return Response(_IMG, media_type="image/png")

    place_name = image_name

    # place_name_org = answers[item_id % len(answers)]

    cache_dir = Path.home() / ".cache" / "m3at" / "mock_cdn"
    cache_dir.mkdir(exist_ok=True, parents=True)

    # place_name = strip_accents(place_name_org.lower())
    # place_name = re.sub(r"[^a-z]+", "_", place_name)

    p = cache_dir / f"{place_name}.{_IMG_TYPE}"

    if p.exists():
        return Response(p.read_bytes(), media_type=f"image/{_IMG_TYPE}")

    print(f"Did not find: {image_name}")
    return Response(_IMG, media_type="image/png")

    # t0 = perf_counter()
    # print(f"calling dall-e for {place_name}")
    # # dall-e-2, 256: $0.016 / image
    # # dall-e-3, 1024: $0.040 / image
    # response = client.images.generate(
    #     # model="dall-e-2", size="256x256",  # $0.016 / image, 9s
    #     model="dall-e-3",
    #     size="1024x1024",  # $0.040 / image, 17s
    #     # prompt=f"a wonderful toursim magazine cover of {place_name_org}",
    #     # prompt=f"an award winning landscape photography of {place_name_org}",
    #     prompt=f"high quality picture, award winning photography of {place_name}, detailed, daytime, aesthetic, magazine cover, 8k",
    #     quality="standard",
    #     n=1,
    # )
    #
    # image_url = response.data[0].url
    # assert image_url is not None
    # r = httpx.get(image_url)
    # b = r.content
    # p.write_bytes(b)
    #
    # delta = perf_counter() - t0
    # print(f"Took {delta:.2f}s, saved under: {p}")


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: str | None = None):
#     return {"item_id": item_id, "q": q}

if __name__ == "__main__":
    # or:
    # uvicorn backend:app --reload
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
