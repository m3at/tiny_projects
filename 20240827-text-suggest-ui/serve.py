import logging
import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Final, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from openai import OpenAI
from openai.types import CompletionUsage
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# uvicorn serve:app --reload --port 8080
# ngrok http --domain=<See ngrok panel>.ngrok-free.app --basic-auth 'user:password'

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# As of 2024/07/24, price per 1M tokens
# https://openai.com/api/pricing/
_PRICES = {
    # Input, output
    "gpt-4o": (5, 15),
    "gpt-3.5-turbo-0125": (0.5, 1.5),
    "gpt-4o-mini": (0.15, 0.6),
}
# Prices are per 1M tokens
PRICES: Final[dict[str, Tuple[int, int]]] = {
    k: (v[0] / 1e6, v[1] / 1e6) for k, v in _PRICES.items()
}

MODEL: Final = "gpt-4o-mini"
assert MODEL in PRICES


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.ico")


@app.get("/diff.js", include_in_schema=False)
async def static_diff_js():
    return FileResponse("static/diff.js")


@app.get("/", response_class=HTMLResponse)
async def index_html():
    html_content = Path("index.html").read_text()
    return HTMLResponse(content=html_content)


class CorrectionPrompt(str, Enum):
    normal = "normal"
    aggressive = "aggressive"
    poetry = "poetry"
    translator = "translator"


class TransformRequest(BaseModel):
    original: str
    correction: str = CorrectionPrompt.normal


class TransformResponse(BaseModel):
    modified: str


def get_cost(usage: CompletionUsage, *, prices=PRICES[MODEL]) -> float:
    a, b = prices
    return (a * usage.prompt_tokens) + (b * usage.completion_tokens)


# SYSTEM_PROMPT = """\
# You rewrite the following content as poetry. You reply with the rewritten text only, without comments."""

SYSTEM_PROMPTS = {
    CorrectionPrompt.normal: """\
You are an expert writter, proficient in British English, fixing minor issues in a draft.
You don't add any comments or remark, but only rewrite the source text.
Rewrite the given content to improve grammar and fix spelling mistakes. Alter as little as necessary. Don't nitpick.""",
    #     CorrectionPrompt.more: """\
    # You are an expert writter, proficient in British English, improving a draft.
    # You don't add any comments or remark, but only rewrite the source text.
    # Rewrite the given content to improve grammar and fix spelling mistakes. Alter as little as necessary.""",
    CorrectionPrompt.aggressive: """\
You are an expert writter, proficient in British English, improving a draft.
You don't add any comments or remark, but only rewrite the source text.
Rewrite the given content to improve grammar, fix spelling mistakes and improve clarity.""",
    CorrectionPrompt.poetry: """\
You are a world-renowned poet, composing delightful rhymes in iambic pentameter.
Rewrite the given content as a poem, without preamble or comments. Make use of your poetic license!""",
    CorrectionPrompt.translator: """\
You are a professional translator, interpreting into idiomatic English.
In the context of an academic essay, translate the given text into English, without preamble or comments. Prefer British spelling.""",
}


@lru_cache(maxsize=64)
def get_llm_pred(original: str, correction: CorrectionPrompt) -> str:
    completion = client.chat.completions.create(
        model=MODEL,
        max_tokens=16382,
        temperature=0.2,
        timeout=30,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPTS[correction]},
            {"role": "user", "content": original},
        ],
    )

    modified_text = completion.choices[0].message.content

    # Log the API call usage
    # logger.info(f"API call usage: {completion.usage}")

    # Calculate and log the approximate cost (adjust the per-token cost as needed)
    usage = completion.usage

    assert usage is not None
    assert modified_text is not None

    estimated_cost = get_cost(usage)
    logger.info(f"Estimated API call cost: ${estimated_cost:.4f}")

    return modified_text


@app.post("/transform", response_model=TransformResponse)
async def transform(request: TransformRequest):
    # logger.info("request")
    logger.info(f"request, {request.correction=}")
    # logger.info(f"request\n\n{request.original}\n\n")
    try:
        modified_text = get_llm_pred(request.original, request.correction)
        return TransformResponse(modified=modified_text)

    except Exception as e:
        logger.error(f"Error during API call: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing the request")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
