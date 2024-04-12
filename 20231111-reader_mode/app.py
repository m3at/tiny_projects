import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from bs4 import BeautifulSoup, Comment
from pathlib import Path
import re

# pip install html-sanitizer
# from html_sanitizer import Sanitizer
# pip install nh3
import nh3

# pip install readability-lxml
from readability import Document

CLEAN_NEWLINES = re.compile(r"[\t\n]+")


# uvicorn app:app --reload

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def read_item():
    html_content = Path("index.html").read_text()
    return HTMLResponse(content=html_content)


def is_relevant_image(tag):
    # Add additional heuristics for image relevance here
    if (
        tag.get("width")
        and int(tag.get("width")) < 100
        or tag.get("height")
        and int(tag.get("height")) < 100
    ):
        return False
    return True


def simplify_html(soup):
    body = soup.find("body")
    if not body:
        body = soup

    for tag in body.find_all(True):
        # Remove all attributes except for a few tags
        if tag.name not in [
            "a",
            "em",
            "strong",
            "u",
            "s",
            "blockquote",
            "ul",
            "ol",
            "li",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
        ]:
            tag.attrs = {}
        if tag.name == "a":
            tag.attrs = {"href": tag.get("href")}
        if tag.name in ["script", "style"]:
            tag.decompose()
        if tag.name == "img" and not is_relevant_image(tag):
            tag.decompose()

    # remove_empty_elements(body)
    # normalize_unicode(body)


def clean_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")

    simplify_html(soup)

    # Remove script and style elements
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()

    # Remove comment elements
    for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
        comment.extract()

    return str(soup)


@app.get("/fetch-content/")
async def fetch_content(url: str):
    try:
        # downloaded = trafilatura.fetch_url('https://github.blog/2019-03-29-leader-spotlight-erin-spiceland/')
        # trafilatura.extract(downloaded)
        # trafilatura.extract(downloaded, output_format="json", include_comments=False)
        # return page_text

        async with httpx.AsyncClient() as client:
            _rez = await client.get(url)
            page_text = _rez.text

            # page_text = nh3.clean("<b><img src=\"\">I'm not trying to XSS you</b>"))
            page_text = nh3.clean(page_text)

            doc = Document(page_text)
            # print(f"{doc.title()=}")
            page_text = doc.summary()

            page_text = CLEAN_NEWLINES.sub("", page_text)
            # re.sub(r'\n+', '\n', page_text)

            # You can add more cleaning or processing here if needed
            # page_text = clean_html(page_text)

            # sanitizer = Sanitizer()  # default configuration
            # cleaned_html = sanitizer.sanitize(str(cleaned_html))

            return page_text
    except httpx.RequestError as e:
        raise HTTPException(status_code=400, detail=f"Error fetching URL: {str(e)}")
