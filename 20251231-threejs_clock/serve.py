#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "fastapi>=0.115.0",
#     "httpx>=0.28.1",
#     "uvicorn[standard]>=0.32.0",
# ]
# ///

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse

# Initialize FastAPI app and connection manager
app = FastAPI(title="Demo")

# Static files directory
static_dir = Path(__file__).parent / "static"


# Static file endpoints
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse((Path("static") / "favicon.ico"))


@app.get("/style.css", include_in_schema=False)
async def style():
    return FileResponse((Path("static") / "style.css"))


@app.get("/script.js", include_in_schema=False)
async def script():
    return FileResponse((Path("static") / "script.js"))


@app.get("/hdri/{filename:path}", include_in_schema=False)
async def hdri(filename: str):
    return FileResponse(static_dir / "hdri" / filename)


@app.get("/textures/{filename:path}", include_in_schema=False)
async def textures(filename: str):
    return FileResponse(static_dir / "textures" / filename)


@app.get("/")
async def get_index():
    index_path = static_dir / "index.html"
    return HTMLResponse(index_path.read_text())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8080)
