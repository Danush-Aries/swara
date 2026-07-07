"""FastAPI application for swara.

Endpoints:
* ``GET  /api/languages`` -- list supported languages and default voices.
* ``POST /api/tts``       -- JSON {text, language, voice?} -> streamed audio/wav.
* ``POST /api/extract``   -- multipart file (.txt/.pdf) -> {text}.
* ``GET  /``              -- accessible single-page UI.

The active TTS engine is chosen by the ``SWARA_ENGINE`` env var (default
``mock``) so the server always starts without heavy dependencies.
"""

from __future__ import annotations

import io
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .extract import ExtractionError, extract_text
from .tts import (
    SUPPORTED_LANGUAGES,
    TTSEngine,
    UnsupportedLanguageError,
    get_engine,
)

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="swara",
    description="Indic text-to-speech accessibility app wrapping AI4Bharat TTS.",
    version="0.1.0",
)

# The engine is created lazily and cached so the (potentially heavy) parler
# model is only loaded once, on first synthesis request.
_engine: TTSEngine | None = None


def get_active_engine() -> TTSEngine:
    global _engine
    if _engine is None:
        _engine = get_engine()
    return _engine


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to synthesize.")
    language: str = Field(..., description="Language code, e.g. 'hi'.")
    voice: str | None = Field(None, description="Optional voice name.")


@app.get("/api/languages")
def list_languages() -> dict:
    """Return supported languages, their display names and default voices."""
    return {
        "engine": get_active_engine().name,
        "languages": [
            {"code": lang.code, "name": lang.name, "voice": lang.voice}
            for lang in SUPPORTED_LANGUAGES
        ],
    }


@app.post("/api/tts")
def tts(req: TTSRequest) -> StreamingResponse:
    """Synthesize speech and stream it back as ``audio/wav``."""
    engine = get_active_engine()
    try:
        audio = engine.synthesize(req.text, req.language, req.voice)
    except UnsupportedLanguageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return StreamingResponse(
        io.BytesIO(audio),
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="swara.wav"'},
    )


@app.post("/api/extract")
async def extract(file: UploadFile = File(...)) -> dict:
    """Extract text from an uploaded ``.txt`` or ``.pdf`` file."""
    data = await file.read()
    try:
        text = extract_text(file.filename or "", data)
    except ExtractionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"filename": file.filename, "text": text}


@app.get("/")
def index() -> FileResponse:
    """Serve the accessible single-page UI."""
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


def run() -> None:
    """Console-script entry point: launch uvicorn."""
    import os

    import uvicorn

    host = os.environ.get("SWARA_HOST", "127.0.0.1")
    port = int(os.environ.get("SWARA_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
