"""Offline test suite for swara.

Everything here runs against the deterministic :class:`MockTTSEngine`, so no
Hugging Face download or heavy dependency is ever required.
"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.extract import ExtractionError, extract_text
from app.tts import (
    SUPPORTED_LANGUAGES,
    MockTTSEngine,
    get_engine,
)


def _is_wav(data: bytes) -> bool:
    return data[:4] == b"RIFF" and data[8:12] == b"WAVE"


# --- Engine selection ------------------------------------------------------

def test_default_engine_is_mock(monkeypatch):
    monkeypatch.delenv("SWARA_ENGINE", raising=False)
    engine = get_engine()
    assert isinstance(engine, MockTTSEngine)
    assert engine.name == "mock"


# --- (a) synthesize returns valid WAV; longer text -> longer audio ---------

def test_synthesize_returns_wav_bytes():
    engine = MockTTSEngine()
    audio = engine.synthesize("नमस्ते दुनिया", "hi")
    assert isinstance(audio, bytes)
    assert len(audio) > 44  # more than just a header
    assert _is_wav(audio)


def test_longer_text_yields_longer_audio():
    engine = MockTTSEngine()
    short = engine.synthesize("Hi", "en")
    long = engine.synthesize("Hi " * 200, "en")
    assert len(long) > len(short)


def test_empty_text_rejected():
    engine = MockTTSEngine()
    with pytest.raises(ValueError):
        engine.synthesize("   ", "hi")


# --- (b) all advertised languages accepted ---------------------------------

@pytest.mark.parametrize("lang", SUPPORTED_LANGUAGES, ids=lambda l: l.code)
def test_all_languages_synthesize(lang):
    engine = MockTTSEngine()
    audio = engine.synthesize("test sentence", lang.code)
    assert _is_wav(audio)


def test_unknown_language_rejected():
    engine = MockTTSEngine()
    with pytest.raises(ValueError):
        engine.synthesize("hello", "zz")


# --- (c) + (d) API endpoints via TestClient --------------------------------

@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("SWARA_ENGINE", "mock")
    main_module._engine = None  # reset cached engine
    return TestClient(main_module.app)


def test_api_tts_returns_wav(client):
    res = client.post("/api/tts", json={"text": "hello world", "language": "en"})
    assert res.status_code == 200
    assert res.headers["content-type"] == "audio/wav"
    assert _is_wav(res.content)


def test_api_tts_bad_language(client):
    res = client.post("/api/tts", json={"text": "hi", "language": "zz"})
    assert res.status_code == 400


def test_api_languages_lists_at_least_13(client):
    res = client.get("/api/languages")
    assert res.status_code == 200
    data = res.json()
    assert data["engine"] == "mock"
    assert len(data["languages"]) >= 13
    codes = {l["code"] for l in data["languages"]}
    for expected in ("hi", "ta", "bn", "te", "en"):
        assert expected in codes


def test_index_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "swara" in res.text.lower()


# --- (e) extract_text: .txt always; .pdf graceful when pypdf missing -------

def test_extract_txt():
    text = extract_text("notes.txt", "नमस्ते\nhello".encode("utf-8"))
    assert "नमस्ते" in text
    assert "hello" in text


def test_extract_unsupported_type():
    with pytest.raises(ExtractionError):
        extract_text("image.png", b"\x89PNG")


def test_extract_pdf_graceful():
    """PDF extraction either works (pypdf installed) or raises a clean,
    informative ExtractionError -- it must never crash unexpectedly."""
    pypdf_available = importlib.util.find_spec("pypdf") is not None
    if pypdf_available:
        # A non-PDF payload should raise a clean ExtractionError, not crash.
        with pytest.raises(ExtractionError):
            extract_text("doc.pdf", b"not really a pdf")
    else:
        with pytest.raises(ExtractionError) as exc:
            extract_text("doc.pdf", b"anything")
        assert "pypdf" in str(exc.value).lower()
