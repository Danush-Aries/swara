"""Text extraction from uploaded documents.

Supports plain ``.txt`` (UTF-8) natively and ``.pdf`` via the optional
``pypdf`` dependency. The PDF path degrades gracefully: if ``pypdf`` is not
installed we raise a clear, user-facing :class:`ExtractionError` rather than
crashing, so the app (and its tests) run fine without the ``pdf`` extra.
"""

from __future__ import annotations

import io
import os


class ExtractionError(ValueError):
    """Raised for unsupported files or missing optional dependencies."""


def _extract_txt(data: bytes) -> str:
    # Try UTF-8 first, then fall back to a lenient decode.
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace")


def _extract_pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader  # lazy import of optional dependency
    except ImportError as exc:
        raise ExtractionError(
            "PDF support requires the 'pypdf' package. Install it with "
            "'uv pip install pypdf' or 'pip install swara[pdf]'."
        ) from exc

    try:
        reader = PdfReader(io.BytesIO(data))
        parts = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # noqa: BLE001 - surface a clean message
        raise ExtractionError(f"Could not read PDF: {exc}") from exc
    return "\n".join(parts).strip()


def extract_text(filename: str, data: bytes) -> str:
    """Extract text from ``data`` based on ``filename``'s extension.

    Supported: ``.txt`` (UTF-8) and ``.pdf`` (needs ``pypdf``). Raises
    :class:`ExtractionError` for anything else or when an optional dependency
    is missing.
    """
    ext = os.path.splitext(filename or "")[1].lower()
    if ext == ".txt":
        return _extract_txt(data)
    if ext == ".pdf":
        return _extract_pdf(data)
    raise ExtractionError(
        f"Unsupported file type {ext or '(none)'!r}. Upload a .txt or .pdf file."
    )
