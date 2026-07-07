"""TTS engine abstraction for swara.

Provides a clean ``TTSEngine`` interface with two implementations:

* :class:`MockTTSEngine` -- deterministic, stdlib-only. Generates a short valid
  WAV of a sine tone whose duration scales with text length. No external deps,
  so tests and CI never need the heavy AI4Bharat model.
* :class:`ParlerTTSEngine` -- the real engine wrapping AI4Bharat's
  ``ai4bharat/indic-parler-tts``. Heavy imports (transformers/torch/soundfile)
  are done lazily, only when the engine is instantiated.

The active engine is chosen by :func:`get_engine`, driven by the ``SWARA_ENGINE``
environment variable (default ``mock`` so the app always works out of the box).
"""

from __future__ import annotations

import io
import math
import os
import struct
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    """A supported language with a code, display name and default voice."""

    code: str
    name: str
    voice: str


# 13 languages: 12 Indian languages + English.
SUPPORTED_LANGUAGES: list[Language] = [
    Language("hi", "Hindi", "Rohit"),
    Language("ta", "Tamil", "Jaya"),
    Language("bn", "Bengali", "Aditi"),
    Language("te", "Telugu", "Prakash"),
    Language("mr", "Marathi", "Sanjay"),
    Language("gu", "Gujarati", "Yash"),
    Language("kn", "Kannada", "Suresh"),
    Language("ml", "Malayalam", "Anjali"),
    Language("pa", "Punjabi", "Divjot"),
    Language("or", "Odia", "Manas"),
    Language("as", "Assamese", "Amit"),
    Language("ur", "Urdu", "Rehan"),
    Language("en", "English", "Mary"),
]

LANGUAGE_CODES: set[str] = {lang.code for lang in SUPPORTED_LANGUAGES}
_LANGUAGE_BY_CODE: dict[str, Language] = {lang.code: lang for lang in SUPPORTED_LANGUAGES}

# Audio format constants shared across engines.
SAMPLE_RATE = 22050
SAMPLE_WIDTH = 2  # 16-bit PCM
N_CHANNELS = 1


class UnsupportedLanguageError(ValueError):
    """Raised when an unknown language code is requested."""


def get_language(code: str) -> Language:
    """Return the :class:`Language` for ``code`` or raise."""
    try:
        return _LANGUAGE_BY_CODE[code]
    except KeyError as exc:  # pragma: no cover - trivial
        raise UnsupportedLanguageError(
            f"Unsupported language {code!r}. Supported: {sorted(LANGUAGE_CODES)}"
        ) from exc


class TTSEngine(ABC):
    """Abstract text-to-speech engine."""

    #: Human-readable identifier for the engine.
    name: str = "abstract"

    @abstractmethod
    def synthesize(self, text: str, language: str, voice: str | None = None) -> bytes:
        """Synthesize ``text`` in ``language`` and return WAV bytes."""

    @staticmethod
    def _validate(text: str, language: str) -> Language:
        if not text or not text.strip():
            raise ValueError("text must be a non-empty string")
        return get_language(language)


class MockTTSEngine(TTSEngine):
    """Deterministic stdlib-only engine.

    Produces a mono 16-bit PCM WAV containing a sine tone. Duration scales with
    text length so that longer text reliably yields longer audio, which the test
    suite relies on. Pitch is derived from the language code so different
    languages produce distinguishable (but deterministic) output.
    """

    name = "mock"

    # Timing model: a small base plus a per-character increment, clamped.
    _BASE_SECONDS = 0.30
    _SECONDS_PER_CHAR = 0.06
    _MAX_SECONDS = 30.0

    def synthesize(self, text: str, language: str, voice: str | None = None) -> bytes:
        lang = self._validate(text, language)

        n_chars = len(text.strip())
        duration = min(
            self._BASE_SECONDS + n_chars * self._SECONDS_PER_CHAR, self._MAX_SECONDS
        )
        n_frames = int(SAMPLE_RATE * duration)

        # Base frequency depends on the language code (deterministic).
        base_freq = 180.0 + (sum(ord(c) for c in lang.code) % 12) * 15.0
        # Voice nudges the pitch a little, still deterministic.
        if voice:
            base_freq += (sum(ord(c) for c in voice) % 5) * 5.0

        amplitude = 0.35 * 32767
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(N_CHANNELS)
            wav.setsampwidth(SAMPLE_WIDTH)
            wav.setframerate(SAMPLE_RATE)
            frames = bytearray()
            two_pi_f = 2.0 * math.pi * base_freq / SAMPLE_RATE
            for i in range(n_frames):
                # Gentle fade in/out to avoid clicks.
                env = 1.0
                fade = int(SAMPLE_RATE * 0.02)
                if i < fade:
                    env = i / fade
                elif i > n_frames - fade:
                    env = max(0.0, (n_frames - i) / fade)
                sample = int(amplitude * env * math.sin(two_pi_f * i))
                frames += struct.pack("<h", sample)
            wav.writeframes(bytes(frames))
        return buf.getvalue()


class ParlerTTSEngine(TTSEngine):
    """Real engine wrapping AI4Bharat ``ai4bharat/indic-parler-tts``.

    All heavy dependencies (torch, transformers, soundfile) are imported lazily
    inside ``__init__`` so that merely importing this module -- or running the
    test suite with the mock engine -- never pulls in gigabytes of libraries or
    triggers a model download.

    The first instantiation downloads and caches the model via Hugging Face;
    subsequent runs work fully offline from the local cache.
    """

    name = "parler"

    MODEL_ID = "ai4bharat/indic-parler-tts"

    def __init__(self, model_id: str | None = None) -> None:
        # Lazy heavy imports -- only executed for the real engine.
        import torch  # type: ignore
        from transformers import AutoTokenizer  # type: ignore
        from parler_tts import ParlerTTSForConditionalGeneration  # type: ignore

        self._torch = torch
        self.model_id = model_id or self.MODEL_ID
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self._model = ParlerTTSForConditionalGeneration.from_pretrained(
            self.model_id
        ).to(self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        # The Parler description tokenizer may differ; fall back to the same one.
        desc_tok = getattr(self._model.config, "text_encoder", None)
        self._desc_tokenizer = AutoTokenizer.from_pretrained(
            getattr(desc_tok, "_name_or_path", self.model_id)
            if desc_tok is not None
            else self.model_id
        )

    def _description(self, language: str, voice: str | None) -> str:
        lang = get_language(language)
        speaker = voice or lang.voice
        return (
            f"{speaker} speaks in {lang.name} with a clear, natural and "
            f"expressive voice, at a moderate pace, recorded with high quality "
            f"and very little background noise."
        )

    def synthesize(self, text: str, language: str, voice: str | None = None) -> bytes:
        self._validate(text, language)
        import soundfile as sf  # type: ignore

        description = self._description(language, voice)
        desc_ids = self._desc_tokenizer(description, return_tensors="pt").to(
            self._device
        )
        prompt_ids = self._tokenizer(text, return_tensors="pt").to(self._device)

        with self._torch.no_grad():
            generation = self._model.generate(
                input_ids=desc_ids.input_ids,
                attention_mask=desc_ids.attention_mask,
                prompt_input_ids=prompt_ids.input_ids,
                prompt_attention_mask=prompt_ids.attention_mask,
            )
        audio = generation.cpu().numpy().squeeze()

        buf = io.BytesIO()
        sr = int(self._model.config.sampling_rate)
        sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
        return buf.getvalue()


def get_engine(name: str | None = None) -> TTSEngine:
    """Return a TTS engine.

    ``name`` overrides the ``SWARA_ENGINE`` environment variable. Defaults to
    ``mock`` so the app always runs without heavy dependencies.
    """
    selected = (name or os.environ.get("SWARA_ENGINE") or "mock").strip().lower()
    if selected == "mock":
        return MockTTSEngine()
    if selected == "parler":
        return ParlerTTSEngine()
    raise ValueError(
        f"Unknown SWARA_ENGINE {selected!r}. Use 'mock' or 'parler'."
    )
