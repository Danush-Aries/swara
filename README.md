# swara

**Free Indic text-to-speech for blind and low-vision students.**

`swara` (स्वर — "voice / tone") is an open-source accessibility app that reads
text aloud in **13 Indian languages** using [AI4Bharat](https://ai4bharat.iitm.ac.in/)'s
open text-to-speech models. It is built for students who are blind or have low
vision and need study material read to them in their own language — not just
English.

## Why

Most screen readers and TTS tools have poor or no support for Indian languages.
A student reading a Hindi, Tamil, or Bengali textbook is often stuck. `swara`
wraps AI4Bharat's `indic-parler-tts` model behind a simple, fully keyboard- and
screen-reader-accessible web page: paste text or upload a `.txt`/`.pdf`, pick a
language, and press **Speak** or **Download WAV**.

## Design: it always runs, model download optional

The real AI4Bharat model is a multi-gigabyte Hugging Face download. To keep the
app instantly runnable and the test suite fast and offline, `swara` uses a clean
`TTSEngine` abstraction with two implementations, selected by the `SWARA_ENGINE`
environment variable:

| `SWARA_ENGINE` | Engine | Dependencies | Behaviour |
|----------------|--------|--------------|-----------|
| `mock` (default) | `MockTTSEngine` | stdlib only | Deterministic sine-tone WAV; duration scales with text length. Great for dev, tests, CI, and demos. |
| `parler` | `ParlerTTSEngine` | `transformers`, `torch`, `soundfile` (extra) | Real AI4Bharat synthesis. First run downloads and caches the model, then works offline. |

This mirrors the testability approach used in `mcp-audit` and
`prompt-injection-lab`: heavy dependencies are lazy-imported and never required
to run the tests.

## Supported languages

Hindi, Tamil, Bengali, Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi,
Odia, Assamese, Urdu, and English.

## Quick start (mock engine — no heavy deps)

```bash
cd swara
uv run --extra dev swara         # starts on http://127.0.0.1:8000
```

Open <http://127.0.0.1:8000> and use the page. The default `mock` engine works
out of the box with zero model downloads.

### API

```bash
# List languages
curl http://127.0.0.1:8000/api/languages

# Synthesize (returns audio/wav)
curl -X POST http://127.0.0.1:8000/api/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"नमस्ते दुनिया","language":"hi"}' --output out.wav
```

## Real synthesis (AI4Bharat parler engine)

```bash
uv pip install -e '.[parler]'    # installs transformers, torch, soundfile
# plus the parler-tts package: pip install git+https://github.com/huggingface/parler-tts.git
SWARA_ENGINE=parler uv run swara
```

The first request downloads `ai4bharat/indic-parler-tts` and caches it under
your Hugging Face cache; subsequent runs are offline.

## Accessibility

The UI is built for screen-reader and keyboard users:

- Every control has a real `<label>`; a skip-to-content link is provided.
- Status updates use an `aria-live` region so speech is announced.
- High-contrast colours, large fonts, and large touch/click targets.
- Fully keyboard operable with a visible focus outline.
- No build step — plain HTML/CSS/vanilla JS.

## Development

```bash
uv run --extra dev pytest -q
```

Tests run entirely offline against `MockTTSEngine`.

## License

MIT © 2026 Dhanush Shankar. See [LICENSE](LICENSE).

Speech synthesis models are provided by AI4Bharat under their own licenses.
