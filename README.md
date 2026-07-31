<div align="center">

# 🦴 MARROW

**Turn any YouTube video into complete, screenshot-rich notes — PDF, HTML, and Markdown — powered by your choice of AI provider, with automatic rotation across all of them.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-38%20passing-brightgreen)](tests/)

</div>

---

Paste a YouTube URL. MARROW pulls the transcript, downloads the video,
extracts the exact frames that matter (slides, diagrams, code, whiteboards),
runs OCR + AI vision on every one of them, and builds you a report you'd
actually want to read back — not just a wall of transcript text.

```
$ python marrow.py
› https://youtube.com/watch?v=XXXX
```

That's the whole interface. No config file to hand-edit, no API to wire up —
MARROW walks you through connecting a free AI provider key the first time
you run it, then remembers everything.

## Contents

- [Features](#features)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Supported AI providers](#supported-ai-providers)
- [Commands](#commands)
- [Batch / scripted use](#batch--scripted-use)
- [Project layout](#project-layout)
- [Configuration & data](#configuration--data)
- [Testing](#testing)
- [Known limitations](#known-limitations)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Interactive shell** — a Gemini-CLI-style chat interface (`python marrow.py`). Paste a URL, ask questions about videos you've already processed, or type `/help`.
- **6 AI providers, automatic rotation** — Google Gemini, OpenRouter (~20 free models), Groq, OpenAI, Anthropic, Mistral. Add as many keys as you want; MARROW rotates across every key + model automatically when one gets rate-limited, and remembers where it left off.
- **Actually reads the screen, not just the audio** — detects slide transitions, extracts clean unaltered frames, deduplicates near-identical ones (structural similarity, not naive pixel diff), then runs local OCR + AI vision on what's left.
- **Three output formats** — PDF, interactive HTML, and plain Markdown, with cropped diagrams, extracted code blocks, key takeaways, and optional quiz cards.
- **Resumable** — an interrupted run picks back up from where it stopped instead of starting over.
- **Ask your own library** — `/ask <question>` searches everything you've already processed and answers with citations back to the source video + timestamp. No re-processing needed.
- **Playlist & channel support**, with `--parallel N` to process several videos at once.
- **Local Whisper fallback** when a video has no captions.
- **Live progress throughout** — real download bars, ticking countdowns, and spinners at every step that used to just sit there silently.
- **Zero-setup first run** — missing Python packages get installed automatically, with plain-English errors instead of tracebacks.

## Quick start

```bash
git clone https://github.com/Sushantshendage/MARROW-AI.git
cd MARROW-AI
pip install -r requirements.txt   # or skip this — MARROW installs anything missing on first run
python marrow.py
```

You'll also need [ffmpeg](https://ffmpeg.org/download.html) on your `PATH`
(used for audio/video muxing) — `marrow.py` checks for it on first run and
tells you if it's missing.

First launch walks you through connecting at least one AI provider: pick one
from the list, grab a free key from the link it shows you, paste it in
(input is hidden), and MARROW verifies it with one lightweight test call.
Run `/keys` any time afterward to add, remove, or re-verify keys.

## How it works

```
YouTube URL
   │
   ├─ 1. Transcript + metadata (captions, or local Whisper if none exist)
   ├─ 2. AI pass 1 — structure & topic analysis of the transcript
   ├─ 3. Video download (only if visuals are needed)
   ├─ 4. Slide-transition detection (local, free)
   ├─ 5. Frame extraction at the moments that matter
   ├─ 6. Deduplication (structural similarity, time-windowed)
   ├─ 7. Local OCR + AI vision analysis on every surviving frame
   ├─ 8. Flowchart / diagram reconstruction (optional)
   └─ 9. Report generation — PDF / HTML / Markdown
```

Every step is designed to degrade gracefully — no captions? Falls back to
Whisper. No vision-capable model available? Falls back to OCR-only. A step
fails on one video in a playlist? The rest keep going.

## Supported AI providers

| Provider | Style | Notes |
|---|---|---|
| Google Gemini | native | generous free tier; Flash / Flash-Lite models |
| **OpenRouter** | OpenAI-compatible | one key unlocks ~20 free models from many labs, several vision-capable |
| Groq | OpenAI-compatible | fastest inference, generous free daily limit |
| OpenAI | OpenAI-compatible | mostly paid; included for people who already have a key |
| Anthropic Claude | native | vision-capable |
| Mistral | OpenAI-compatible | free tier |

Every key you add expands into one rotation slot per model that provider
offers. MARROW sticks with the current slot until it's actually rate-limited
(not just because it *could* switch), then rotates: next model on the same
key → next key → wraps back around once everything's been tried. A
vision-analysis call automatically skips any slot whose model can't see
images.

## Commands

Inside the interactive shell:

```
<paste a YouTube URL>   process it — classic flags still work after it,
                         e.g. "URL --no-vision --quiz"
<a plain question>      searched against your saved notes (same as /ask)
/keys                   add / remove / verify AI provider keys
/library                list videos you've already processed
/ask <question>         ask a question across your saved notes
/settings                fallback models, key priority, theme
/theme <name>           ember (default) · aurora · mono
/raw <flags>            pass anything straight to the batch engine —
                         /raw --help shows every original flag
/version, /clear, /help, /quit
```

## Batch / scripted use

The classic one-shot invocation still works if you'd rather script it:

```bash
python marrow.py "https://youtube.com/watch?v=XXXX" --no-vision --quiz
python marrow_engine.py "https://youtube.com/watch?v=XXXX"   # equivalent — calls the engine directly

# a few flags worth knowing about
python marrow_engine.py "URL" --format pdf          # pdf | html | md | both
python marrow_engine.py "PLAYLIST_URL" --parallel 3 # process 3 videos at once
python marrow_engine.py --ask "what did that video say about X?"
```

Run `python marrow_engine.py --help` for the full flag reference.

## Project layout

```
marrow.py                interactive shell — start here
marrow_engine.py          the processing pipeline (download → extract → analyze → build)
marrow_providers.py       AI provider registry + rotation manager
marrow_config.py          reads/writes ~/.marrow/config.json
marrow_ui.py              theme, banner, panels, live progress bars/spinners
marrow_logging.py         rotating file logger (~/.marrow/logs/marrow.log)
marrow_installer.py       first-run dependency auto-install
marrow_setup_wizard.py    the "connect a provider" flow (first run + /keys)
marrow_settings.py        in-shell settings (fallback models, key priority, theme)
tests/                    pytest suite — config, rotation, and dedup logic
requirements.txt
LICENSE
```

## Configuration & data

Everything lives under `~/.marrow/`:

- `~/.marrow/config.json` — your keys, theme, and rotation position. Created
  with owner-only (`0600`) permissions from the moment it's written, same
  pattern as `~/.aws/credentials`. Every read-modify-write runs under a lock,
  so `--parallel` mode can't have two threads clobber each other's state.
- `~/.marrow/cache/` — your processed-video library (this is what `/library`
  and `/ask` search). Point `--global-cache-dir` elsewhere if you'd rather
  keep it somewhere else.
- `~/.marrow/logs/marrow.log` — rotated, capped at a few MB. Debug output for
  best-effort steps (cleanup, optional metadata) that intentionally never
  interrupt a run but are worth checking if something keeps quietly failing.

## Testing

```bash
pip install pytest
pytest tests/
```

38 tests, no real network calls — covers config persistence (atomic writes,
permissions, concurrency), the rotation manager (sticky-until-limited,
rotate-on-429, cross-provider wraparound, vision-only filtering, cooldown
handling), and the frame-deduplication logic.

## Known limitations

- Free-tier limits and model line-ups shift often across every provider —
  what's in `marrow_providers.py` (`PROVIDERS` dict) reflects a point in
  time; check it against each provider's current docs occasionally.
- No auto-updater — "upgrading" means pulling a newer copy of these files.
  Your config and library are untouched; MARROW just shows a small
  "updated vX → vY" note when it notices the version changed.
- See `CHANGELOG_v1.2.0.md` and `PRODUCTION_READINESS.md` for the full
  history of what's been audited, fixed, and left as a deliberate trade-off.

## Contributing

Issues and PRs welcome. If you're adding a new AI provider: OpenAI-compatible
ones are a ~6-line addition to the `PROVIDERS` dict in `marrow_providers.py`;
native ones (Gemini/Anthropic-shaped APIs) need one small adapter function.
Please run `pytest tests/` before opening a PR.

## License

MIT — see [LICENSE](LICENSE).

