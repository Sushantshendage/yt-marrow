# MARROW

An interactive CLI for turning YouTube videos into comprehensive notes (PDF +
web page) — same processing engine as before, redesigned with a Gemini-CLI-style
interactive shell, and upgraded to talk to **multiple AI providers with automatic
rotation** instead of only Gemini.

## What changed from `video_notes_extractor_5.py`

| | Before | Now |
|---|---|---|
| Interface | one-shot batch script (`python script.py <url>`) | interactive chat-style shell (`python marrow.py`), batch mode still works |
| AI backend | Gemini only, one key from `GEMINI_API_KEY` | Gemini + OpenRouter + Groq + OpenAI + Anthropic + Mistral, any number of keys |
| On rate limit | falls back through a hardcoded Gemini model list | rotates across **every** key+model you've connected, any provider |
| First run | you had to have all pip packages pre-installed | auto-installs anything missing, explains errors in plain English |
| Look | plain `print()` output | banner, colored panels, bordered input, live status bar |
| During processing | silent for however long a step took (downloads, AI calls, Whisper) | live spinners, ticking countdowns, and a real download progress bar throughout |
| Long videos | local dedup could silently drop most of your requested screenshots | fixed — see `CHANGELOG_v1.2.0.md` |

Your existing usage still works exactly as before — see [Batch/scripted use](#batchscripted-use).

Latest changes: `CHANGELOG_v1.2.0.md`. Previous: `CHANGELOG_v1.1.1.md`,
`CHANGELOG_v1.1.md`.

## Setup

```bash
pip install -r requirements.txt   # or let MARROW do this for you on first run
python marrow.py
```

Want to verify the rotation/config logic on your machine before trusting
it with real keys? `pip install pytest && pytest tests/` — 38 tests, no
network calls, runs in well under a second.

First launch walks you through connecting at least one AI provider:

1. Pick a provider from the numbered list
2. Get a free key from the link it shows you, paste it in (input is hidden)
3. MARROW verifies it with one lightweight test call
4. Add another, or type done

You can add **as many keys as you want** — multiple keys from the same
provider, or a mix of providers. Every one becomes a slot in the rotation.
Run `/keys` any time afterward to add, list, remove, or re-verify keys.

## Providers included

| Provider | Style | Notes |
|---|---|---|
| Google Gemini | native | generous free tier; Flash / Flash-Lite models |
| **OpenRouter** | OpenAI-compatible | one key unlocks ~20 free models from many labs, several vision-capable — the deepest rotation pool from a single key |
| Groq | OpenAI-compatible | fastest inference, generous free RPD |
| OpenAI | OpenAI-compatible | mostly paid; included for people who already have a key |
| Anthropic Claude | native | vision-capable |
| Mistral | OpenAI-compatible | free tier |

Free-tier limits and model line-ups shift constantly across every provider —
what's listed in `marrow_providers.py` (`PROVIDERS` dict near the top) was
current as of July 2026. If a model gets renamed or a provider changes its
free lineup, just edit that one dict — no other file needs to know about it.
Adding a brand-new *OpenAI-compatible* provider is a 6-line addition; native
providers (Gemini/Anthropic-shaped) need one small adapter function.

## How rotation works

Every key you add expands into one slot per model that provider offers, in
this order: **key 1's models (in order), then key 2's models, then key
3's…**, wrapping back to key 1 after the last one.

MARROW sticks with the current slot until it actually gets rate-limited
(HTTP 429/503) — it doesn't rotate just because it *could*. When a slot gets
limited:
1. Try the next model on the **same key** first.
2. Once every model on that key is exhausted, move to the **next key**
   (which may be a different provider entirely).
3. Once *everything* has been tried, wrap back around to key 1's first
   model and keep going (limits reset over time, so this isn't wasted).

A screenshot-analysis call automatically skips any slot whose model can't
see images — it only ever lands on a vision-capable model, regardless of
where the rotation cursor currently is.

The current cursor position is saved in `~/.marrow/config.json`, so restarting
MARROW doesn't reset back to hammering your first provider every time.

This logic lives entirely in `marrow_providers.py::call_ai_rotating()` — the
4,600-line processing engine didn't need to change beyond one small hook,
so nothing about *how* videos get processed changed at all.

## Commands (inside the shell)

```
<paste a YouTube URL>   process it (any classic flag still works after it,
                         e.g. "URL --no-vision --quiz")
<a plain question>      searched against your saved notes (same as /ask)
/keys                   list / add / remove / verify AI provider keys
/library                list videos you've already processed
/ask <question>         ask a question across your saved notes
/theme <name>           ember (default) · aurora · mono
/raw <flags>            pass anything straight to the batch engine, e.g.
                         /raw --help  to see every original flag
/version                show the installed MARROW version
/clear, /help, /quit
```

## Batch/scripted use

Nothing about the original flag set changed — either call still works:

```bash
python marrow.py "https://youtube.com/watch?v=XXXX" --no-vision --quiz
python marrow_engine.py "https://youtube.com/watch?v=XXXX"   # equivalent, calls the engine directly
```

If you run it this way without ever having set up `/keys`, it falls back to
asking for a single bare Gemini key (old behavior) — MARROW's multi-provider
system is additive, not a breaking requirement.

## Where things live

```
marrow.py                the interactive shell (start here)
marrow_engine.py          the processing pipeline (was video_notes_extractor_5.py)
marrow_providers.py       provider registry + rotation manager
marrow_config.py          reads/writes ~/.marrow/config.json
marrow_logging.py         shared file logger (~/.marrow/logs/marrow.log)
marrow_installer.py       first-run dependency auto-install
marrow_setup_wizard.py    the "connect a provider" flow (first run + /keys)
marrow_ui.py              banner, colors, panels, bordered input, status bar
tests/                    pytest suite for config persistence + rotation logic
PRODUCTION_READINESS.md   audit findings, what was fixed, what's still a known trade-off
```

`~/.marrow/config.json` holds your keys, theme, and rotation position —
0600 permissions from the moment the file is created (not chmod'd after
the fact), owner-only, plain JSON (same pattern as `~/.aws/credentials`
or `gh`'s config — MARROW is a local tool, there's no server-side secrets
store to hand this off to). Every read-modify-write to this file runs
under a lock, so `--parallel` mode (several videos processed on separate
threads at once) can't have two threads clobber each other's rotation
state. `~/.marrow/cache/` holds your processed-video library (renamed
from `~/.video_notes_extractor_cache/` — if you have an old cache there,
either point `--global-cache-dir` at it or copy its contents into the
new folder). `~/.marrow/logs/marrow.log` (rotated, capped at a few MB)
holds debug output for the "best effort, don't crash the run" failure
paths — cleanup steps, optional metadata fetches, etc. — that intentionally
never interrupt a video, but are worth knowing about if one keeps failing.

## Honest limitations of this pass

- The provider list's exact model IDs and free-tier limits move fast
  industry-wide — double check `marrow_providers.py` against each
  provider's docs every so often.
- Key verification and the rotation logic (sticky-until-limited,
  rotate-on-429, cross-provider wraparound, vision-only filtering,
  cooldown skipping, invalid-key handling) are covered by `tests/`
  (`pytest tests/` — 38 tests, all passing, no real network calls made).
  Config persistence, its atomic-write/permission behavior, and the
  concurrency lock have their own tests too. I still couldn't make a
  *live* call to Gemini/OpenRouter/Groq/etc. from this sandbox to confirm
  against the real APIs end-to-end, so please sanity-check your first
  real run.
- "Upgrading" here means replacing these files with a newer copy — there's
  no package registry/auto-updater built (that would need a place to
  publish releases). Your config and library are untouched by an upgrade;
  MARROW just prints a small "updated vX → vY" note on the next run if it
  notices the version changed.
- See `PRODUCTION_READINESS.md` for the full audit: what was found, what
  was fixed, and a couple of things left as deliberate trade-offs rather
  than silently "fixed" — e.g. the config-file lock is per-process
  (covers `--parallel`'s threads) but not cross-process (two separate
  `marrow` invocations in two terminals at once aren't locked against
  each other; a rare scenario that self-heals on the next write either way).
