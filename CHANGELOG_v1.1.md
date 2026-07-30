# MARROW — changes in this pass

## New: /settings
One place for everything that controls the AI rotation:
- **Fallback models per key** — pulls a *live* model list straight from
  the provider (OpenRouter, Groq, Gemini, Anthropic, OpenAI, Mistral),
  showing vision support / context window / free-or-paid for each one.
  Pick which models a key should use and in what order (e.g. `1,3,2`).
  Falls back to the built-in list with a warning if the live call fails.
- **Key priority** — reorder which key/provider gets tried first, second,
  etc. (e.g. `2,1,3`).
- **Theme** — moved here, still works via `/theme <name>` too.
- Adding a new key (`/keys add`) now offers the live model picker right
  away, so this isn't a separate thing you have to remember to configure.

## New: a menu after every video
Processing a video no longer drops you back to a bare prompt. You get:
open the report you just made, process another, ask a question, open
the library, or jump into Settings.

## New: /library shows every file, and can open one directly
Previously it only listed video titles. It now shows which of
PDF/HTML/MD exist for each video, and lets you pick a number to open one
— no more hunting through folders by hand.

## Fixed
- **Garbled/overlapping terminal output.** The status bar used to be
  pinned to the terminal's last row (prompt_toolkit's `bottom_toolbar`).
  Termux doesn't fully support that trick, so it collided with the
  engine's normal progress output. It's now a plain line that scrolls
  like everything else.
- **"1" typed at the main prompt did nothing useful.** The startup tips
  were numbered (1., 2., 3...), which reads like a menu but wasn't one.
  They're bullets now, and every prompt shows a one-line hint of what to
  type.
- **`Opening: .../search.html` instead of the actual report.** After
  processing a single (non-playlist) video, auto-open was opening the
  generic search tool instead of that video's own report. It now opens
  the video's own HTML report first (falling back to PDF, then MD).
- **Auto-open silently doing nothing on Termux.** Plain
  `webbrowser.open()` looks for a desktop opener (xdg-open/gio) that
  doesn't exist on Termux. It now tries `termux-open` first there, which
  hands the file to Android's normal app chooser.
- **Overview flowchart missing from the HTML report.** The PDF and
  Markdown reports both included it; the HTML report only ever embedded
  *section*-level flowcharts, so a video with just an overview flowchart
  (no section ones) showed none at all.
- **`API calls:   API calls: 3 (no limit)`** — duplicated label in the
  per-video summary box.
- **Misaligned startup config box.** `AI provider:` and `Global cache:`
  didn't line up in the same column as everything else above them.
- **Stale Groq model list** — one curated model
  (`meta-llama/llama-4-scout-17b-16e-instruct`) is no longer served by
  Groq; replaced with the current lineup (`openai/gpt-oss-120b/20b`,
  `llama-3.1-8b-instant`, `llama-3.3-70b-versatile`, `qwen/qwen3.6-27b`).
  This matters less now that fallback models are fetched live anyway.
- A little dead/unused code cleanup along the way (an always-no-op
  filter in the rotation logic, unused imports).
