#!/usr/bin/env python3
"""
MARROW engine — Complete Video Understanding Tool
=============================================================
Extracts EVERYTHING from a YouTube video into comprehensive notes.
No information should be missed — the output mirrors the video content.

This is the processing engine behind the MARROW CLI (see marrow.py for
the interactive shell). It no longer talks to Gemini directly — every
AI call goes through marrow_providers.call_ai_rotating(), which rotates
across however many providers/models the user has connected via /keys.
This file can still be run standalone (`python marrow_engine.py <url>`)
for scripted/batch use; it falls back to a single bare GEMINI_API_KEY
if no MARROW provider config exists, so old workflows keep working.

MULTI-PROVIDER, WITH AUTOMATIC ROTATION ON RATE LIMITS:
  - Pass 1 uses 1 API call per video (transcript analysis)
  - Pass 2 batches 8 screenshots per API call (85% fewer calls)
  - Rate limiting enforced (6.5s between calls)
  - Local Tesseract OCR used when available (zero API calls for code/text)
  - Frame deduplication removes redundant screenshots before Vision
  - A typical video costs 3-6 API calls total

PIPELINE:
  1. Fetches transcript (with timestamps).
  2. PASS 1 — Transcript to Gemini (1 API call, temperature=0):
       • Detects video type (coding/slides/lecture/tutorial/trading/general)
       • Splits into sections with faithful, complete notes
       • AI determines exact screenshot timestamps (smarter than keywords)
       • Extracts code blocks from spoken content
       • Extracts table data
       • Generates Mermaid flowcharts for processes & overview
  3. Downloads video (auto quality: 720p coding/slides, 480p lectures).
  4. Slide detection via OpenCV frame differencing (FREE, local).
  5. Extracts frames at AI timestamps + slide transitions.
  6. Frame deduplication via SSIM (FREE, local — removes redundant shots).
  7. Local OCR via Tesseract if available (FREE — extracts code/text locally).
  8. PASS 2 — Batched Gemini Vision (8 images per call):
       • Only for frames that need complex visual analysis
       • Identifies ALL visual elements (diagrams, charts, code, formulas, UI)
       • Extracts on-screen code and table data
  9. Renders Mermaid flowcharts via mermaid.ink API (FREE).
  10. Builds PDF + Markdown reports with ALL content:
       • Complete notes per section
       • ALL screenshots as full, uncropped frames
       • Code blocks (copy-ready, monospace)
       • Data tables
       • Process flowcharts + overview flowchart
  11. Merges per-video PDFs into combined_report.pdf.

RESUME / CHECKPOINTING:
    Every network- or API-dependent step (transcript fetch, metadata fetch,
    Pass 1 analysis, video download, Pass 2 vision batches, extracted
    frames) is cached to disk inside each video's output folder as it
    completes. If a run is interrupted — lost internet, a hit API budget,
    or the process being killed — simply run the same command again: work
    already done is reused and the video picks up right where it stopped,
    instead of being reprocessed from scratch. Pass --fresh to ignore all
    of that and force a full reprocess.

    If a critical step fails outright (transcript never available, Pass 1
    AI analysis errors out, or the video download fails), that video STOPS
    right there instead of quietly finishing with missing/broken content —
    a report.pdf/html/md is only ever written from a fully-completed
    pipeline. The exact reason is printed, and a plain re-run of the same
    command retries only the failed step (everything before it stays
    cached). A playlist run still moves on to the remaining videos; the
    final summary lists exactly which ones need a re-run. Pass
    --degrade-on-error to opt back into the old behaviour of falling back
    to lower-quality output and continuing instead of stopping.

REQUIREMENTS:
    pip install yt-dlp youtube-transcript-api opencv-python-headless \\
        numpy pillow reportlab pypdf requests markdown pygments

    Optional (for free local OCR — reduces API calls further):
        pip install pytesseract
        + Install Tesseract: https://github.com/tesseract-ocr/tesseract

    System: ffmpeg must be installed and on PATH.

GEMINI API KEY (free):
    Get a free key: https://aistudio.google.com/app/apikey
    Then: export GEMINI_API_KEY="your-key-here"
    Or pass: --api-key your-key

USAGE:
    python marrow_engine.py <url> [options]

    # Single video (default: PDF + Markdown output)
    python marrow_engine.py "https://www.youtube.com/watch?v=XXXX"

    # Playlist
    python marrow_engine.py "https://www.youtube.com/playlist?list=XXXX" --max-videos 10

    # Budget mode — limit total API calls
    python marrow_engine.py "URL" --max-api-calls 15

    # Skip vision (only Pass 1 = 1 API call per video)
    python marrow_engine.py "URL" --no-vision

    # Zero API calls (keyword heuristics, local processing only)
    python marrow_engine.py "URL" --skip-ai

    # Interrupted earlier (internet dropped / API limit hit)? Just re-run the
    # same command — already-completed work resumes automatically.
    python marrow_engine.py "URL"

    # Ignore saved progress and reprocess everything from scratch
    python marrow_engine.py "URL" --fresh

OUTPUT:
    output/
      combined_report.pdf
      <video_id>/
        report.pdf
        report.md
        sections.json             <- Pass 1 analysis (resume checkpoint)
        transcript_cache.json     <- cached transcript (resume checkpoint)
        video_meta_cache.json     <- cached metadata (resume checkpoint)
        vision_cache.json         <- Pass 2 results so far (resume checkpoint)
        frame_05_32.png           <- full, uncropped screenshot
        overview_flowchart.png
        flowchart_section_0.png
"""

import re
import os
import sys
import json
import time
import uuid
import base64
import argparse
import subprocess
import shutil
import threading
import contextvars
import builtins
import webbrowser
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
import html as html_escape_module
from datetime import datetime
from pathlib import Path
import markdown

import cv2
import numpy as np
import requests
from PIL import Image as PILImage, ImageDraw, ImageFont
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled, NoTranscriptFound, VideoUnavailable,
)
import yt_dlp

try:
    import marrow_providers as mrp
except ImportError:
    mrp = None  # engine.py still works standalone with a bare GEMINI_API_KEY

try:
    from marrow_logging import get_logger
    _log = get_logger()
except ImportError:
    import logging as _logging
    _log = _logging.getLogger("marrow")
    _log.addHandler(_logging.NullHandler())

try:
    import marrow_ui as ui
except ImportError:
    ui = None  # engine.py still works standalone (plain text) without it


# ── Live-progress helpers used throughout the pipeline below ──────────
# Thin wrappers so call sites don't need an `if ui:` guard everywhere.
# Every one of these is a total no-op (identical to the old behaviour)
# when marrow_ui isn't importable, stdout isn't a real terminal, or a
# --parallel worker thread has opted out (see marrow_ui.live_ok()) — see
# marrow_ui.py's module docstring for why that matters.

class _NullDownloadProgress:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def hook(self, d):
        pass


def _spinner(label):
    if ui is not None:
        return ui.spinner(label)
    from contextlib import nullcontext
    return nullcontext()


def _download_progress(label="Downloading"):
    if ui is not None:
        return ui.DownloadProgress(label)
    return _NullDownloadProgress()


def _progress_iter(iterable, total, label):
    if ui is not None:
        return ui.progress_iter(iterable, total, label)
    return iter(iterable)


def _live_wait(seconds, label):
    if ui is not None:
        ui.live_wait(seconds, label)
    else:
        time.sleep(seconds)

# YouTube video IDs are always 11 chars of [A-Za-z0-9_-] in practice, but
# we accept a little slack either side rather than hardcode "exactly 11"
# in case that ever changes upstream. This exists purely as defense in
# depth: video_id normally comes straight from yt-dlp's own metadata
# (already validated against YouTube), and is then used to build local
# filesystem paths (`base_dir / video_id`). If a future yt-dlp version,
# a corrupted cache entry, or a hand-edited resume file ever produced
# something containing "/" or "..", this stops it from escaping the
# intended output directory instead of silently trusting it.
_VALID_VIDEO_ID = re.compile(r'^[A-Za-z0-9_-]{1,32}$')


def _require_safe_video_id(video_id):
    """Raises ValueError if video_id isn't safe to use as a bare path
    component. Returns video_id unchanged otherwise (so this can be used
    inline: `video_id = _require_safe_video_id(video_id)`)."""
    if not video_id or not _VALID_VIDEO_ID.match(str(video_id)):
        raise ValueError(
            f"Refusing to use {video_id!r} as a video ID — it doesn't look "
            "like a real YouTube video ID, and using it directly to build "
            "a filesystem path would be unsafe."
        )
    return video_id

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Image as RLImage, Paragraph, Spacer,
    Table, TableStyle, ListFlowable, ListItem, PageBreak,
    Preformatted, KeepTogether,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader, PdfWriter

# ── Optional: Tesseract OCR for FREE local code/text extraction ──
try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False


def open_file_for_viewing(path):
    """Opens `path` in whatever the platform considers 'the' viewer for it.
    Used both for --open at the end of a run and for opening a file
    straight from /library.

    Plain `webbrowser.open()` quietly does nothing on Termux (Android) —
    it looks for a desktop opener like xdg-open/gio, neither of which
    exists there — so a report generated on-device never actually opened
    even though the code reported success. Termux ships its own opener,
    termux-open, that hands the file to Android's normal "open with" app
    chooser instead, so that's tried first when available.

    Returns True if an opener was launched, False otherwise (the caller
    can then show the path so the person can open it by hand)."""
    path = Path(path).resolve()
    if os.environ.get('TERMUX_VERSION') or shutil.which('termux-open'):
        try:
            subprocess.run(['termux-open', str(path)], check=True,
                            capture_output=True, timeout=15)
            return True
        except Exception:
            pass  # fall through and try the regular way instead
    try:
        return bool(webbrowser.open(path.as_uri()))
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════
# ERROR HANDLING: stop-and-resume instead of "continue with broken output"
# ═══════════════════════════════════════════════════════════════════
#
# VideoProcessingError marks a STAGE-LEVEL failure that should stop THIS
# video's pipeline right where it is (transcript, Pass 1 analysis, any
# downloaded video, extracted frames, and vision results done so far all
# stay cached on disk exactly as-is). It is caught one level up, in
# main()'s per-video runner, which:
#   - prints the exact reason clearly (no silent fallback / no report
#     built from incomplete data),
#   - does NOT mark the video as done, so report.pdf/html/md are never
#     written from partial data and re-running the same command later
#     picks up from this exact point instead of starting over,
#   - moves on to the next video in a playlist rather than killing the
#     whole run.
# Pass --degrade-on-error to restore the old behaviour of falling back to
# lower-quality output (keyword heuristics / no screenshots) and
# continuing instead of stopping.
class VideoProcessingError(Exception):
    pass


# ── Thread-tagged logging for --parallel runs ────────────────────────
# When several videos are processed at once (--parallel > 1), their log
# lines interleave on the same terminal, which reads as garbled. Each
# worker thread gets its OWN contextvars.Context by default (that's a
# CPython guarantee — a value set inside one thread is invisible to
# others), so tagging the current video id here and shadowing the
# built-in print() to prepend it costs nothing at every one of this
# script's existing print() call sites, no need to touch each one
# individually.
_video_tag_ctx = contextvars.ContextVar('video_tag', default='')


def print(*args, sep=' ', end='\n', file=None, flush=False):
    tag = _video_tag_ctx.get()

    # Anything using a non-default end/file/flush (there are none left in
    # this file as of this version, but a future call site or an import
    # of this module's print() from elsewhere might) falls straight back
    # to plain builtins.print — the styled path below assumes one whole
    # line at a time.
    if end != '\n' or file is not None or ui is None or not ui.console.is_terminal:
        text = sep.join(str(a) for a in args)
        if tag:
            text = f"[{tag}] {text}"
        builtins.print(text, end=end, file=file, flush=flush)
        return

    text = sep.join(str(a) for a in args)
    ui.print_engine_line(text, tag=tag or None)


# ═══════════════════════════════════════════════════════════════════
# CONSTANTS & FREE TIER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

# Gemini Free Tier Limits (as of July 2026 — the whole 2.x-flash family
# is being sunset by Google and is already returning intermittent/hard
# 404 "no longer available" errors, so it's deliberately NOT used below):
#   gemini-3.5-flash:      ~10 RPM | 250K TPM | 1500 RPD | vision ✓ (GA, May 2026)
#   gemini-3.1-flash-lite: ~15 RPM | 1M TPM   | 1500 RPD | vision ✓ (GA, cheap/fast)
#   gemini-flash-latest:   experimental alias, Google auto-updates it to
#                          whatever their current best Flash model is —
#                          kept as the LAST fallback so this script keeps
#                          working even after 3.5/3.1 eventually get
#                          deprecated too, without needing another manual
#                          model-name update.
#
# Strategy: Use cheapest model for text (Pass 1), vision model only
# when needed (Pass 2), batch images to minimize calls.

TEXT_MODEL = "gemini-3.5-flash"       # Pass 1: transcript analysis (1 call/video)
VISION_MODEL = "gemini-3.5-flash"     # Pass 2: screenshot analysis (batched)

# Fallback models: if the primary model gets rate-limited (429) or otherwise
# fails after its own retries, we roll over to the next model in this list
# before giving up. First entry = primary model (kept in sync with the
# TEXT_MODEL/VISION_MODEL constants above by main()). Override with
# --fallback-models "model-a,model-b" on the CLI.
TEXT_MODEL_FALLBACKS = [TEXT_MODEL, "gemini-3.1-flash-lite", "gemini-flash-latest"]
VISION_MODEL_FALLBACKS = [VISION_MODEL, "gemini-3.1-flash-lite", "gemini-flash-latest"]

VISION_BATCH_SIZE = 5                  # Images per Vision API call
VISION_BATCH_MAX_TOKENS = 24576         # Output budget for a whole batch response —
                                        # generous on purpose: each image can legitimately
                                        # need a full paragraph + several content_regions +
                                        # a verbatim code/table extraction, and a response
                                        # that gets cut off (MAX_TOKENS) mid-array is invalid
                                        # JSON, which used to silently cost every remaining
                                        # screenshot in that batch its analysis.
FREE_TIER_DELAY = 6.5                  # Seconds between API calls (safe for 10 RPM)

# Screenshots are always SAVED at full quality for the report. Before a copy
# is sent to the Vision API, it's downscaled/recompressed to this size —
# Gemini charges vision tokens by resolution tile, and OCR/description
# quality barely changes below this size, so this cuts vision tokens
# substantially at ~no quality cost in the report itself.
VISION_MAX_DIMENSION = 1280
VISION_JPEG_QUALITY = 85

# Global cache: if the same video shows up in two different playlists (or
# two separate runs with different --output-dir), it's only ever processed
# once. Disable with --no-global-cache.
DEFAULT_GLOBAL_CACHE_DIR = str(Path(os.environ.get('MARROW_HOME', str(Path.home() / '.marrow'))) / 'cache')

# Metadata-only yt-dlp calls (video info/chapters, channel branding, playlist
# listing) never need a playable stream URL, so they don't need YouTube's JS
# challenge solved. Since most machines don't have a JS runtime (deno/node)
# installed, yt-dlp's default client mix wastes several seconds per video
# retrying JS-requiring clients and printing the "No supported JavaScript
# runtime" warning before falling back — for metadata-only calls, pinning
# player_client to the non-JS default set skips that entirely. Real video
# downloads (download_video, below) deliberately do NOT use this — they
# want every available client/format for the best quality, and will use a
# JS runtime automatically if one is installed. To remove the warning
# everywhere AND get full-quality downloads, install deno once:
# https://github.com/yt-dlp/yt-dlp/wiki/EJS
_YDL_METADATA_ARGS = {'extractor_args': {'youtube': {'player_client': ['default']}}}

NUMBER_PATTERN = re.compile(r'\b\d{2,6}(?:\.\d+)?\b')
VISUAL_KEYWORDS = re.compile(
    r'\b(diagram|graph|chart|figure|formula|equation|table|proof|derivation|'
    r'dekho|dikh|yahan|is diagram|is graph|is table|jaise likha|whiteboard|'
    r'board pe|screen pe|as shown|as you can see|iske according|slide|'
    r'code|function|class|import|variable|output|terminal|console|editor|'
    r'IDE|result|demo|flowchart|algorithm|step by step|architecture|'
    r'workflow|pipeline|structure|layout|design|interface|component)\b',
    re.IGNORECASE,
)

# ═══════════════════════════════════════════════════════════════════
# AI PROMPTS
# ═══════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_V4 = """You are an expert video content analyzer. Temperature=0. Be maximally faithful, never creative.

You receive a timestamped transcript. Extract EVERYTHING — miss NOTHING.

0. LANGUAGE — DECIDE THIS FIRST, BEFORE WRITING ANYTHING ELSE:
   Read enough of the transcript to identify EXACTLY what language(s) it is in — a single language (e.g. Hindi, Marathi, Tamil, English), or a code-switched mix (e.g. "Hindi-English mixed / Hinglish", "Marathi-English mixed"). Report this in "detected_language". This is not a formality — every "heading" and every "notes" field you write for EVERY section, from the very first to the very last, MUST be in exactly that language/mix. This is the single most important rule in this prompt:
   - Do NOT translate anything to English. Do NOT normalize a Hindi/Hinglish/Marathi/etc. video into clean English notes — that defeats the entire purpose for a reader who wants notes in the video's own language.
   - Do NOT drift back to English partway through just because the transcript is long or a later portion is harder to follow — the language you commit to in "detected_language" applies to the LAST section exactly as much as the first.
   - If the speaker genuinely code-switches (e.g. explains a concept in Hindi, then says a technical term or sentence in English), preserve that same natural mix in your notes instead of forcing everything into one language — that mix IS the video's actual language.
   - Only write in English if the transcript itself is genuinely, entirely in English.
   - **SCRIPT RULE (read carefully — this is the rule most often gotten wrong):** "Hinglish" describes the SPOKEN code-switching pattern, not a script to write in. When you write Hindi words or sentences, use actual Devanagari script (देवनागरी) — NOT Hindi words spelled out in Roman/Latin letters (that romanized chat-style spelling, e.g. "yeh important hai", is WRONG even though it is commonly called "Hinglish"). English words/sentences stay in Latin script as normal. So a Hindi-English mixed sentence must look like: "यह एक महत्वपूर्ण concept है जो हमें समझना होगा।" — Hindi portions in Devanagari, English portions (technical terms, code, proper nouns the speaker actually said in English) in Latin script. NEVER write an entire section by transliterating Hindi into Latin letters — that is not "preserving the language," it silently defeats rule 0 just as much as translating to English would.

1. VIDEO TYPE: "coding" | "slides" | "lecture" | "tutorial" | "trading" | "whiteboard" | "general"

2. SECTIONS: Split into logical topics covering the ENTIRE transcript. As many as needed.
   - **IF OFFICIAL YOUTUBE CHAPTERS ARE PROVIDED** (see "OFFICIAL CHAPTERS" block below, if present): treat their timestamps as ground-truth section boundaries. Each chapter should map to one section starting at that exact timestamp (you may split a single chapter into more than one section if it covers clearly distinct sub-topics, but never merge two chapters into one section or ignore a chapter boundary). Use the chapter title as the section heading, kept/translated into "detected_language" (never flattened to English) unless the transcript makes a more descriptive heading obvious.
   - Headings you write yourself (no official chapter) must ALSO be in "detected_language" — do not default to English headings over non-English notes.

3. NOTES: Comprehensive, attractive, and highly readable notes per section:
   - **CRITICAL LANGUAGE RULE:** The output MUST be in the EXACT SAME LANGUAGE as "detected_language" above — see rule 0. Do NOT translate everything to English.
   - Transform the spoken words into well-structured, clear, and professional educational notes. DO NOT just provide a raw transcript.
   - Use attractive Markdown formatting: heavily use **bolding** for key terms, `inline code`, blockquotes (>), and bullet points.
   - **CRITICAL FORMATTING RULE:** Whenever there is a mathematical formula, equation, or variable, you MUST format it using standard Markdown LaTeX math blocks (e.g., `$$E=mc^2$$` for block equations, or `\\(x = 5\\)` for inline math). DO NOT use `$` for inline math. DO NOT wrap plain numbers or currency values in math tags.
   - Whenever there is code, you MUST format it using markdown code blocks with the correct language tag.
   - **CRITICAL TABLE RULE:** If the content includes tabular/structured data (rows and columns of comparable items — e.g. fees, dates, scores, comparisons), do NOT write it out as prose sentences or a wall of bold labels in `notes`. Write only a short one-line intro sentence in `notes` (e.g. "Fee structure is as follows:") and put the actual rows/columns ONLY in the separate `table` JSON field. Never duplicate the same tabular data in both places.
   - Explain the concepts clearly so the reader can easily understand the topic without watching the video.
   - PRESERVE every specific detail: numbers, formulas, definitions, steps, examples, names, and URLs.
   - Do NOT add external information that wasn't discussed in the video.

4. SCREENSHOT TIMESTAMPS: For each visual moment, provide exact seconds + type. Be generous:
   - Diagrams, charts, graphs, figures
   - Code on screen (editor, terminal, IDE)
   - Slides being shown
   - Formulas, equations (board/screen)
   - Tables, data displays
   - UI demonstrations
   - Architecture diagrams, flowcharts
   - ANY visual reference the speaker mentions
   Types: "diagram" | "chart" | "table" | "code" | "slide" | "formula" | "whiteboard" | "ui_demo" | "architecture" | "general"

5. CODE BLOCKS: For coding videos — extract ALL code mentioned/dictated. Include language, complete code, and description. For non-coding videos, extract any commands/code mentioned.

6. TABLE DATA: Extract tabular/structured data as {"headers":[...], "rows":[[...]]}

7. FLOWCHARTS: Mermaid syntax for processes, algorithms, workflows. Use simple syntax:
   flowchart TD
     A[Step] --> B{Decision}
     B -->|Yes| C[Action]
     B -->|No| D[Other]

8. OVERVIEW FLOWCHART: One flowchart showing the video's entire content structure/flow.

9. IMPORTANCE TAGGING: For EVERY section, set "importance" to either:
   - "must_know" — core concept, definition, or step the video is actually about
   - "extra" — a tangent, side-note, story, ad/sponsor mention, or "by the way" aside that a student revising for exams could skip
   Default to "must_know" when genuinely unsure — only mark "extra" when it's clearly a digression.

OUTPUT — strict JSON, no markdown fences:
{
  "detected_language": "string — e.g. 'Hindi', 'English', 'Hindi-English mixed (Hinglish)', 'Marathi-English mixed'",
  "video_type": "string",
  "overview_flowchart": "mermaid string or null",
  "sections": [
    {
      "start_seconds": 0,
      "heading": "Section Title",
      "notes": "Complete notes...",
      "importance": "must_know",
      "screenshot_timestamps": [
        {"seconds": 15, "visual_type": "code", "description": "what to capture"}
      ],
      "code_blocks": [
        {"language": "python", "code": "code here", "description": "what it does"}
      ],
      "has_visual": true,
      "table": null,
      "flowchart": null
    }
  ]
}

RULES:
- Valid JSON only — no trailing commas
- Cover ENTIRE transcript start to finish
- For flowcharts: use \\n for newlines in JSON, quote labels with special chars"""


BATCH_VISION_PROMPT = """You are analyzing {count} screenshots from an educational video. For EACH image, provide a thorough analysis. You are also given the Transcript Context for each image.

**CRITICAL LANGUAGE RULE:** Write DESCRIPTION and CAPTION in the EXACT SAME language/script as the Transcript Context (e.g. if the context is in Marathi, write in Marathi; if it's Hinglish, write in that same mix). Do NOT translate to English. Use each language's own native script (e.g. Devanagari for Hindi) — never Roman-transliterated text, even for a Hindi-English code-mix; only the genuinely English words stay in Latin script.

**CRITICAL CONTENT RULE:** The caption's job is to tell the reader WHAT CONCEPT/TOPIC is being taught at this moment, not to describe the physical scene. Never write generic scene descriptions like "Presenter pointing at whiteboard" or "Teacher explaining diagram" — these tell the reader nothing they can't already see. Instead:
   - If the image is a whiteboard/handwritten board, slide, or on-screen text that is small, blurry, cluttered, or hard to read, rely primarily on the Transcript Context to name the specific concept, topic, or data being discussed right then (e.g. "GK, Maths व Marathi साठी दैनिक अभ्यास वेळापत्रक" instead of "Daily study slots breakdown").
   - Only describe literal visual composition (e.g. "person pointing at board") if the Transcript Context gives no usable topic information at all.

For each screenshot, identify:
1. DESCRIPTION: What CONTENT/CONCEPT is shown or being discussed — grounded in the Transcript Context, not just pixels
2. CAPTION: A short, content-accurate title/caption for the image (max 10 words), in the same language as the Transcript Context, naming the actual topic/concept
3. VISUAL TYPE: diagram | chart | table | code | slide | formula | whiteboard | ui_demo | architecture | talking_head | blank | general
4. IS USEFUL: false ONLY for pure talking-head (just face) or blank/transition. Everything else = true
5. CONTENT REGIONS: ALL meaningful visual elements worth cropping. This includes:
   - Diagrams, charts, graphs (crop each one)
   - Code in editors/terminals (crop the code area)
   - Formulas/equations (crop each formula)
   - Tables/data displays (crop the table)
   - Important text blocks (crop key text)
   - Architecture diagrams (crop the diagram)
   - UI elements being demonstrated (crop the UI)
   For each: type, bbox [x%, y%, w%, h%] (0-100 scale), description
5. EXTRACTED CODE: If code is visible, extract it EXACTLY with language tag
6. TABLE DATA: If a table is visible, extract headers and rows

**CRITICAL — ONE ENTRY PER IMAGE:** You MUST return EXACTLY {count} objects — one for every image, never fewer. Never skip an image and never merge two images into one entry, even if two images look similar or one seems low-value; if an image is genuinely blank/redundant, still return an entry for it with "is_useful": false. "image_index" MUST match the 0-based "Image N" label shown above each image (the first image is index 0), and each index from 0 to {count_minus_1} must appear EXACTLY once — this index, not array position, is what gets used to attach your analysis to the correct image, so a wrong index will attach your description to the wrong screenshot.

Output strict JSON array (one object per image, same order as images):
[
  {{
    "image_index": 0,
    "description": "string",
    "caption": "string",
    "visual_type": "string",
    "is_useful": true,
    "content_regions": [
      {{"type": "diagram", "bbox": [10, 5, 60, 50], "description": "flowchart showing process"}}
    ],
    "extracted_code": {{"language": "python", "code": "..."}} or null,
    "table_data": {{"headers": [...], "rows": [...]}} or null
  }}
]

IMPORTANT: Include ALL visually useful elements in content_regions — not just code/tables.
Diagrams, charts, formulas, architecture drawings, UI elements — crop EVERYTHING that helps understanding."""


SINGLE_VISION_PROMPT = """Analyze this screenshot from an educational video. Be thorough.

**LANGUAGE RULE:** Write the description in the same language/script as the video (infer from any visible on-screen text). Do not translate to English. Use native script (e.g. Devanagari for Hindi), never Roman-transliterated Hindi.
**CONTENT RULE:** Describe the CONCEPT being taught, not the physical scene. If the frame is a cluttered or handwritten whiteboard that's hard to read, say what topic is being covered rather than "person pointing at board".

1. DESCRIPTION: What CONTENT/CONCEPT is shown (be specific about the topic, not just objects in frame)
2. VISUAL TYPE: diagram | chart | table | code | slide | formula | whiteboard | ui_demo | architecture | talking_head | blank | general
3. IS USEFUL: false only for pure talking-head or blank screen
4. CONTENT REGIONS: ALL meaningful visual elements worth cropping:
   - Diagrams, charts, graphs, architecture diagrams
   - Code in editors/terminals
   - Formulas/equations
   - Tables/data
   - Important text blocks, UI elements
   For each: type, bbox [x%, y%, w%, h%] (0-100), description
5. EXTRACTED CODE: If code visible, extract exactly with language
6. TABLE DATA: If table visible, extract headers + rows

Output strict JSON:
{
  "description": "string",
  "visual_type": "string",
  "is_useful": true,
  "content_regions": [
    {"type": "diagram", "bbox": [10, 5, 60, 50], "description": "process flowchart"}
  ],
  "extracted_code": null,
  "table_data": null
}

IMPORTANT: Identify ALL visual elements, not just code/tables. Crop diagrams, charts, formulas, UI — everything visually useful."""


KEY_TAKEAWAYS_PROMPT = """You are given the complete section-by-section notes for an educational video (below, as JSON). Write a short "Key Takeaways" summary for the TOP of the report — the 5-10 most important points a student should remember, in priority order.

**LANGUAGE RULE:** Write in the EXACT SAME language/script as the notes themselves. Do not translate. If the notes are a Hindi-English mix, use Devanagari for the Hindi parts and Latin script for the English parts — never Roman-transliterated Hindi.
**CONTENT RULE:** Each point should be a single, information-dense sentence — a fact, definition, formula, or conclusion — not a vague topic label. Skip tangents/asides; focus on "must_know" material.

Output strict JSON, no markdown fences:
{"takeaways": ["point 1", "point 2", "..."]}"""


QUIZ_PROMPT = """You are given the complete section-by-section notes for an educational video (below, as JSON). Generate revision flashcards covering the material, so a student can self-test without rewatching the video.

**LANGUAGE RULE:** Write in the EXACT SAME language/script as the notes themselves. Do not translate. If the notes are a Hindi-English mix, use Devanagari for the Hindi parts and Latin script for the English parts — never Roman-transliterated Hindi.
**COUNT:** Generate between 5 and 15 cards depending on how much material there is — enough to cover every "must_know" section at least once, but don't pad with trivial or repetitive questions.
**CONTENT RULE:** Each question should test a specific fact, definition, formula, or step from the notes — not a vague "what was discussed" question. Prefer short, unambiguous answers over essay-length ones.

Output strict JSON, no markdown fences:
{"cards": [{"question": "string", "answer": "string"}]}"""


QA_PROMPT = """You answer questions using ONLY the excerpts provided below — each excerpt is labeled with the video title and a timestamp. These excerpts come from a student's own saved video notes.

**GROUNDING RULE (most important):** Answer strictly from the given excerpts. Do NOT use outside knowledge to fill gaps, and do NOT guess. If the excerpts don't actually answer the question, say so plainly instead of guessing.
**LANGUAGE RULE:** Answer in the same language the person asked in.
**CITATION RULE:** For every claim in your answer, note which excerpt(s) it came from using their given [N] label.

Output strict JSON, no markdown fences:
{"found": true/false, "answer": "string (empty if found is false)", "used_excerpts": [1, 3]}
If nothing in the excerpts is relevant, set "found": false and leave "answer" empty — do not apologize at length, do not speculate."""


# ═══════════════════════════════════════════════════════════════════
# RATE LIMITER & API BUDGET
# ═══════════════════════════════════════════════════════════════════

class RateLimiter:
    """Enforces delays between API calls to respect free tier RPM limits.
    Thread-safe: when multiple videos are processed in parallel (--parallel),
    all worker threads share ONE rate limiter, because the RPM limit is
    per API key, not per thread."""

    def __init__(self, delay=FREE_TIER_DELAY):
        self.delay = delay
        self._last_call = 0
        self._lock = threading.Lock()

    def wait(self):
        with self._lock:
            elapsed = time.time() - self._last_call
            if elapsed < self.delay:
                wait_time = self.delay - elapsed
                if ui is not None:
                    ui.live_wait(wait_time, "Respecting rate limit")
                else:
                    print(f"    ⏳ Rate limit: waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
            self._last_call = time.time()


class APIBudget:
    """Tracks and limits Gemini API calls across the entire run.
    Thread-safe for use across parallel video-processing workers."""

    def __init__(self, max_calls=None):
        self.max_calls = max_calls
        self.used = 0
        self._lock = threading.Lock()

    def available(self):
        with self._lock:
            return self.max_calls is None or self.used < self.max_calls

    def use(self, n=1):
        with self._lock:
            self.used += n

    def remaining(self):
        with self._lock:
            if self.max_calls is None:
                return float('inf')
            return max(0, self.max_calls - self.used)

    def __str__(self):
        with self._lock:
            if self.max_calls is None:
                return f"API calls: {self.used} (no limit)"
            return f"API calls: {self.used}/{self.max_calls}"


_rate_limiter = RateLimiter()


# ═══════════════════════════════════════════════════════════════════
# SAFE FILE I/O — atomic writes with retry
# ═══════════════════════════════════════════════════════════════════
# Root cause of the "OSError: [Errno 9] Bad file descriptor" crash: this
# script was being run under the Microsoft Store / WindowsApps build of
# Python (visible in the traceback path: ...WindowsApps\PythonSoftware
# Foundation.Python...), whose sandboxed/virtualized file-system layer can
# occasionally hand back a file object with an already-invalid descriptor
# on a completely ordinary open()+write() — especially right after a
# subprocess call (yt-dlp). It's rare, transient, and outside this
# script's control — switching to the regular python.org installer (not
# the Store one) removes the sandbox layer entirely and is the real fix.
# Until/unless that happens, every JSON/text write below goes through
# these retry-safe atomic helpers instead of a bare open(): write to a
# temp file first (so a failed write can never corrupt the previous good
# cache) and retry a couple of times. Just as important as the Windows
# fix: sections.json used to be written with a bare open() and NO
# try/except at all (every other cache write in this file already had
# one) — one failed write there aborted the rest of process_video
# (screenshots, PDF, everything) even though the AI analysis had already
# been fetched and paid for. These helpers never raise, so that can't
# happen again.

def _safe_write_json(path, data, retries=3, **json_kwargs):
    """Atomic, retry-safe JSON write. Returns True/False, never raises —
    callers should treat a failed cache write as non-fatal and carry on
    with the data they already have in memory."""
    path = Path(path)
    tmp_path = path.with_name(path.name + f'.tmp{os.getpid()}')
    json_kwargs.setdefault('ensure_ascii', False)
    last_err = None
    for attempt in range(retries):
        try:
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, **json_kwargs)
                f.flush()
            os.replace(tmp_path, path)
            return True
        except OSError as e:
            last_err = e
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            if attempt < retries - 1:
                time.sleep(0.4 * (attempt + 1))
    print(f"    [!] Could not save {path.name} (non-fatal, continuing): {last_err}")
    return False


def _safe_write_text(path, text, retries=3):
    """Same atomic-write-with-retry pattern as _safe_write_json, for plain
    text/HTML/Markdown output files. Returns True/False, never raises."""
    path = Path(path)
    tmp_path = path.with_name(path.name + f'.tmp{os.getpid()}')
    last_err = None
    for attempt in range(retries):
        try:
            with open(tmp_path, 'w', encoding='utf-8') as f:
                f.write(text)
                f.flush()
            os.replace(tmp_path, path)
            return True
        except OSError as e:
            last_err = e
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            if attempt < retries - 1:
                time.sleep(0.4 * (attempt + 1))
    print(f"    [!] Could not save {path.name} (non-fatal, continuing): {last_err}")
    return False


# ═══════════════════════════════════════════════════════════════════
# TRANSCRIPT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def get_transcript(video_id, language=None):
    """Fetch transcript with timestamps. Returns list of dicts or None.

    Language selection deliberately does NOT hardcode a 'hi'/'en' preference
    for every video — a fixed bias like that silently pulls in whichever of
    those two happens to exist even when the video is actually in Marathi,
    Tamil, pure English, or anything else, and the person never finds out
    which language/track was actually used (which then gets blamed on Pass 1
    "translating wrong" when the transcript itself was already the issue).

    Preference order:
      1. `language`, if the caller explicitly asked for one.
      2. The video's own manually-created transcript (any language) — most
         likely to faithfully reflect the real spoken mix, since creator-
         uploaded auto-translated tracks are the ones actually likely to be
         a plain English translation.
      3. Any auto-generated transcript (any language) as a fallback.

    Whichever track is used, its language code and generated/manual status
    is printed so a wrong pick is visible and fixable with --language."""
    ytt_api = YouTubeTranscriptApi()

    if language:
        try:
            fetched = ytt_api.fetch(video_id, languages=[language])
            print(f"    Transcript: using '{language}' (as requested).")
            return fetched.to_raw_data()
        except NoTranscriptFound:
            print(f"    [!] Requested --language '{language}' not available "
                  f"for this video — falling back to whatever exists.")
        except TranscriptsDisabled:
            print(f"  ✗ Transcripts disabled for {video_id}")
            return None
        except VideoUnavailable:
            print(f"  ✗ Video unavailable: {video_id}")
            return None
        except Exception as e:
            print(f"  ✗ Transcript error for {video_id}: {e}")
            return None

    try:
        transcript_list = ytt_api.list(video_id)
    except TranscriptsDisabled:
        print(f"  ✗ Transcripts disabled for {video_id}")
        return None
    except VideoUnavailable:
        print(f"  ✗ Video unavailable: {video_id}")
        return None
    except Exception as e:
        print(f"  ✗ No transcript at all for {video_id}: {e}")
        return None

    manual, generated = [], []
    for t in transcript_list:
        (generated if t.is_generated else manual).append(t)

    for t in (manual + generated):
        try:
            fetched = t.fetch()
            kind = 'auto-generated' if t.is_generated else 'manual'
            print(f"    Transcript: using '{t.language_code}' ({kind}) "
                  f"— pass --language to override if this is wrong.")
            return fetched.to_raw_data()
        except Exception:
            continue  # try the next available transcript track

    print(f"  ✗ No usable transcript track for {video_id}")
    return None


_WHISPER_BACKEND = None  # resolved lazily: 'faster_whisper' | 'openai_whisper' | 'none'


def _resolve_whisper_backend():
    global _WHISPER_BACKEND
    if _WHISPER_BACKEND is not None:
        return _WHISPER_BACKEND
    try:
        import faster_whisper  # noqa: F401
        _WHISPER_BACKEND = 'faster_whisper'
    except ImportError:
        try:
            import whisper  # noqa: F401
            _WHISPER_BACKEND = 'openai_whisper'
        except ImportError:
            _WHISPER_BACKEND = 'none'
    return _WHISPER_BACKEND


def transcribe_with_whisper(video_id, work_dir, model_size='base'):
    """Fallback for videos with NO YouTube captions at all (auto or manual):
    download audio-only (small/fast, no video) and transcribe locally with
    Whisper. Returns the same [{'start': float, 'text': str}, ...] shape as
    get_transcript(), or None if no Whisper backend is installed or
    transcription fails. Runs entirely on the local machine — only the
    audio download touches the network."""
    backend = _resolve_whisper_backend()
    if backend == 'none':
        print("    [!] No captions AND no local Whisper installed. Install "
              "one to transcribe audio locally: pip install faster-whisper")
        return None

    audio_path = str(Path(work_dir) / f"{video_id}_audio.m4a")
    try:
        print(f"    No captions found — downloading audio for local "
              f"Whisper transcription...")
        ydl_opts = {'format': 'bestaudio/best', 'outtmpl': audio_path, 'quiet': True}
        with _download_progress("Downloading audio") as dp:
            ydl_opts['progress_hooks'] = [dp.hook]
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
    except Exception as e:
        print(f"    [X] Audio download failed: {e}")
        return None

    if not os.path.exists(audio_path):
        candidates = list(Path(work_dir).glob(f"{video_id}_audio.*"))
        audio_path = str(candidates[0]) if candidates else None
    if not audio_path or not os.path.exists(audio_path):
        print("    [X] Downloaded audio file not found.")
        return None

    try:
        print(f"    Transcribing locally with Whisper ({backend}, "
              f"'{model_size}' model) — this can take a while on CPU...")
        segments_out = []
        with _spinner(f"Transcribing with Whisper ({model_size})…"):
            if backend == 'faster_whisper':
                from faster_whisper import WhisperModel
                model = WhisperModel(model_size, device='cpu', compute_type='int8')
                segments, _ = model.transcribe(audio_path)
                for seg in segments:
                    segments_out.append({'start': float(seg.start), 'text': seg.text.strip()})
            else:
                import whisper
                model = whisper.load_model(model_size)
                result = model.transcribe(audio_path)
                for seg in result.get('segments', []):
                    segments_out.append({'start': float(seg['start']), 'text': seg['text'].strip()})
        if segments_out:
            print(f"    [OK] Whisper produced {len(segments_out)} segments.")
        return segments_out or None
    except Exception as e:
        print(f"    [X] Whisper transcription failed: {e}")
        return None
    finally:
        try:
            os.remove(audio_path)
        except Exception as e:
            _log.debug("Whisper temp audio cleanup failed for %s: %s", audio_path, e)


def fetch_video_metadata(video_id):
    """Lightweight, download-free metadata fetch used to build a richer HTML
    header (thumbnail, duration, views, upload date, channel avatar/banner)
    and to pull the creator's own chapter markers if they exist.
    Best-effort: any failure here should never stop the pipeline."""
    meta = {}
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        with yt_dlp.YoutubeDL({'quiet': True, 'skip_download': True,
                               **_YDL_METADATA_ARGS}) as ydl:
            info = ydl.extract_info(url, download=False)
        meta['thumbnail'] = info.get('thumbnail')
        meta['duration'] = info.get('duration')
        meta['view_count'] = info.get('view_count')
        meta['upload_date'] = info.get('upload_date')
        meta['channel_url'] = info.get('channel_url') or info.get('uploader_url')
        # Creator-provided chapters (ground truth for section boundaries, if any)
        raw_chapters = info.get('chapters') or []
        chapters = []
        for ch in raw_chapters:
            try:
                start = int(ch.get('start_time', 0))
                chapter_title = str(ch.get('title', '')).strip()
                if chapter_title:
                    chapters.append({'start_seconds': start, 'title': chapter_title})
            except (TypeError, ValueError):
                continue
        meta['chapters'] = chapters
    except Exception as e:
        print(f"    [!] Metadata fetch failed (non-fatal): {e}")
        return meta

    # Channel avatar/banner: best-effort, depends on yt-dlp's channel-page
    # extraction and may not be available on every version/channel.
    channel_url = meta.get('channel_url')
    if channel_url:
        try:
            with yt_dlp.YoutubeDL({'quiet': True, 'extract_flat': True,
                                   'skip_download': True,
                                   **_YDL_METADATA_ARGS}) as ydl:
                ch_info = ydl.extract_info(channel_url, download=False)
            thumbs = ch_info.get('thumbnails') or []
            avatar = next((t.get('url') for t in thumbs
                          if 'avatar' in (t.get('id') or '')), None)
            banner = next((t.get('url') for t in thumbs
                          if 'banner' in (t.get('id') or '')), None)
            meta['channel_avatar'] = avatar or (thumbs[-1].get('url') if thumbs else None)
            meta['channel_banner'] = banner
        except Exception:
            pass  # channel branding is a nice-to-have, not required
    return meta


def get_video_list(url, max_videos=10):
    """Extract video list from URL (video/playlist/channel).
    Returns (videos, collection_title) — collection_title is the playlist's
    or channel's own title when the URL points to a collection, else None."""
    ydl_opts = {'quiet': True, 'extract_flat': True, 'skip_download': True,
               **_YDL_METADATA_ARGS}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    videos = []
    collection_title = None
    if info and info.get('entries') is not None:
        collection_title = info.get('title')
        for entry in info['entries']:
            if entry is None:
                continue
            vid = entry.get('id')
            title = entry.get('title', vid)
            channel = entry.get('channel') or entry.get('uploader') or ''
            if vid:
                videos.append((vid, title, channel))
            if len(videos) >= max_videos:
                break
    elif info:
        videos.append((
            info.get('id'), info.get('title', info.get('id')),
            info.get('channel') or info.get('uploader') or '',
        ))
    return videos, collection_title


def build_timestamped_text(transcript, max_chars=60000):
    """Build timestamped string for AI prompt."""
    out = []
    total = 0
    truncated = False
    for entry in transcript:
        m, s = divmod(int(entry['start']), 60)
        chunk = f"[{m}:{s:02d}] {entry['text']} "
        if total + len(chunk) > max_chars:
            truncated = True
            break
        out.append(chunk)
        total += len(chunk)
    return ''.join(out), truncated


def _format_chapters_block(chapters):
    """Render creator-provided YouTube chapters as a ground-truth block that
    gets prepended to the transcript sent to Pass 1."""
    if not chapters:
        return ''
    lines = ["=== OFFICIAL YOUTUBE CHAPTERS (ground truth for section boundaries) ==="]
    for ch in chapters:
        m, s = divmod(int(ch['start_seconds']), 60)
        lines.append(f"[{m}:{s:02d}] {ch['title']}")
    lines.append("=== END OFFICIAL CHAPTERS ===\n")
    return '\n'.join(lines)


# Indic + Arabic/Urdu script ranges — used only for a rough same-language
# sanity check between the transcript and the generated notes, never to
# judge translation quality precisely.
_INDIC_SCRIPT_PATTERN = re.compile(
    r'[\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF\u0B00-\u0B7F'
    r'\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F\u0600-\u06FF]'
)
_LATIN_LETTER_PATTERN = re.compile(r'[A-Za-z]')


def _script_letter_counts(text):
    """Returns (non_latin_script_letters, latin_letters) in `text`."""
    if not text:
        return (0, 0)
    return (len(_INDIC_SCRIPT_PATTERN.findall(text)),
            len(_LATIN_LETTER_PATTERN.findall(text)))


def _warn_if_language_drifted(transcript_text, sections):
    """CRITICAL LANGUAGE RULE sanity check (heuristic-only, non-blocking).

    If the source transcript clearly contains a meaningful amount of a
    non-Latin script (Hindi/Marathi/Bengali/Tamil/Urdu/etc.) but the notes
    Pass 1 generated are almost entirely English/Latin script, the model
    most likely ignored the "write in the video's own language" rule and
    silently translated everything to English instead. This can't be
    auto-fixed without another API call, so it just prints a clear warning
    the person can act on (e.g. re-run, or open an issue with --fresh)."""
    t_non_latin, t_latin = _script_letter_counts(transcript_text)
    t_total = t_non_latin + t_latin
    if t_total < 200 or (t_non_latin / t_total) < 0.15:
        return  # transcript itself isn't meaningfully non-Latin-script

    notes_text = ' '.join(s.get('notes', '') for s in sections)
    n_non_latin, n_latin = _script_letter_counts(notes_text)
    n_total = n_non_latin + n_latin
    if n_total == 0 or (n_non_latin / n_total) >= 0.02:
        return  # notes still carry a reasonable share of the original script

    print("    [!] LANGUAGE CHECK: the transcript has a significant amount "
          "of non-English script, but the generated notes are almost "
          "entirely in English. The AI likely translated instead of "
          "preserving the video's own language/mix — if this looks wrong, "
          "re-run with --fresh (a repeat run can come out differently since "
          "this is a known model-following issue, not a caching issue).")


# ═══════════════════════════════════════════════════════════════════
# PASS 1: GEMINI TRANSCRIPT ANALYSIS (1 API call per video)
# ═══════════════════════════════════════════════════════════════════

def _call_gemini(api_key, model, contents, system_prompt=None,
                 max_tokens=8192, budget=None):
    """Generic Gemini API call with rate limiting and budget tracking.
    Returns parsed JSON response or raises RuntimeError."""
    if budget and not budget.available():
        raise RuntimeError("API budget exhausted")

    _rate_limiter.wait()

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": max_tokens,
            "response_mime_type": "application/json",
        },
    }
    if system_prompt:
        body["system_instruction"] = {"parts": [{"text": system_prompt}]}

    max_retries = 4
    for attempt in range(max_retries):
        try:
            with _spinner(f"Calling {model}…"):
                resp = requests.post(url, headers={"x-goog-api-key": api_key},
                                     json=body, timeout=180)

            if resp.status_code in (429, 503):
                wait_time = 5 * (2 ** attempt)  # 5s, 10s, 20s, 40s
                print(f"    [!] API busy (Error {resp.status_code}). "
                      f"Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                _live_wait(wait_time, "Retrying shortly")
                continue
                
            break # Success or different error, exit retry loop
            
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Network error: {e}")
            _live_wait(5, "Network hiccup — retrying")

    if budget:
        budget.use()

    if resp.status_code in (401, 403):
        raise RuntimeError("API key invalid/unauthorized.")
    if not resp.ok:
        raise RuntimeError(f"API error {resp.status_code}: {resp.text[:400]}")

    data = resp.json()
    candidate = (data.get('candidates') or [{}])[0]
    finish_reason = candidate.get('finishReason')
    parts = candidate.get('content', {}).get('parts', [])
    text = parts[0].get('text') if parts else None

    if not text:
        if finish_reason == 'MAX_TOKENS':
            raise RuntimeError("Output truncated (MAX_TOKENS). Try --max-chars lower.")
        raise RuntimeError(f"No text in response (finishReason={finish_reason}).")

    cleaned = text.strip()
    if cleaned.startswith('```'):
        cleaned = re.sub(r'^```(?:json)?', '', cleaned).strip()
        cleaned = re.sub(r'```$', '', cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        # The model occasionally emits a stray literal backslash inside a
        # string (common with Hinglish/mixed-script transcripts, currency
        # symbols, or math-like notation) — that's an invalid JSON escape
        # even though everything else about the response is fine. Try one
        # cheap local repair (escape any backslash that isn't already a
        # valid JSON escape) before giving up on this model entirely.
        repaired = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', cleaned)
        if repaired != cleaned:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass
        raise RuntimeError(f"Invalid JSON response: {e}")


def _call_gemini_with_fallback(api_key, models, contents, system_prompt=None,
                               max_tokens=8192, budget=None):
    """Try each model in `models` in order (first = primary). If a model
    comes back rate-limited/unavailable even after its own internal retries,
    roll over to the next model instead of failing the whole video —
    reliability upgrade so a single model's free-tier quota doesn't stall
    the run.

    If MARROW's multi-provider config has at least one key connected
    (mrp.rotation_enabled()), this delegates entirely to
    marrow_providers.call_ai_rotating() instead — `api_key`/`models` are
    then ignored, since the rotation manager decides which provider+model
    to use from everything the person connected via /keys. Without any
    MARROW keys configured, this falls back to the original bare-Gemini
    behavior (useful for scripted use with just GEMINI_API_KEY set)."""
    if mrp is not None and mrp.rotation_enabled():
        def _announce_switch(provider_display, model_id):
            print(f"    ↻ switched provider — now using {provider_display} · {model_id}")
        return mrp.call_ai_rotating(
            contents, system_prompt=system_prompt, max_tokens=max_tokens,
            budget=budget, json_mode=True, on_switch=_announce_switch,
        )

    models = [m for m in dict.fromkeys(models) if m]  # dedupe, keep order
    if not models:
        raise RuntimeError("No model configured")

    last_err = None
    for i, model in enumerate(models):
        try:
            return _call_gemini(api_key, model, contents,
                                system_prompt=system_prompt,
                                max_tokens=max_tokens, budget=budget)
        except RuntimeError as e:
            msg = str(e)
            last_err = e
            # These won't be fixed by switching models — stop immediately.
            if 'budget exhausted' in msg or 'unauthorized' in msg:
                raise
            if i < len(models) - 1:
                print(f"    [!] {model} unavailable ({msg[:120]}) "
                      f"— falling back to {models[i + 1]}...")
                continue
    raise last_err or RuntimeError("All fallback models failed")


def analyze_with_gemini_v4(timestamped_text, api_key, budget, chapters=None):
    """Pass 1: Comprehensive transcript analysis. 1 API call (+ automatic
    fallback to a backup model if the primary one is rate-limited)."""
    print("    Calling Gemini (Pass 1: transcript analysis)...")

    prompt_text = timestamped_text
    if chapters:
        prompt_text = _format_chapters_block(chapters) + "\n" + timestamped_text
        print(f"    Using {len(chapters)} official YouTube chapter(s) as ground truth.")

    contents = [{"role": "user", "parts": [{"text": prompt_text}]}]
    parsed = _call_gemini_with_fallback(api_key, TEXT_MODEL_FALLBACKS, contents,
                                        system_prompt=SYSTEM_PROMPT_V4,
                                        max_tokens=65536, budget=budget)

    # Validate and clean
    result = {
        'video_type': str(parsed.get('video_type', 'general')).lower(),
        'detected_language': str(parsed.get('detected_language', '')).strip(),
        'overview_flowchart': parsed.get('overview_flowchart'),
        'sections': [],
    }
    if result['detected_language']:
        print(f"    Detected language: {result['detected_language']}")

    sections = parsed.get('sections')
    if not isinstance(sections, list) or not sections:
        raise RuntimeError("No sections in response.")

    for s in sections:
        if not isinstance(s, dict) or not s.get('notes'):
            continue

        clean_ss = []
        for ss in (s.get('screenshot_timestamps') or []):
            if isinstance(ss, dict) and 'seconds' in ss:
                clean_ss.append({
                    'seconds': max(0, int(ss['seconds'])),
                    'visual_type': str(ss.get('visual_type', 'general')),
                    'description': str(ss.get('description', '')),
                })

        clean_cb = []
        for cb in (s.get('code_blocks') or []):
            if isinstance(cb, dict) and cb.get('code'):
                clean_cb.append({
                    'language': str(cb.get('language', '')),
                    'code': str(cb['code']),
                    'description': str(cb.get('description', '')),
                })

        importance = str(s.get('importance', 'must_know')).strip().lower()
        if importance not in ('must_know', 'extra'):
            importance = 'must_know'

        result['sections'].append({
            'start_seconds': max(0, int(s.get('start_seconds', 0))),
            'heading': str(s.get('heading', ''))[:200],
            'notes': str(s['notes']).strip(),
            'importance': importance,
            'screenshot_timestamps': clean_ss,
            'code_blocks': clean_cb,
            'has_visual': bool(s.get('has_visual', False)),
            'table': s.get('table') if isinstance(s.get('table'), dict) else None,
            'flowchart': s.get('flowchart') if isinstance(s.get('flowchart'), str) else None,
        })

    if not result['sections']:
        raise RuntimeError("All sections empty after cleanup.")
    return result


def analyze_with_regex(transcript):
    """--skip-ai fallback: zero API calls, keyword heuristics."""
    if not transcript:
        return {'video_type': 'general', 'overview_flowchart': None, 'sections': []}

    window = 30
    duration = transcript[-1]['start'] if transcript else 0
    sections = []
    t = 0
    while t < duration + window:
        bucket = [e for e in transcript if t <= e['start'] < t + window]
        if bucket:
            text = ' '.join(e['text'] for e in bucket)
            has_visual = bool(VISUAL_KEYWORDS.search(text))
            ss_ts = []
            if has_visual:
                ss_ts.append({
                    'seconds': int(bucket[0]['start']),
                    'visual_type': 'general',
                    'description': 'keyword-detected visual',
                })
            sections.append({
                'start_seconds': int(bucket[0]['start']),
                'heading': '',
                'notes': text.strip(),
                'importance': 'must_know',
                'screenshot_timestamps': ss_ts,
                'code_blocks': [],
                'has_visual': has_visual,
                'table': None,
                'flowchart': None,
            })
        t += window

    return {'video_type': 'general', 'overview_flowchart': None, 'sections': sections}


# ═══════════════════════════════════════════════════════════════════
# OPTIONAL EXTRA PASSES — summary & quiz (1 small API call each, off by
# default; enabled with --summary / --quiz)
# ═══════════════════════════════════════════════════════════════════

def _notes_json_for_prompt(analysis, max_chars=40000):
    """Compact {heading, notes, importance} list used as input for the
    summary/quiz passes — cheaper than resending the whole transcript."""
    compact = [{
        'heading': s.get('heading', ''),
        'importance': s.get('importance', 'must_know'),
        'notes': s.get('notes', ''),
    } for s in analysis.get('sections', [])]
    text = json.dumps(compact, ensure_ascii=False)
    if len(text) > max_chars:
        # Trim notes proportionally rather than dropping whole sections
        text = json.dumps(compact, ensure_ascii=False)[:max_chars]
    return text


def generate_key_takeaways(analysis, api_key, budget):
    """Optional 1 extra API call: 5-10 point 'Key Takeaways' summary for the
    top of the report. Returns list of strings, or [] on failure."""
    print("    Generating key takeaways summary (1 extra API call)...")
    notes_text = _notes_json_for_prompt(analysis)
    contents = [{"role": "user", "parts": [{"text": notes_text}]}]
    try:
        parsed = _call_gemini_with_fallback(api_key, TEXT_MODEL_FALLBACKS, contents,
                                            system_prompt=KEY_TAKEAWAYS_PROMPT,
                                            max_tokens=2048, budget=budget)
        pts = parsed.get('takeaways') if isinstance(parsed, dict) else None
        if isinstance(pts, list):
            return [str(p).strip() for p in pts if str(p).strip()]
    except RuntimeError as e:
        print(f"    [!] Key takeaways generation failed (non-fatal): {e}")
    return []


def generate_quiz(analysis, api_key, budget):
    """Optional 1 extra API call: revision Q&A flashcards.
    Returns list of {question, answer} dicts, or [] on failure."""
    print("    Generating revision quiz/flashcards (1 extra API call)...")
    notes_text = _notes_json_for_prompt(analysis)
    contents = [{"role": "user", "parts": [{"text": notes_text}]}]
    try:
        parsed = _call_gemini_with_fallback(api_key, TEXT_MODEL_FALLBACKS, contents,
                                            system_prompt=QUIZ_PROMPT,
                                            max_tokens=4096, budget=budget)
        cards = parsed.get('cards') if isinstance(parsed, dict) else None
        clean = []
        if isinstance(cards, list):
            for c in cards:
                if isinstance(c, dict) and c.get('question') and c.get('answer'):
                    clean.append({'question': str(c['question']).strip(),
                                  'answer': str(c['answer']).strip()})
        return clean
    except RuntimeError as e:
        print(f"    [!] Quiz generation failed (non-fatal): {e}")
    return []


# ═══════════════════════════════════════════════════════════════════
# LOCAL OCR (FREE — no API calls)
# ═══════════════════════════════════════════════════════════════════

def local_ocr_extract(image_path):
    """Extract text from screenshot using Tesseract OCR. Completely free."""
    if not HAS_TESSERACT:
        return None
    try:
        img = PILImage.open(image_path)
        text = pytesseract.image_to_string(img)
        if text and len(text.strip()) > 20:
            return text.strip()
    except Exception as e:
        _log.debug("Tesseract OCR failed for %s: %s", image_path, e)
    return None


def detect_code_in_text(text):
    """Heuristic: does this OCR text look like code?"""
    if not text:
        return False, ''
    indicators = [
        'def ', 'class ', 'import ', 'from ', 'return ', 'if __name__',
        'function ', 'const ', 'let ', 'var ', '=> ', 'async ',
        '#include', 'public ', 'private ', 'void ', 'static ',
        'print(', 'console.log', 'System.out', 'fmt.Println',
        'package ', 'using ', 'namespace ',
    ]
    score = sum(1 for i in indicators if i in text)
    if re.search(r'[{}\[\]();=]', text):
        score += 1
    if re.search(r'^\s{2,}\S', text, re.MULTILINE):
        score += 1

    # Guess language
    lang = ''
    if 'def ' in text or 'import ' in text or 'print(' in text:
        lang = 'python'
    elif 'function ' in text or 'const ' in text or 'console.log' in text:
        lang = 'javascript'
    elif '#include' in text or 'int main' in text:
        lang = 'c'
    elif 'public class' in text or 'System.out' in text:
        lang = 'java'

    return score >= 3, lang


# ═══════════════════════════════════════════════════════════════════
# PASS 2: GEMINI VISION (BATCHED — saves API calls)
# ═══════════════════════════════════════════════════════════════════

def _encode_image_for_vision(image_path):
    """Read an image off disk and return (mime_type, base64_data) ready for
    the Vision API. The ORIGINAL file on disk (used in the report) is never
    touched — this only shrinks the in-memory COPY that gets sent to Gemini.
    Downscaling to VISION_MAX_DIMENSION and re-encoding as JPEG cuts vision
    tokens substantially (Gemini bills by resolution tile) with no visible
    quality loss in the report, since OCR/description doesn't need full-res.
    Falls back to sending the raw file untouched if Pillow can't process it."""
    try:
        img = PILImage.open(image_path)
        img = img.convert('RGB')
        w, h = img.size
        longest = max(w, h)
        if longest > VISION_MAX_DIMENSION:
            scale = VISION_MAX_DIMENSION / longest
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                             PILImage.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=VISION_JPEG_QUALITY, optimize=True)
        return 'image/jpeg', base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception:
        # Fall back to sending the original file as-is
        with open(image_path, 'rb') as f:
            data = base64.b64encode(f.read()).decode('utf-8')
        ext = os.path.splitext(image_path)[1].lower()
        mime = {'.png': 'image/png', '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg'}.get(ext, 'image/png')
        return mime, data


def analyze_screenshots_batch(image_items, api_key, budget, cache_path=None):
    """Analyze multiple screenshots in ONE API call (batched).
    image_items: list of (timestamp_sec, image_path, context)
    Returns dict: {timestamp: vision_result_dict}

    Batching VISION_BATCH_SIZE images per call cuts API usage significantly
    vs. one call per screenshot.

    Every image sent in is guaranteed to end up with an entry in the
    returned dict (or be visibly reported as failed) — see the "safety
    net" below. The model is explicitly instructed to return exactly one
    result per image, but it doesn't always comply (free/weaker rotation
    models especially): it can quietly omit a frame it judges redundant
    instead of returning is_useful=false for it as instructed. Because
    the omitted item still has a well-formed response around it, this
    doesn't raise an error and would otherwise leave that screenshot
    silently missing from the final report with zero indication anything
    went wrong — exactly the "26 screenshots in, only some come out"
    failure mode. After every batch (whether the batch call itself
    succeeded or fell back to single-image calls after failing), this
    checks that every image in that batch actually got a result and
    retries — individually, so one problem image can't cost its whole
    batch — any that didn't, before moving on.

    If cache_path is given, results already on disk are loaded first (so
    screenshots analyzed in an earlier, interrupted run are never re-sent
    to the API) and the running results are saved back to disk after every
    single batch — so if the API limit is hit or the internet drops
    partway through, re-running the video picks up right after the last
    screenshot that finished, instead of redoing everything."""
    results = {}
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                results = {int(k): v for k, v in json.load(f).items()}
        except Exception:
            results = {}

    if results:
        before = len(image_items)
        image_items = [item for item in image_items if item[0] not in results]
        skipped = before - len(image_items)
        if skipped:
            print(f"    [resume] {skipped} screenshot(s) already analyzed, skipping.")

    if not image_items or not budget.available():
        return results

    def _save_progress():
        if cache_path:
            _safe_write_json(cache_path, {str(k): v for k, v in results.items()})

    for batch_start in range(0, len(image_items), VISION_BATCH_SIZE):
        if not budget.available():
            print(f"    ✗ Budget exhausted ({budget}), skipping remaining vision "
                  f"— re-run to resume from here.")
            break

        batch = image_items[batch_start:batch_start + VISION_BATCH_SIZE]
        print(f"    Vision batch {batch_start // VISION_BATCH_SIZE + 1}: "
              f"analyzing {len(batch)} screenshot(s)...")

        parts = []
        valid_batch = []

        for img_path_ctx in batch:
            ts = img_path_ctx[0]
            img_path = img_path_ctx[1]
            try:
                mime, img_data = _encode_image_for_vision(img_path)

                m, s = divmod(ts, 60)
                context = img_path_ctx[2] if len(img_path_ctx)>2 else ""
                parts.append({"text": f"\n--- Image {len(valid_batch)} "
                              f"(at {m:02d}:{s:02d}) ---\nTranscript Context:\n{context}\n"})
                parts.append({"inline_data": {"mime_type": mime, "data": img_data}})
                valid_batch.append((ts, img_path))
            except Exception as e:
                print(f"    ✗ Could not read {os.path.basename(img_path)}: {e}")

        if not valid_batch:
            continue

        prompt = BATCH_VISION_PROMPT.format(count=len(valid_batch),
                                            count_minus_1=len(valid_batch) - 1)
        parts.append({"text": prompt})

        contents = [{"role": "user", "parts": parts}]

        try:
            parsed = _call_gemini_with_fallback(api_key, VISION_MODEL_FALLBACKS, contents,
                                                max_tokens=VISION_BATCH_MAX_TOKENS, budget=budget)

            # Parse results — map each returned item back to ITS OWN image
            # using the "image_index" field the model was asked to return,
            # never by raw list position. The model doesn't always return
            # exactly one item per image in the same order it received them
            # (it occasionally skips a frame it considers redundant, merges
            # two, or reorders) — trusting position `i` in that case silently
            # shifts every caption/description/extracted-code after the gap
            # onto the WRONG screenshot for the rest of the batch. Using the
            # explicit index keeps every result glued to the correct image
            # even when the returned array is short, long, or out of order.
            if isinstance(parsed, list):
                unmatched = []
                claimed = set()
                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    idx = item.get('image_index')
                    if isinstance(idx, int) and 0 <= idx < len(valid_batch) and idx not in claimed:
                        ts = valid_batch[idx][0]
                        results[ts] = item
                        claimed.add(idx)
                    else:
                        unmatched.append(item)
                # Fallback for items with a missing/invalid/duplicate index —
                # place them (in order) into whichever slots are still empty,
                # so a malformed index doesn't lose the analysis entirely.
                if unmatched:
                    free_slots = [i for i in range(len(valid_batch)) if i not in claimed]
                    if len(unmatched) != len(free_slots):
                        print(f"    [!] Vision batch: {len(parsed)} result(s) for "
                              f"{len(valid_batch)} image(s) sent — some entries had "
                              f"no valid image_index; best-effort matching used.")
                    for item, slot in zip(unmatched, free_slots):
                        results[valid_batch[slot][0]] = item
            elif isinstance(parsed, dict):
                # Single result — apply to first image
                if valid_batch:
                    results[valid_batch[0][0]] = parsed

            _save_progress()  # checkpoint: this batch is safely done now

        except RuntimeError as e:
            print(f"    ✗ Vision batch failed: {e}")
            # Fallback: try single-image analysis for this batch
            for ts, img_path in valid_batch:
                if not budget.available():
                    break
                single_result = _analyze_screenshot_single(img_path, api_key, budget)
                if single_result:
                    results[ts] = single_result
                    _save_progress()  # checkpoint each single-image success too

        # ── Safety net: verify every image THIS batch was responsible for
        # actually landed in `results` — see the docstring above for why
        # this can't just be assumed even on a "successful" batch call.
        # Runs no matter which branch above ran, and retries only the
        # specific screenshots still missing (not the whole batch), so
        # it's cheap in the common case where nothing is actually missing.
        missing = [(ts, p) for ts, p in valid_batch if ts not in results]
        if missing and budget.available():
            print(f"    [!] {len(missing)}/{len(valid_batch)} screenshot(s) in "
                  f"this batch came back with no analysis — retrying "
                  f"individually...")
            for ts, img_path in missing:
                if not budget.available():
                    break
                single_result = _analyze_screenshot_single(img_path, api_key, budget)
                if single_result:
                    results[ts] = single_result
                    _save_progress()
            still_missing = [ts for ts, _ in missing if ts not in results]
            if still_missing:
                times = ", ".join(f"{t // 60:02d}:{t % 60:02d}" for t in still_missing)
                print(f"    ⚠ {len(still_missing)} screenshot(s) still couldn't be "
                      f"analyzed after retrying ({times}) — they'll still appear "
                      f"in the report, just without an AI caption/description.")

    return results


def _analyze_screenshot_single(image_path, api_key, budget):
    """Fallback: analyze a single screenshot. Used when batch fails."""
    if not budget.available():
        return None

    try:
        mime, img_data = _encode_image_for_vision(image_path)
    except Exception:
        return None

    contents = [{"role": "user", "parts": [
        {"inline_data": {"mime_type": mime, "data": img_data}},
        {"text": SINGLE_VISION_PROMPT},
    ]}]

    try:
        return _call_gemini_with_fallback(api_key, VISION_MODEL_FALLBACKS, contents,
                                          max_tokens=4096, budget=budget)
    except RuntimeError:
        return None


# ═══════════════════════════════════════════════════════════════════
# VIDEO DOWNLOAD & FRAME EXTRACTION
# ═══════════════════════════════════════════════════════════════════

def download_video(video_id, out_dir, quality='1080', keep_audio=False):
    """Download video. Returns path.

    quality: '480'/'720'/'1080'/'1440'/'2160'/'best' (numeric height cap, no
    'p' suffix — yt-dlp's format filter needs a plain number).
    keep_audio: False (default) downloads VIDEO-ONLY — screenshots don't
    need an audio track, so skipping it (and skipping the mux step) cuts
    download time and disk use with zero impact on frame/screenshot
    quality. Set True (used automatically with --keep-video) to get a
    normal watchable video+audio file instead."""
    out_path = str(out_dir / f"{video_id}.mp4")
    height_filter = '' if quality in (None, 'best') else f'[height<={quality}]'

    if keep_audio:
        fmt = f'best{height_filter}[ext=mp4]/best{height_filter}/best'
    else:
        fmt = (f'bestvideo{height_filter}[ext=mp4]/bestvideo{height_filter}'
              f'/best{height_filter}[ext=mp4]/best{height_filter}')

    ydl_opts = {
        'format': fmt,
        'outtmpl': out_path,
        'quiet': True,
        'merge_output_format': 'mp4',
    }
    with _download_progress(f"Downloading video ({quality}p)") as dp:
        ydl_opts['progress_hooks'] = [dp.hook]
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
    return out_path


def detect_slide_transitions(video_path, threshold=30.0, sample_interval=1.0):
    """Detect slide transitions via frame differencing (FREE, local)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sample_frames = max(1, int(fps * sample_interval))
    transitions = []
    prev_gray = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_frames == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (320, 240))
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                if float(np.mean(diff)) > threshold:
                    ts = int(frame_idx / fps)
                    if not transitions or (ts - transitions[-1]) > 2:
                        transitions.append(ts)
            prev_gray = gray
        frame_idx += 1

    cap.release()
    print(f"    Found {len(transitions)} slide transition(s) (local)")
    return transitions


def extract_frame(video_path, timestamp_sec, out_path):
    """Extract a single frame at timestamp via ffmpeg."""
    cmd = ['ffmpeg', '-y', '-ss', str(max(0, timestamp_sec)),
           '-i', video_path, '-frames:v', '1', '-q:v', '2', out_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def extract_best_unaltered_frame(video_path, target_sec, out_path, window_sec=5.0, cap=None):
    """Extract the highest quality frame with maximum edges and minimum face
    obstruction.

    Pass an already-open `cap` (cv2.VideoCapture) to reuse the same decoder
    across every screenshot for a video instead of re-opening the file each
    time — this is the dominant cost when a video has many screenshots, and
    reusing the handle doesn't change the algorithm or output in any way,
    so quality is identical, just faster. If `cap` is omitted, one is
    opened and closed just for this call (unchanged old behavior)."""
    owns_cap = cap is None
    if cap is None:
        cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        if owns_cap:
            cap.release()
        extract_frame(video_path, target_sec, out_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_sec = max(0, target_sec - window_sec)
    
    best_score = -999999
    best_frame = None

    # Sample 5 frames in the window leading up to the target timestamp
    for i in range(5):
        t = start_sec + (i * window_sec / 4.0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Edge Density (Maximize content/text on board)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # 2. Face Penalty (Minimize teacher obstruction)
        face_penalty = 0
        if _FACE_CASCADE is not None:
            faces = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            for (x, y, w, h) in faces:
                face_penalty += (w * h) / (gray.shape[0] * gray.shape[1])

        # Score = high content - high obstruction
        score = (edge_density * 100) - (face_penalty * 500)
        
        if score > best_score:
            best_score = score
            best_frame = frame

    if owns_cap:
        cap.release()

    if best_frame is not None:
        cv2.imwrite(out_path, best_frame)
    else:
        extract_frame(video_path, target_sec, out_path)


# ── Webcam PiP detection (FREE, local) ──

def _load_face_cascade():
    """Load Haar cascade for face detection."""
    cascade_filename = 'haarcascade_frontalface_default.xml'
    local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), cascade_filename)
    candidates = []
    cv2_data = getattr(cv2, 'data', None)
    if cv2_data is not None:
        candidates.append(os.path.join(cv2_data.haarcascades, cascade_filename))
    candidates.append(local_path)

    for path in candidates:
        if path and os.path.isfile(path):
            cascade = cv2.CascadeClassifier(path)
            if not cascade.empty():
                return cascade

    try:
        url = ('https://raw.githubusercontent.com/opencv/opencv/master/'
               'data/haarcascades/' + cascade_filename)
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        with open(local_path, 'wb') as f:
            f.write(resp.content)
        cascade = cv2.CascadeClassifier(local_path)
        if not cascade.empty():
            return cascade
    except Exception as e:
        print(f"  Warning: Haar cascade unavailable ({e})")
        return None
    return None


_FACE_CASCADE = _load_face_cascade()


def declutter_frame(image_path):
    """Remove webcam PiP via inpainting. Returns cleaned cv2 image, or None for talking-head."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    if _FACE_CASCADE is None:
        return img
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1,
                                            minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return img

    for (x, y, fw, fh) in faces:
        if (fw * fh) / (w * h) > 0.15:
            return None  # Dominant face = talking head

    # Use OpenCV inpainting to seamlessly fill the face region
    # instead of drawing ugly gray rectangles
    mask = np.zeros((h, w), dtype=np.uint8)
    result = img.copy()
    any_inpaint = False
    for (x, y, fw, fh) in faces:
        near_corner = ((x < w * 0.25 or x + fw > w * 0.75) and
                       (y < h * 0.25 or y + fh > h * 0.75))
        if near_corner:
            pad = int(0.03 * w)
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(w, x + fw + pad), min(h, y + fh + pad)
            mask[y0:y1, x0:x1] = 255
            any_inpaint = True

    if any_inpaint:
        result = cv2.inpaint(result, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    return result


def add_timestamp_overlay(img_bgr, timestamp_str):
    """Burn professional YouTube-style timestamp overlay on frame."""
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(img_rgb).convert('RGBA')
    overlay = PILImage.new('RGBA', pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("arialbd.ttf", 44) # Bold arial
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", 44)
        except Exception:
            font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), timestamp_str, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x, pad_y, margin = 20, 12, 25
    x1, y1 = w - tw - pad_x * 2 - margin, h - th - pad_y * 2 - margin
    x2, y2 = w - margin, h - margin
    draw.rounded_rectangle([x1, y1, x2, y2], radius=16, fill=(0, 0, 0, 190))
    draw.text((x1 + pad_x, y1 + pad_y - 4), timestamp_str, font=font, fill=(255, 255, 255, 255))
    combined = PILImage.alpha_composite(pil_img, overlay).convert('RGB')
    return cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)


# ═══════════════════════════════════════════════════════════════════
# FRAME DEDUPLICATION (FREE, local)
# ═══════════════════════════════════════════════════════════════════

def _local_ssim(gray_a, gray_b, window=7):
    """Windowed structural similarity between two equally-sized grayscale
    images, computed with box-filtered local statistics (mean, variance,
    covariance) instead of scikit-image, so this needs no extra
    dependency beyond OpenCV/NumPy (already required). Unlike a plain
    mean-absolute-pixel-difference — which averages over the WHOLE frame
    and is easily fooled by two images that share a large uniform
    background — this responds to local structural change (added/changed
    text, a moved diagram, a different equation) even when it only
    affects a small part of the frame, which is exactly what
    distinguishes "a genuinely different slide on the same template" from
    "the same slide held on screen for a few extra seconds"."""
    x = gray_a.astype(np.float64)
    y = gray_b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ksize = (window, window)
    mu_x = cv2.boxFilter(x, -1, ksize)
    mu_y = cv2.boxFilter(y, -1, ksize)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x2 = cv2.boxFilter(x * x, -1, ksize) - mu_x2
    sigma_y2 = cv2.boxFilter(y * y, -1, ksize) - mu_y2
    sigma_xy = cv2.boxFilter(x * y, -1, ksize) - mu_xy
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / \
               ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2))
    return float(np.mean(ssim_map))


def deduplicate_frames(screenshot_map, threshold=0.93, max_gap_sec=8):
    """Remove near-duplicate screenshots using structural similarity
    (SSIM) between nearby frames.

    Two frames are only ever compared — and only ever considered a
    duplicate — if they land within `max_gap_sec` seconds of each other.
    This matters because the timestamps feeding this function are NOT
    evenly spaced: they're specific moments (an AI-picked highlight, a
    detected slide transition) that can be seconds or many MINUTES apart.
    A previous version compared every frame only to whichever one it had
    most recently decided to keep, with no time limit at all — so two
    screenshots from completely different parts of a video (say, minute
    3 and minute 34) that simply used the same slide template, the same
    code editor theme, or the same whiteboard framing were treated as
    "the same shot" and the later, genuinely-different one was deleted
    before Vision analysis ever got to see it. On a video where every
    slide shares one template, this could silently erase most of the
    requested screenshots — exactly the "asked for 26, kept only a
    handful" failure. Restricting comparisons to genuinely nearby
    timestamps means this only ever fires for its intended case: the same
    slide/frame held on screen long enough that more than one timestamp
    landed on it.

    The similarity metric itself was also upgraded from a plain whole-
    frame mean-pixel-difference to windowed SSIM (see _local_ssim above):
    measured on real slide pairs, two DIFFERENT slides sharing one
    template scored ~0.97 similarity on the old metric — a hair's-width
    below a same-image score of 1.00 — versus ~0.97 on SSIM too but with
    genuinely different-content pairs scoring materially lower (~0.88),
    giving a threshold real room to tell them apart instead of splitting
    a near-zero margin."""
    if len(screenshot_map) <= 1:
        return screenshot_map

    sorted_ts = sorted(screenshot_map.keys())
    keep = {}
    prev_gray = None
    prev_ts = None
    removed = 0

    for ts in sorted_ts:
        img = cv2.imread(screenshot_map[ts])
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, (160, 120))

        if prev_gray is not None and prev_ts is not None and (ts - prev_ts) <= max_gap_sec:
            similarity = _local_ssim(prev_gray, gray_small)
            if similarity > threshold:
                # Close in time AND structurally near-identical —
                # genuinely the same moment held on screen, not two
                # different discussions that just look alike.
                try:
                    os.remove(screenshot_map[ts])
                except Exception as e:
                    _log.debug("Couldn't remove duplicate frame %s: %s", screenshot_map[ts], e)
                removed += 1
                continue

        keep[ts] = screenshot_map[ts]
        prev_gray = gray_small
        prev_ts = ts

    if removed:
        print(f"    Dedup: removed {removed} near-duplicate frame(s) (local)")
    return keep


# ═══════════════════════════════════════════════════════════════════
# SMART VISUAL CROPPING (FREE, local)
# ═══════════════════════════════════════════════════════════════════

def smart_crop_visual(image_path, bbox_pct, output_path):
    """Crop region based on Vision AI percentage coordinates."""
    img = cv2.imread(image_path)
    if img is None:
        return False
    h, w = img.shape[:2]

    x = int(w * bbox_pct[0] / 100)
    y = int(h * bbox_pct[1] / 100)
    cw = int(w * bbox_pct[2] / 100)
    ch = int(h * bbox_pct[3] / 100)

    # 5% padding
    pad_x, pad_y = int(cw * 0.05), int(ch * 0.05)
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    cw = min(w - x, cw + 2 * pad_x)
    ch = min(h - y, ch + 2 * pad_y)

    if cw < 50 or ch < 50:
        return False

    cropped = img[y:y + ch, x:x + cw]
    if cropped.size == 0:
        return False

    cv2.imwrite(output_path, cropped)
    return True


def auto_crop_content_region(image_path):
    """Fallback: auto-crop via contour detection (FREE, local)."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return img

    best, best_area = None, 0
    full_area = w * h
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if best_area < area < full_area * 0.95 and area > full_area * 0.05:
            best, best_area = cnt, area

    if best is None:
        return img

    x, y, cw, ch = cv2.boundingRect(best)
    pad = int(0.02 * max(w, h))
    x, y = max(0, x - pad), max(0, y - pad)
    cw = min(w - x, cw + 2 * pad)
    ch = min(h - y, ch + 2 * pad)
    return img[y:y + ch, x:x + cw]


# ═══════════════════════════════════════════════════════════════════
# FLOWCHART GENERATION (FREE — mermaid.ink API)
# ═══════════════════════════════════════════════════════════════════

def generate_flowchart_image(mermaid_code, output_path):
    """Render Mermaid → PNG via mermaid.ink (free, no API key needed)."""
    if not mermaid_code or not mermaid_code.strip():
        return False

    try:
        encoded = base64.urlsafe_b64encode(
            mermaid_code.encode('utf-8')).decode('utf-8')
        url = f"https://mermaid.ink/img/{encoded}?type=png&bgColor=white"
        resp = requests.get(url, timeout=30)

        if resp.ok and 'image' in resp.headers.get('content-type', ''):
            with open(output_path, 'wb') as f:
                f.write(resp.content)
            img = cv2.imread(output_path)
            if img is not None and img.size > 0:
                return True
            os.remove(output_path)
    except Exception as e:
        print(f"    [!] Flowchart render failed: {e}")
    return False


# ═══════════════════════════════════════════════════════════════════
# SCREENSHOT TIMESTAMP COLLECTION
# ═══════════════════════════════════════════════════════════════════

def collect_screenshot_timestamps(analysis_result, slide_transitions=None,
                                  max_screenshots=50):
    """Merge AI timestamps + slide transitions, deduplicate (2s window)."""
    seen = set()
    timestamps = []

    for section in analysis_result.get('sections', []):
        for ss in section.get('screenshot_timestamps', []):
            sec = ss['seconds']
            bucket = sec // 2
            if bucket not in seen:
                seen.add(bucket)
                timestamps.append((sec, ss['visual_type'], ss['description']))

        if section.get('has_visual'):
            sec = section['start_seconds']
            bucket = sec // 2
            if bucket not in seen:
                seen.add(bucket)
                timestamps.append((sec, 'general', 'section visual'))

    if slide_transitions:
        for sec in slide_transitions:
            bucket = sec // 2
            if bucket not in seen:
                seen.add(bucket)
                timestamps.append((sec, 'slide', 'slide transition'))

    timestamps.sort(key=lambda x: x[0])

    if len(timestamps) > max_screenshots:
        step = len(timestamps) / max_screenshots
        timestamps = [timestamps[int(i * step)] for i in range(max_screenshots)]

    return timestamps


# ═══════════════════════════════════════════════════════════════════
# PDF BUILDING
# ═══════════════════════════════════════════════════════════════════

def escape_xml(text):
    return str(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def inline_format(text):
    escaped = escape_xml(text)
    # Inline code: `code` -> <font name="Courier" color="#D6336C">code</font>
    escaped = re.sub(r'`([^`]+)`', r'<font name="Courier" color="#D6336C">\1</font>', escaped)
    # Inline math: \(math\) -> <font color="#2563EB"><i>math</i></font>
    escaped = re.sub(r'\\\((.+?)\\\)', r'<font color="#2563EB"><i>\1</i></font>', escaped)
    # Also attempt to clean up any $...$ that sneaks through, but require it to contain letters to avoid currency overlaps like $100 - $200
    escaped = re.sub(r'(?<!\$)\$([a-zA-Z][^\$]*?|[^\$]*?[a-zA-Z][^\$]*?)\$(?!\$)', r'<font color="#2563EB"><i>\1</i></font>', escaped)
    # Bold: **bold**
    escaped = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', escaped)
    # Italic: *italic*
    escaped = re.sub(r'(?<!\*)\*(.+?)\*(?!\*)', r'<i>\1</i>', escaped)
    return escaped


def markdown_to_flowables(md_text, styles):
    """Convert markdown-ish text to ReportLab flowables, including block math and quotes."""
    flowables = []
    bullet_buffer = []
    lines = md_text.replace('\r\n', '\n').split('\n')
    i = 0

    def flush_bullets():
        nonlocal bullet_buffer
        if bullet_buffer:
            items = [ListItem(Paragraph(inline_format(b), styles['Normal']))
                     for b in bullet_buffer]
            flowables.append(ListFlowable(items, bulletType='bullet', leftIndent=14, spaceAfter=8))
            bullet_buffer = []

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            flush_bullets()
            i += 1
            continue

        # Block Math
        if line.startswith('$$'):
            flush_bullets()
            math_content = line[2:].strip()
            # If closed on same line: $$ E=mc^2 $$
            if math_content.endswith('$$') and len(math_content) >= 2:
                math_content = math_content[:-2].strip()
                i += 1
            else:
                # Multi-line math block
                math_lines = [math_content] if math_content else []
                i += 1
                while i < len(lines):
                    if lines[i].strip().endswith('$$'):
                        math_lines.append(lines[i].strip()[:-2].strip())
                        i += 1
                        break
                    math_lines.append(lines[i].strip())
                    i += 1
                math_content = '\n'.join(math_lines)
            
            math_style = ParagraphStyle('MathBlock', parent=styles['Normal'],
                                        textColor=colors.HexColor('#1E3A8A'),
                                        alignment=1, fontName='Helvetica-Oblique', fontSize=11)
            t = Table([[Paragraph(escape_xml(math_content), math_style)]], colWidths=[5 * inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#EFF6FF')),
                ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#BFDBFE')),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ]))
            flowables.append(t)
            flowables.append(Spacer(1, 8))
            continue

        # Blockquotes
        if line.startswith('>'):
            flush_bullets()
            quote_text = line[1:].strip()
            quote_style = ParagraphStyle('Blockquote', parent=styles['Normal'],
                                         textColor=colors.HexColor('#4B5563'),
                                         leftIndent=10)
            t = Table([[Paragraph(inline_format(quote_text), quote_style)]], colWidths=[5.2 * inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F9FAFB')),
                ('LINEBEFORE', (0, 0), (0, -1), 3, colors.HexColor('#9CA3AF')),
                ('LEFTPADDING', (0, 0), (-1, -1), 10),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            flowables.append(t)
            flowables.append(Spacer(1, 6))
            i += 1
            continue

        # Bullets
        bm = re.match(r'^[-*•]\s+(.*)', line)
        if bm:
            bullet_buffer.append(bm.group(1))
            i += 1
            continue
        nm = re.match(r'^\d+[.)]\s+(.*)', line)
        if nm:
            bullet_buffer.append(nm.group(1))
            i += 1
            continue
            
        flush_bullets()
        flowables.append(Paragraph(inline_format(line), styles['Normal']))
        flowables.append(Spacer(1, 4))
        i += 1

    flush_bullets()
    return flowables


def build_code_block_flowable(code_text, language=''):
    """Dark-themed code block for PDF."""
    code_style = ParagraphStyle('CodeBlockText', fontName='Courier',
                                fontSize=7.5, leading=10,
                                textColor=colors.HexColor('#D4D4D4'))
    display = ''
    if language:
        display = f"  [{language}]\n\n"
    display += code_text

    pre = Preformatted(display, code_style)
    t = Table([[pre]], colWidths=[5.2 * inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#1E293B')),  # Modern Slate Dark
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#334155')),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
    ]))
    return t


def build_table_flowable(table_data, styles):
    """Styled data table."""
    if not table_data or not table_data.get('headers') or not table_data.get('rows'):
        return None
    headers = [Paragraph(f"<b>{escape_xml(h)}</b>", styles['Normal'])
               for h in table_data['headers']]
    data = [headers]
    for row in table_data['rows']:
        data.append([Paragraph(escape_xml(str(c)), styles['Normal']) for c in row])
    t = Table(data, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2A333A')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#999999')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.white, colors.HexColor('#F2F2F2')]),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    return t


def add_image_to_story(story, img_path, max_width=5.4 * inch,
                       max_height=4.0 * inch, caption=None, styles=None):
    """Add image with proper aspect ratio scaling."""
    if not os.path.exists(img_path):
        return
    try:
        pil_img = PILImage.open(img_path)
        iw, ih = pil_img.size
        if iw <= 0 or ih <= 0:
            return
        aspect = iw / ih
        if iw / max_width > ih / max_height:
            w = min(max_width, iw)
            h = w / aspect
        else:
            h = min(max_height, ih)
            w = h * aspect
        story.append(RLImage(str(img_path), width=w, height=h))
        if caption and styles:
            cap_style = ParagraphStyle('ImgCap', parent=styles['Normal'],
                                       fontSize=8,
                                       textColor=colors.HexColor('#666'),
                                       alignment=1)
            story.append(Paragraph(escape_xml(caption), cap_style))
    except Exception as e:
        _log.debug("Couldn't add image %s to PDF: %s", img_path, e)


def _register_hindi_font():
    """Register a Hindi-capable TTF font for PDF rendering.
    Searches for Nirmala UI (Win10+) or Mangal (older Windows) and registers it.
    Returns the registered font name, or 'Helvetica' as fallback."""
    font_candidates = [
        ('NirmalaUI', r'C:\Windows\Fonts\Nirmala.ttf'),
        ('NirmalaUI', r'C:\Windows\Fonts\nirmala.ttf'),
        ('Mangal', r'C:\Windows\Fonts\mangal.ttf'),
        ('Mangal', r'C:\Windows\Fonts\Mangal.ttf'),
    ]
    # Also check Linux / Mac common paths
    linux_fonts = [
        ('NotoSansDevanagari', '/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf'),
        ('FreeSans', '/usr/share/fonts/truetype/freefont/FreeSans.ttf'),
    ]
    font_candidates.extend(linux_fonts)

    for font_name, font_path in font_candidates:
        if os.path.isfile(font_path):
            try:
                pdfmetrics.registerFont(TTFont(font_name, font_path))
                return font_name
            except Exception:
                continue
    return 'Helvetica'


_HINDI_FONT = _register_hindi_font()


def _yt_ts_link(source_url, seconds):
    """Build a YouTube deep-link that jumps straight to `seconds` into the
    video (e.g. https://www.youtube.com/watch?v=XXXX&t=123s)."""
    sep = '&' if '?' in source_url else '?'
    return f"{source_url}{sep}t={max(0, int(seconds))}s"


class _BookmarkedDocTemplate(SimpleDocTemplate):
    """SimpleDocTemplate that drops a real PDF outline/bookmark entry for
    every flowable carrying a `_bookmark = (key, title, level)` attribute.
    This makes each report.pdf individually navigable via the viewer's
    bookmark sidebar, and — because merge_pdfs() imports each file's
    outline as nested children of that video's top-level entry — it's also
    what turns combined_report.pdf's bookmark panel into a full course
    index (video -> section, click to jump to the page)."""

    def afterFlowable(self, flowable):
        bookmark = getattr(flowable, '_bookmark', None)
        if bookmark:
            key, text, level = bookmark
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=level, closed=True)


def build_pdf_report(video_dir, video_title, channel, source_url,
                     analysis, screenshot_map, cropped_visuals_map,
                     vision_code_blocks, local_code_blocks,
                     flowchart_images, key_takeaways=None, quiz_cards=None,
                     output_name='report'):
    """Build comprehensive PDF with ALL visual content, including Hindi
    support, per-section PDF bookmarks (master index), importance tags,
    and per-section "watch this part" deep-links."""
    pdf_path = video_dir / f'{output_name}.pdf'
    doc = _BookmarkedDocTemplate(str(pdf_path), pagesize=A4,
                                 topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    styles = getSampleStyleSheet()

    # Override all styles to use Hindi-capable font
    for style_name in styles.byName:
        styles[style_name].fontName = _HINDI_FONT

    hs = ParagraphStyle('SecHead', parent=styles['Heading2'],
                         fontName=_HINDI_FONT,
                         textColor=colors.HexColor('#1B2430'), spaceBefore=14)
    sub = ParagraphStyle('Sub', parent=styles['Normal'], fontSize=9,
                          fontName=_HINDI_FONT,
                          textColor=colors.HexColor('#888'))
    fclbl = ParagraphStyle('FCLbl', parent=styles['Normal'], fontSize=9,
                            fontName=_HINDI_FONT,
                            textColor=colors.HexColor('#4B5563'), alignment=1)
    cdesc = ParagraphStyle('CDesc', parent=styles['Normal'], fontSize=8,
                            fontName=_HINDI_FONT,
                            textColor=colors.HexColor('#6B7280'), leftIndent=10)
    croplbl = ParagraphStyle('CropLbl', parent=styles['Normal'], fontSize=8,
                              fontName=_HINDI_FONT,
                              textColor=colors.HexColor('#4B5563'),
                              alignment=1, spaceAfter=4)
    linklbl = ParagraphStyle('LinkLbl', parent=styles['Normal'], fontSize=8.5,
                              fontName=_HINDI_FONT,
                              textColor=colors.HexColor('#2563EB'), spaceAfter=10)
    extralbl = ParagraphStyle('ExtraLbl', parent=styles['Normal'], fontSize=8,
                               fontName=_HINDI_FONT, textColor=colors.HexColor('#B45309'))

    vtype = analysis.get('video_type', 'general')
    sections = analysis.get('sections', [])
    type_labels = {
        'coding': 'Coding Tutorial', 'slides': 'Presentation',
        'lecture': 'Lecture', 'tutorial': 'Tutorial',
        'trading': 'Trading/Finance', 'whiteboard': 'Whiteboard',
        'general': 'General',
    }

    # ── Header / Footer Callbacks ──
    def on_first_page(canvas, doc):
        canvas.saveState()
        canvas.restoreState()

    def on_later_pages(canvas, doc):
        canvas.saveState()
        canvas.setFont(_HINDI_FONT, 9)
        canvas.setFillColor(colors.HexColor('#6B7280'))
        # Header
        canvas.drawString(0.6 * inch, A4[1] - 0.45 * inch, escape_xml(video_title[:60] + ("..." if len(video_title) > 60 else "")))
        canvas.setStrokeColor(colors.HexColor('#E5E7EB'))
        canvas.line(0.6 * inch, A4[1] - 0.55 * inch, A4[0] - 0.6 * inch, A4[1] - 0.55 * inch)
        # Footer
        canvas.drawString(A4[0] / 2.0 - 0.25 * inch, 0.4 * inch, f"Page {doc.page}")
        canvas.restoreState()

    # ── Title Page ──
    # Large Title
    title_style = ParagraphStyle('MainTitle', parent=styles['Title'], fontName=_HINDI_FONT,
                                 fontSize=24, leading=28, textColor=colors.HexColor('#1E3A8A'),
                                 spaceAfter=20, alignment=1)
    story = [Spacer(1, 2 * inch)]
    story.append(Paragraph(escape_xml(video_title), title_style))
    story.append(Spacer(1, 0.5 * inch))

    # Meta Info Table
    meta_style = ParagraphStyle('Meta', parent=styles['Normal'], fontName=_HINDI_FONT,
                                fontSize=11, textColor=colors.HexColor('#4B5563'), alignment=1)
    meta_data = []
    if channel:
        meta_data.append([Paragraph(f"<b>Channel:</b> {escape_xml(channel)}", meta_style)])
    meta_data.append([Paragraph(f"<b>Source:</b> <link href='{escape_xml(source_url)}' color='blue'>{escape_xml(source_url)}</link>", meta_style)])
    meta_data.append([Paragraph(f"<b>Type:</b> {escape_xml(type_labels.get(vtype, vtype))}", meta_style)])
    meta_data.append([Paragraph(f"<b>Sections:</b> {len(sections)}", meta_style)])
    
    meta_table = Table(meta_data, colWidths=[6 * inch])
    meta_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(meta_table)
    story.append(PageBreak())

    # ── Key Takeaways (optional, --summary) ──
    if key_takeaways:
        kt_head = Paragraph("Key Takeaways", ParagraphStyle(
            'KTHead', parent=styles['Heading2'], fontName=_HINDI_FONT,
            textColor=colors.HexColor('#1B2430')))
        kt_head._bookmark = ('takeaways', 'Key Takeaways', 0)
        story.append(kt_head)
        items = [ListItem(Paragraph(inline_format(t), styles['Normal']))
                 for t in key_takeaways]
        story.append(ListFlowable(items, bulletType='bullet', leftIndent=14, spaceAfter=8))
        story.append(PageBreak())

    # ── Overview Flowchart ──
    ofc = flowchart_images.get('overview')
    if ofc and os.path.exists(ofc):
        story.append(Paragraph("Video Overview", styles['Heading3']))
        add_image_to_story(story, ofc, max_width=5.4 * inch,
                          max_height=3.5 * inch,
                          caption="Content flow overview", styles=styles)
        story.append(Spacer(1, 16))

    # ── Sections ──
    all_ss_in_section = {}
    for sec in sections:
        for ss in sec.get('screenshot_timestamps', []):
            all_ss_in_section.setdefault(sec['start_seconds'], []).append(ss)

    for idx, sec in enumerate(sections):
        m, s = divmod(sec['start_seconds'], 60)
        htxt = f"[{m:02d}:{s:02d}]"
        if sec['heading']:
            htxt += f" {sec['heading']}"
        if sec.get('importance') == 'extra':
            htxt += "  (extra)"
        head_para = Paragraph(escape_xml(htxt), hs)
        bookmark_title = (sec['heading'] or f"[{m:02d}:{s:02d}]")[:100]
        head_para._bookmark = (f'sec_{idx}', bookmark_title, 0)
        story.append(head_para)
        story.append(Paragraph(
            f'<link href="{escape_xml(_yt_ts_link(source_url, sec["start_seconds"]))}" '
            f'color="#2563EB">▶ Watch this part at {m:02d}:{s:02d}</link>', linklbl))

        # Notes
        story.extend(markdown_to_flowables(sec['notes'], styles))

        # Code blocks from transcript (Pass 1)
        for cb in sec.get('code_blocks', []):
            story.append(Spacer(1, 8))
            if cb.get('description'):
                story.append(Paragraph(escape_xml(cb['description']), cdesc))
                story.append(Spacer(1, 4))
            story.append(build_code_block_flowable(cb['code'],
                                                    cb.get('language', '')))
            story.append(Spacer(1, 8))

        # Table
        tf = build_table_flowable(sec.get('table'), styles)
        if tf:
            story.append(Spacer(1, 6))
            story.append(tf)
            story.append(Spacer(1, 8))

        # ── ALL visual content for this section ──
        for ss in sec.get('screenshot_timestamps', []):
            ts = ss['seconds']

            # Vision-extracted code
            vcb = vision_code_blocks.get(ts)
            if vcb and vcb.get('code'):
                story.append(Spacer(1, 6))
                story.append(Paragraph(
                    f"<i>Code from screen [{ts // 60:02d}:{ts % 60:02d}]:</i>",
                    cdesc))
                story.append(Spacer(1, 4))
                story.append(build_code_block_flowable(
                    vcb['code'], vcb.get('language', '')))
                story.append(Spacer(1, 6))

            # Local OCR code (if no vision code found)
            if not vcb:
                lcb = local_code_blocks.get(ts)
                if lcb and lcb.get('code'):
                    story.append(Spacer(1, 6))
                    story.append(Paragraph(
                        f"<i>Code detected [{ts // 60:02d}:{ts % 60:02d}]:</i>",
                        cdesc))
                    story.append(Spacer(1, 4))
                    story.append(build_code_block_flowable(
                        lcb['code'], lcb.get('language', '')))
                    story.append(Spacer(1, 6))

            # Cropped visual elements (diagrams, charts, formulas, etc.)
            crops = cropped_visuals_map.get(ts, [])
            for ci in crops:
                cp = ci.get('path')
                cd = ci.get('description', ci.get('type', 'Visual element'))
                if cp and os.path.exists(cp):
                    story.append(Spacer(1, 6))
                    story.append(Paragraph(
                        f"▸ {escape_xml(cd)}", croplbl))
                    add_image_to_story(story, cp, max_width=5.0 * inch,
                                      max_height=3.5 * inch, styles=styles)

            # ALWAYS include the full screenshot (if useful)
            # This ensures ALL visual info is in the report
            full_img = screenshot_map.get(ts)
            if full_img and os.path.exists(full_img):
                story.append(Spacer(1, 6))
                ts_lbl = f"[{ts // 60:02d}:{ts % 60:02d}]"
                desc = ss.get('description', '')
                if desc:
                    ts_lbl += f" — {desc}"
                add_image_to_story(story, full_img, max_width=5.4 * inch,
                                  max_height=3.2 * inch,
                                  caption=ts_lbl, styles=styles)

        # Section flowchart
        fp = flowchart_images.get(idx)
        if fp and os.path.exists(fp):
            story.append(Spacer(1, 8))
            story.append(Paragraph("Process Flow:", fclbl))
            add_image_to_story(story, fp, max_width=5.0 * inch,
                              max_height=3.0 * inch, styles=styles)

        story.append(Spacer(1, 20))

    # ── Revision Quiz appendix (optional, --quiz) ──
    if quiz_cards:
        story.append(PageBreak())
        q_head = Paragraph("Revision Quiz", ParagraphStyle(
            'QuizHead', parent=styles['Heading2'], fontName=_HINDI_FONT,
            textColor=colors.HexColor('#1B2430')))
        q_head._bookmark = ('quiz', 'Revision Quiz', 0)
        story.append(q_head)
        story.append(Paragraph(
            f"{len(quiz_cards)} question(s) — also exported as flashcards.txt "
            f"(Anki-importable).", sub))
        story.append(Spacer(1, 10))
        q_style = ParagraphStyle('QStyle', parent=styles['Normal'], fontName=_HINDI_FONT,
                                 fontSize=10.5, spaceBefore=10)
        a_style = ParagraphStyle('AStyle', parent=styles['Normal'], fontName=_HINDI_FONT,
                                 fontSize=10, textColor=colors.HexColor('#374151'),
                                 leftIndent=14, spaceAfter=4)
        for i, card in enumerate(quiz_cards, 1):
            story.append(Paragraph(f"<b>Q{i}.</b> {inline_format(card['question'])}", q_style))
            story.append(Paragraph(f"<b>A:</b> {inline_format(card['answer'])}", a_style))

    doc.build(story, onFirstPage=on_first_page, onLaterPages=on_later_pages)
    return pdf_path


# ═══════════════════════════════════════════════════════════════════
# MARKDOWN REPORT BUILDER
# ═══════════════════════════════════════════════════════════════════


import base64
def get_base64_image(image_path):
    if not image_path or not os.path.exists(image_path): return ""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

# ═══════════════════════════════════════════════════════════════════
# HTML REPORT BUILDER
# ═══════════════════════════════════════════════════════════════════
def _fmt_duration(seconds):
    """1234 -> '20:34' or '1:05:12'."""
    try:
        seconds = int(seconds)
    except (TypeError, ValueError):
        return None
    if seconds <= 0:
        return None
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _fmt_count(n):
    """12345 -> '12.3K', 2500000 -> '2.5M'."""
    try:
        n = int(n)
    except (TypeError, ValueError):
        return None
    if n <= 0:
        return None
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _fmt_upload_date(d):
    """'20260115' -> '15 Jan 2026'."""
    if not d or len(str(d)) != 8:
        return None
    try:
        return datetime.strptime(str(d), "%Y%m%d").strftime("%d %b %Y")
    except Exception:
        return None


# Matches $$...$$, \(...\), \[...\] blocks so they survive Markdown untouched.
_MATH_PATTERN = re.compile(r'(\$\$.+?\$\$|\\\(.+?\\\)|\\\[.+?\\\])', re.DOTALL)


def _protect_math(text):
    """Swap LaTeX math blocks for plain placeholders so Markdown's underscore/
    asterisk handling doesn't mangle them before MathJax gets to render them."""
    placeholders = {}

    def repl(m):
        key = f"MATHPLACEHOLDERZ{len(placeholders)}ZEND"
        placeholders[key] = m.group(0)
        return key

    return _MATH_PATTERN.sub(repl, text or ''), placeholders


def _restore_math(rendered_html, placeholders):
    for key, val in placeholders.items():
        rendered_html = rendered_html.replace(key, val)
    return rendered_html


def _render_code_html(code, language=''):
    """Syntax-highlight with Pygments if available, else a plain <pre> block."""
    code = code or ''
    try:
        from pygments import highlight
        from pygments.lexers import get_lexer_by_name, guess_lexer
        from pygments.formatters import HtmlFormatter
        try:
            lexer = get_lexer_by_name(language) if language else guess_lexer(code)
        except Exception:
            lexer = guess_lexer(code)
        return highlight(code, lexer, HtmlFormatter(nowrap=False, cssclass='codehilite'))
    except Exception:
        return f'<pre class="codehilite"><code>{html_escape_module.escape(code)}</code></pre>'


def _render_table_html(table):
    """Render a {headers, rows} dict as a real HTML table."""
    if not table or not table.get('headers') or not table.get('rows'):
        return ''
    esc = html_escape_module.escape
    thead = ''.join(f'<th>{esc(str(h))}</th>' for h in table['headers'])
    tbody = ''
    for row in table['rows']:
        cells = ''.join(f'<td>{esc(str(c))}</td>' for c in row)
        tbody += f'<tr>{cells}</tr>'
    return (f'<div class="table-wrapper"><table><thead><tr>{thead}</tr></thead>'
            f'<tbody>{tbody}</tbody></table></div>')


_HTML_CSS_TEMPLATE = """
    :root { --bg: #f8fafc; --card-bg: #ffffff; --text: #1e293b; --text-muted: #64748b; --accent: #2563eb; --border: #e2e8f0; }
    @media (prefers-color-scheme: dark) { :root { --bg: #0f172a; --card-bg: #1e293b; --text: #f8fafc; --text-muted: #94a3b8; --accent: #3b82f6; --border: #334155; } }
    :root[data-theme="dark"] { --bg: #0f172a; --card-bg: #1e293b; --text: #f8fafc; --text-muted: #94a3b8; --accent: #3b82f6; --border: #334155; }
    :root[data-theme="light"] { --bg: #f8fafc; --card-bg: #ffffff; --text: #1e293b; --text-muted: #64748b; --accent: #2563eb; --border: #e2e8f0; }
    * { box-sizing: border-box; }
    body { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); line-height: 1.7; margin: 0; padding: 0 0 40px; transition: background .2s, color .2s; }
    .container { max-width: 900px; margin: 0 auto; padding: 0 20px; }
    .hero { __HERO_BG__ background-size: cover; background-position: center; padding: 56px 20px 36px; text-align: center; color: #fff; margin-bottom: 24px; }
    .hero h1 { font-size: 2rem; font-weight: 800; margin: 16px auto 12px; max-width: 800px; }
    .channel-row { display: flex; align-items: center; justify-content: center; }
    .channel-badge { display: inline-flex; align-items: center; gap: 10px; background: rgba(15,23,42,0.35); backdrop-filter: blur(6px); -webkit-backdrop-filter: blur(6px); border: 1px solid rgba(255,255,255,0.25); border-radius: 999px; padding: 6px 18px 6px 6px; box-shadow: 0 4px 10px rgba(0,0,0,0.15); }
    .channel-avatar { width: 44px; height: 44px; border-radius: 50%; object-fit: cover; border: 2px solid rgba(255,255,255,0.9); flex-shrink: 0; }
    .channel-avatar-fallback { display: flex; align-items: center; justify-content: center; background: var(--accent); color: #fff; font-weight: 700; font-size: 1.1rem; }
    .channel-name { font-weight: 700; font-size: 0.98rem; letter-spacing: 0.2px; }
    .meta-chips { display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; margin: 16px 0; }
    .chip { background: rgba(255,255,255,0.18); padding: 5px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 500; backdrop-filter: blur(4px); }
    .watch-btn { display: inline-block; margin-top: 8px; background: var(--accent); color: #fff !important; padding: 10px 22px; border-radius: 24px; text-decoration: none; font-weight: 600; }
    .toc { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; max-width: 900px; margin: 0 auto 36px; padding: 0 20px; }
    .toc-chip { background: var(--card-bg); border: 1px solid var(--border); color: var(--text) !important; padding: 6px 14px; border-radius: 20px; font-size: 0.82rem; text-decoration: none; font-weight: 500; }
    .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 16px; padding: 40px; margin-bottom: 30px; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); scroll-margin-top: 16px; }
    .timestamp-badge { display: inline-flex; align-items: center; gap: 6px; background: var(--accent); color: white; padding: 6px 14px; border-radius: 20px; font-weight: 600; font-size: 0.95rem; margin-bottom: 8px; box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.3); }
    .badges-row { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin-bottom: 20px; }
    .extra-badge { display: inline-block; background: #fef3c7; color: #b45309; padding: 4px 12px; border-radius: 20px; font-size: 0.78rem; font-weight: 700; }
    .jump-link { display: inline-flex; align-items: center; gap: 4px; font-size: 0.85rem; color: var(--accent) !important; text-decoration: none; font-weight: 600; }
    .jump-link:hover { text-decoration: underline; }
    h2 { margin-top: 0; font-size: 1.6rem; border-bottom: 2px solid var(--border); padding-bottom: 15px; margin-bottom: 20px; }
    img { max-width: 100%; border-radius: 12px; margin-top: 25px; border: 1px solid var(--border); box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
    .content { font-size: 1.05rem; }
    .content strong { color: var(--accent); }
    .content table, .table-wrapper table { border-collapse: collapse; width: 100%; margin: 16px 0; }
    .content th, .content td, .table-wrapper th, .table-wrapper td { border: 1px solid var(--border); padding: 8px 12px; text-align: left; }
    .content th, .table-wrapper th { background: var(--accent); color: #fff; }
    .table-wrapper { overflow-x: auto; margin-top: 20px; }
    .codehilite, pre { border-radius: 10px; padding: 16px; overflow-x: auto; font-size: 0.88rem; }
    .content code { background: rgba(37,99,235,0.12); padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }
    .content pre code { background: none; padding: 0; }
    .code-label { font-size: 0.85rem; color: var(--text-muted); font-weight: 600; margin-top: 22px; margin-bottom: 6px; }
    .screenshot-wrapper { text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px dashed var(--border); }
    .shot-link { display: block; margin-top: 8px; font-size: 0.85rem; }
    .shot-link a { color: var(--accent) !important; text-decoration: none; font-weight: 600; }

    /* Key takeaways */
    .takeaways { background: var(--card-bg); border: 1px solid var(--border); border-left: 4px solid var(--accent); border-radius: 14px; padding: 26px 32px; margin: 0 auto 30px; max-width: 900px; }
    .takeaways h3 { margin: 0 0 12px; font-size: 1.1rem; }
    .takeaways ul { margin: 0; padding-left: 20px; }
    .takeaways li { margin-bottom: 8px; }

    /* Floating controls */
    .fab-group { position: fixed; bottom: 22px; right: 22px; z-index: 80; display: flex; flex-direction: column; gap: 10px; }
    .fab { width: 48px; height: 48px; border-radius: 50%; background: var(--accent); color: #fff; border: none; box-shadow: 0 6px 16px rgba(0,0,0,0.3); font-size: 1.25rem; cursor: pointer; display: flex; align-items: center; justify-content: center; }
    .fab:hover { filter: brightness(1.1); }

    /* Sidebar drawer */
    .drawer-backdrop { position: fixed; inset: 0; background: rgba(0,0,0,0.45); opacity: 0; pointer-events: none; transition: opacity .2s; z-index: 90; }
    .drawer-backdrop.open { opacity: 1; pointer-events: auto; }
    .side-drawer { position: fixed; top: 0; left: 0; bottom: 0; width: 310px; max-width: 85vw; background: var(--card-bg); border-right: 1px solid var(--border); z-index: 100; transform: translateX(-105%); transition: transform .25s ease; overflow-y: auto; padding: 22px 0; }
    .side-drawer.open { transform: translateX(0); }
    .side-drawer h3 { padding: 0 22px; margin: 0 0 14px; font-size: 1rem; color: var(--text-muted); }
    .side-drawer a { display: flex; align-items: center; gap: 9px; padding: 11px 22px; color: var(--text) !important; text-decoration: none; font-size: 0.9rem; border-left: 3px solid transparent; }
    .side-drawer a:hover { background: rgba(37,99,235,0.08); border-left-color: var(--accent); }
    .side-drawer .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--accent); flex-shrink: 0; }
    .side-drawer .dot.extra { background: #d97706; }

    /* Language picker (client-side translation via Gemini) */
    .lang-panel { position: fixed; bottom: 84px; right: 22px; z-index: 100; background: var(--card-bg); border: 1px solid var(--border); border-radius: 14px; box-shadow: 0 10px 25px rgba(0,0,0,0.25); padding: 14px; width: 220px; max-height: 60vh; overflow-y: auto; display: none; }
    .lang-panel.open { display: block; }
    .lang-panel h4 { margin: 0 0 10px; font-size: 0.85rem; color: var(--text-muted); font-weight: 600; }
    .lang-panel .lang-note { font-size: 0.72rem; color: var(--text-muted); margin-bottom: 10px; line-height: 1.4; }
    .lang-btn { display: block; width: 100%; text-align: left; background: none; border: none; color: var(--text); padding: 7px 8px; border-radius: 8px; font-size: 0.88rem; cursor: pointer; font-family: 'Inter', sans-serif; }
    .lang-btn:hover { background: rgba(37,99,235,0.1); }
    .lang-btn.original { border-top: 1px solid var(--border); margin-top: 6px; padding-top: 10px; font-weight: 600; color: var(--accent); }
    .lang-status { font-size: 0.75rem; color: var(--accent); margin-top: 8px; min-height: 1em; }
    .lang-chip { background: rgba(255,255,255,0.18); }

    /* Quiz / flashcards */
    .quiz-section { max-width: 900px; margin: 0 auto 40px; padding: 0 20px; }
    .quiz-section h2 { border: none; padding-bottom: 0; }
    .quiz-hint { color: var(--text-muted); font-size: 0.9rem; margin-bottom: 18px; }
    .quiz-hint a { color: var(--accent); }
    .quiz-card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 18px 22px; margin-bottom: 12px; cursor: pointer; }
    .quiz-card .q { font-weight: 600; }
    .quiz-card .a { margin-top: 0; max-height: 0; overflow: hidden; color: var(--text-muted); transition: margin-top .15s, max-height .15s; }
    .quiz-card.revealed .a { margin-top: 12px; padding-top: 12px; border-top: 1px dashed var(--border); max-height: 500px; }
    .quiz-card .tap-hint { font-size: 0.75rem; color: var(--text-muted); margin-top: 6px; }
    __PYGMENTS_CSS__
"""


def build_html_report(video_dir, video_title, channel, source_url, analysis,
                      screenshot_map, flowchart_images, video_meta=None,
                      cropped_visuals_map=None, vision_code_blocks=None,
                      local_code_blocks=None, key_takeaways=None,
                      quiz_cards=None, output_name='report'):
    cropped_visuals_map = cropped_visuals_map or {}
    vision_code_blocks = vision_code_blocks or {}
    local_code_blocks = local_code_blocks or {}
    video_meta = video_meta or {}
    esc = html_escape_module.escape

    html_path = video_dir / f'{output_name}.html'
    sections = analysis.get('sections', [])

    safe_title = esc(video_title or '')
    safe_channel = esc(channel or '')

    # ── Pygments theme (best-effort; degrades gracefully if unavailable) ──
    pygments_css = ''
    try:
        from pygments.formatters import HtmlFormatter
        pygments_css = HtmlFormatter(style='friendly').get_style_defs('.codehilite')
    except Exception as e:
        _log.debug("Pygments syntax-highlight CSS unavailable: %s", e)

    # ── Header / hero metadata ──
    # NOTE: view count (and any similar constantly-changing stat like likes)
    # is intentionally left out of the header — only stable facts are shown.
    thumbnail = video_meta.get('thumbnail')
    channel_avatar = video_meta.get('channel_avatar')
    duration_str = _fmt_duration(video_meta.get('duration'))
    date_str = _fmt_upload_date(video_meta.get('upload_date'))

    meta_chips = []
    if duration_str:
        meta_chips.append(f'<span class="chip">⏱️ {duration_str}</span>')
    if date_str:
        meta_chips.append(f'<span class="chip">📅 {date_str}</span>')
    meta_chips.append(f'<span class="chip">📚 {len(sections)} Sections</span>')
    detected_language = (analysis.get('detected_language') or '').strip()
    if detected_language:
        meta_chips.append(f'<span class="chip">🗣️ {esc(detected_language)}</span>')

    if channel_avatar:
        avatar_html = f'<img class="channel-avatar" src="{esc(channel_avatar)}" alt="{safe_channel}">'
    else:
        avatar_html = f'<div class="channel-avatar channel-avatar-fallback">{(safe_channel[:1] or "?").upper()}</div>'

    hero_bg = f'background-image: linear-gradient(180deg, rgba(15,23,42,0.55), rgba(15,23,42,0.88)), url("{esc(thumbnail)}");' if thumbnail else 'background: linear-gradient(135deg, var(--accent), #8b5cf6);'

    css = _HTML_CSS_TEMPLATE.replace('__HERO_BG__', hero_bg).replace('__PYGMENTS_CSS__', pygments_css)

    toc_html = ''.join(
        f'<a href="#sec-{idx}" class="toc-chip">{sec["start_seconds"] // 60:02d}:{sec["start_seconds"] % 60:02d} · {esc(sec.get("heading") or "Section")}</a>'
        for idx, sec in enumerate(sections)
    )

    # Sidebar drawer — always-available jump-to-section nav (fixes having
    # to scroll back to the top TOC on long videos)
    drawer_links = ''.join(
        f'<a href="#sec-{idx}" onclick="closeDrawer()">'
        f'<span class="dot{" extra" if sec.get("importance") == "extra" else ""}"></span>'
        f'{sec["start_seconds"] // 60:02d}:{sec["start_seconds"] % 60:02d} · {esc(sec.get("heading") or "Section")}'
        f'</a>'
        for idx, sec in enumerate(sections)
    )

    takeaways_html = ''
    if key_takeaways:
        items = ''.join(f'<li data-vne-takeaway="{i}">{esc(t)}</li>' for i, t in enumerate(key_takeaways))
        takeaways_html = f'''
    <div class="takeaways">
        <h3>🔑 Key Takeaways</h3>
        <ul>{items}</ul>
    </div>'''

    theme_init_script = '''<script>
(function(){
  var root = document.documentElement;
  var saved = null;
  try { saved = localStorage.getItem('vne-theme'); } catch(e) {}
  if (saved === 'dark' || saved === 'light') root.setAttribute('data-theme', saved);
})();
</script>'''

    # ── Language picker — lets the reader view this report translated into
    # ANY language on their own screen, regardless of what language the
    # notes were generated in (e.g. video/notes in English, reader wants
    # Marathi). Translates on demand using the reader's OWN Gemini API key
    # (same one used to generate this report) directly from the browser —
    # no server involved. The first view of a given language needs
    # internet + the key; the translated text is then cached on-device
    # (localStorage) so re-opening it later is instant and works offline.
    # (Previously this used Google's embedded website-translate widget,
    # which depends on an unofficial Google script that regularly fails
    # to load/inject on locally-opened HTML files — that's why translation
    # wasn't working. This version has no such dependency.)
    _LANG_OPTIONS = [
        ('hi', 'हिंदी Hindi', 'Hindi'), ('mr', 'मराठी Marathi', 'Marathi'),
        ('bn', 'বাংলা Bengali', 'Bengali'), ('ta', 'தமிழ் Tamil', 'Tamil'),
        ('te', 'తెలుగు Telugu', 'Telugu'), ('kn', 'ಕನ್ನಡ Kannada', 'Kannada'),
        ('ml', 'മലയാളം Malayalam', 'Malayalam'), ('gu', 'ગુજરાતી Gujarati', 'Gujarati'),
        ('pa', 'ਪੰਜਾਬੀ Punjabi', 'Punjabi'), ('ur', 'اردو Urdu', 'Urdu'),
        ('or', 'ଓଡ଼ିଆ Odia', 'Odia'), ('as', 'অসমীয়া Assamese', 'Assamese'),
        ('ne', 'नेपाली Nepali', 'Nepali'), ('sa', 'संस्कृत Sanskrit', 'Sanskrit'),
        ('en', 'English', 'English'), ('es', 'Español', 'Spanish'),
        ('fr', 'Français', 'French'), ('de', 'Deutsch', 'German'),
        ('pt', 'Português', 'Portuguese'), ('ru', 'Русский', 'Russian'),
        ('zh-CN', '中文 Chinese', 'Chinese (Simplified)'), ('ja', '日本語 Japanese', 'Japanese'),
        ('ko', '한국어 Korean', 'Korean'), ('ar', 'العربية Arabic', 'Arabic'),
        ('it', 'Italiano', 'Italian'), ('tr', 'Türkçe', 'Turkish'),
        ('vi', 'Tiếng Việt', 'Vietnamese'), ('id', 'Bahasa Indonesia', 'Indonesian'),
        ('th', 'ไทย Thai', 'Thai'),
    ]
    lang_buttons_html = ''.join(
        f'<button class="lang-btn" onclick="setReportLanguage(\'{code}\')">{esc(disp)}</button>'
        for code, disp, _eng in _LANG_OPTIONS
    )
    lang_name_map_json = json.dumps({code: eng for code, _disp, eng in _LANG_OPTIONS})

    # Raw (pre-render) source text — Markdown notes, not rendered HTML — so
    # Markdown syntax, code fences, and math delimiters survive the round
    # trip through Gemini intact instead of getting mangled.
    translation_source = {
        'sections': [
            {'heading': sec.get('heading') or 'Section', 'notes': sec.get('notes') or ''}
            for sec in sections
        ],
        'takeaways': key_takeaways or [],
        'quiz': [{'q': c.get('question', ''), 'a': c.get('answer', '')} for c in (quiz_cards or [])],
    }
    translation_source_json = json.dumps(translation_source, ensure_ascii=False).replace('</', '<\\/')
    report_id = uuid.uuid4().hex[:10]
    models_json = json.dumps(TEXT_MODEL_FALLBACKS)

    translate_prompt_text = (
        "You translate structured educational notes for a student reading app.\n"
        "You are given a JSON object and a target language. Translate ONLY the "
        "natural-language text into the target language, following these rules exactly:\n\n"
        "1. Preserve the JSON structure and all keys exactly - translate string VALUES only.\n"
        "2. Inside every 'notes' string: this is Markdown. Preserve ALL Markdown syntax "
        "characters exactly as-is (**, *, backtick, >, #, -, |, blank lines) - translate "
        "only the words, never the formatting symbols.\n"
        "3. NEVER translate or alter text inside triple-backtick code fences, inline code "
        "spans, or LaTeX math delimited by $$...$$, \\(...\\), or \\[...\\] - copy those "
        "spans character-for-character, unchanged, including variable names and numbers.\n"
        "4. Use the correct native script for the target language (e.g. Devanagari for "
        "Hindi, Bengali script for Bengali, Tamil script for Tamil) - never write the "
        "target language using Roman/Latin transliteration.\n"
        "5. Keep proper nouns, brand names, URLs, and numbers unchanged unless the "
        "language has a standard localized form.\n"
        "6. Well-established technical/English terms may stay in Latin script if a "
        "native speaker of the target language would naturally keep them that way - but "
        "every other word must be in the target language's own script.\n"
        "7. Output STRICT JSON only, in the exact same shape as the input, no markdown "
        "fences, no commentary, no extra keys."
    )
    translate_prompt_json = json.dumps(translate_prompt_text)

    _LANG_PANEL_TEMPLATE = r'''
    <div class="lang-panel" id="lang-panel">
        <h4>🌐 View this report in...</h4>
        <div class="lang-note">Translated using your own Gemini API key (same one you used to generate this report), right in your browser. First view of a language needs internet + your key; after that it's cached on this device.</div>
        __LANG_BUTTONS__
        <button class="lang-btn original" onclick="setReportLanguage('')">↺ Original</button>
        <div class="lang-status" id="lang-status"></div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/16.3.0/lib/marked.umd.min.js"></script>
    <script id="vne-source-data" type="application/json">__SOURCE_JSON__</script>
    <script>
    var VNE_REPORT_ID = "__REPORT_ID__";
    var VNE_LANG_NAMES = __LANG_NAME_MAP_JSON__;
    var VNE_MODELS = __MODELS_JSON__;
    var VNE_TRANSLATE_PROMPT = __PROMPT_JSON__;
    var VNE_ORIGINAL = JSON.parse(document.getElementById('vne-source-data').textContent);
    var VNE_ORIGINAL_HTML = { headings: {}, notes: {} };
    var VNE_MATH_RE = /(\$\$[\s\S]+?\$\$|\\\([\s\S]+?\\\)|\\\[[\s\S]+?\\\])/g;

    function toggleLangPanel(){ document.getElementById('lang-panel').classList.toggle('open'); }

    function vneGetApiKey(){
        var k = null;
        try { k = localStorage.getItem('vne-gemini-key'); } catch(e) {}
        if (!k) {
            k = prompt('Paste your Gemini API key (same one used to generate this report). Stored only on this device, sent only to Google.');
            if (k) { k = k.trim(); try { localStorage.setItem('vne-gemini-key', k); } catch(e) {} }
        }
        return k;
    }

    function vneCacheKey(code){ return 'vne-tr-' + VNE_REPORT_ID + '-' + code; }

    function vneProtectMath(text){
        var placeholders = [];
        var out = (text || '').replace(VNE_MATH_RE, function(m){
            placeholders.push(m);
            return 'MATHPLACEHOLDERZ' + (placeholders.length - 1) + 'ZEND';
        });
        return [out, placeholders];
    }
    function vneRestoreMath(html, placeholders){
        for (var i = 0; i < placeholders.length; i++) {
            html = html.split('MATHPLACEHOLDERZ' + i + 'ZEND').join(placeholders[i]);
        }
        return html;
    }
    function vneRenderNotes(mdText){
        var pair = vneProtectMath(mdText);
        var html = marked.parse(pair[0], { gfm: true, breaks: true });
        return vneRestoreMath(html, pair[1]);
    }

    async function vneCallGemini(payload, langName, apiKey){
        var promptText = VNE_TRANSLATE_PROMPT + '\n\nTarget language: ' + langName +
            '\n\nJSON to translate:\n' + JSON.stringify(payload);
        var body = {
            contents: [{ parts: [{ text: promptText }] }],
            generationConfig: { temperature: 0, maxOutputTokens: 32768, response_mime_type: 'application/json' }
        };
        var lastErr = null;
        for (var i = 0; i < VNE_MODELS.length; i++) {
            var model = VNE_MODELS[i];
            for (var attempt = 0; attempt < 2; attempt++) {
                var resp;
                try {
                    resp = await fetch('https://generativelanguage.googleapis.com/v1beta/models/' + model + ':generateContent', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json', 'x-goog-api-key': apiKey },
                        body: JSON.stringify(body)
                    });
                } catch (netErr) { lastErr = netErr; break; }
                if (resp.status === 401 || resp.status === 403) {
                    throw new Error('API key invalid or unauthorized.');
                }
                if (resp.status === 429 || resp.status === 503) {
                    await new Promise(function(r){ setTimeout(r, 3000); });
                    continue;
                }
                if (!resp.ok) { lastErr = new Error('API error ' + resp.status); break; }
                var data = await resp.json();
                var candidate = (data.candidates || [])[0] || {};
                var parts = (candidate.content || {}).parts || [];
                var text = parts[0] && parts[0].text;
                if (!text) { lastErr = new Error('Empty response from model.'); break; }
                var cleaned = text.trim().replace(/^```(json)?/, '').replace(/```$/, '').trim();
                try { return JSON.parse(cleaned); }
                catch (e) {
                    var repaired = cleaned.replace(/\\(?!["\\\/bfnrtu])/g, '\\\\');
                    try { return JSON.parse(repaired); }
                    catch (e2) { lastErr = e2; break; }
                }
            }
        }
        throw lastErr || new Error('Translation failed.');
    }

    function vneApplyTakeawaysQuiz(src){
        document.querySelectorAll('[data-vne-takeaway]').forEach(function(el){
            var i = +el.dataset.vneTakeaway;
            if (src.takeaways && src.takeaways[i] != null) el.textContent = src.takeaways[i];
        });
        document.querySelectorAll('[data-vne-quiz-q]').forEach(function(el){
            var i = +el.dataset.vneQuizQ;
            if (src.quiz && src.quiz[i]) el.textContent = 'Q' + (i + 1) + '. ' + src.quiz[i].q;
        });
        document.querySelectorAll('[data-vne-quiz-a]').forEach(function(el){
            var i = +el.dataset.vneQuizA;
            if (src.quiz && src.quiz[i]) el.textContent = src.quiz[i].a;
        });
    }

    async function setReportLanguage(code){
        var panel = document.getElementById('lang-panel');
        if (panel) panel.classList.remove('open');
        var status = document.getElementById('lang-status');
        if (status) status.textContent = '';

        if (!code) {
            document.querySelectorAll('[data-vne-heading]').forEach(function(el){
                var v = VNE_ORIGINAL_HTML.headings[el.dataset.vneHeading];
                if (v != null) el.textContent = v;
            });
            document.querySelectorAll('[data-vne-notes]').forEach(function(el){
                var v = VNE_ORIGINAL_HTML.notes[el.dataset.vneNotes];
                if (v != null) el.innerHTML = v;
            });
            vneApplyTakeawaysQuiz(VNE_ORIGINAL);
            return;
        }

        var cacheKey = vneCacheKey(code);
        var translated = null;
        try { var raw = localStorage.getItem(cacheKey); if (raw) translated = JSON.parse(raw); } catch(e) {}

        if (!translated) {
            var apiKey = vneGetApiKey();
            if (!apiKey) return;
            if (status) status.textContent = 'Translating with Gemini…';
            try {
                translated = await vneCallGemini(VNE_ORIGINAL, VNE_LANG_NAMES[code] || code, apiKey);
                try { localStorage.setItem(cacheKey, JSON.stringify(translated)); } catch(e) {}
            } catch (err) {
                if (status) status.textContent = '';
                alert('Translation failed: ' + err.message + '\n\nCheck your Gemini API key and internet connection, then try again.');
                return;
            }
        }
        if (status) status.textContent = '';

        document.querySelectorAll('[data-vne-heading]').forEach(function(el){
            var t = translated.sections && translated.sections[+el.dataset.vneHeading];
            if (t && t.heading) el.textContent = t.heading;
        });
        document.querySelectorAll('[data-vne-notes]').forEach(function(el){
            var t = translated.sections && translated.sections[+el.dataset.vneNotes];
            if (t) el.innerHTML = vneRenderNotes(t.notes || '');
        });
        vneApplyTakeawaysQuiz(translated);
        if (window.MathJax && MathJax.typesetPromise) MathJax.typesetPromise();
    }
    </script>'''

    lang_panel_html = (_LANG_PANEL_TEMPLATE
        .replace('__LANG_BUTTONS__', lang_buttons_html)
        .replace('__SOURCE_JSON__', translation_source_json)
        .replace('__REPORT_ID__', report_id)
        .replace('__LANG_NAME_MAP_JSON__', lang_name_map_json)
        .replace('__MODELS_JSON__', models_json)
        .replace('__PROMPT_JSON__', translate_prompt_json))

    page = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{safe_title}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
{theme_init_script}
<script>
window.MathJax = {{ tex: {{ inlineMath: [["\\\\(", "\\\\)"]], displayMath: [["$$", "$$"]] }} }};
</script>
<script id="MathJax-script" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.js"></script>
<style>{css}</style>
</head>
<body>
    <div class="drawer-backdrop" id="drawer-backdrop" onclick="closeDrawer()"></div>
    <div class="side-drawer" id="side-drawer">
        <h3>📚 Jump to section</h3>
        {drawer_links}
    </div>
    {lang_panel_html}
    <div class="fab-group">
        <button class="fab" onclick="toggleDrawer()" title="Sections" aria-label="Sections">☰</button>
        <button class="fab" onclick="toggleLangPanel()" title="View in another language" aria-label="View in another language">🌐</button>
        <button class="fab" onclick="toggleTheme()" title="Toggle dark mode" aria-label="Toggle dark mode">🌓</button>
    </div>
    <div class="hero">
        <div class="channel-row"><div class="channel-badge">{avatar_html}<span class="channel-name">{safe_channel}</span></div></div>
        <h1>{safe_title}</h1>
        <div class="meta-chips">{''.join(meta_chips)}</div>
        <a href="{source_url}" class="watch-btn" target="_blank" rel="noopener">▶ Watch Original Video</a>
    </div>
    <div class="toc">{toc_html}</div>
    {takeaways_html}
    <div class="container">
'''

    # Overview flowchart — generated the same way as the PDF/Markdown
    # reports' "Video Overview" section, but this was previously never
    # written into the HTML output at all (only per-section flowcharts
    # were), so a video with only an overview flowchart showed none.
    ofc = flowchart_images.get('overview')
    if ofc and os.path.exists(ofc):
        ofc_b64 = get_base64_image(ofc)
        page += (
            '<div class="card">'
            '<h2>Video Overview</h2>'
            '<div class="screenshot-wrapper">'
            f'<img src="data:image/png;base64,{ofc_b64}" alt="Content flow overview">'
            '</div></div>'
        )

    for idx, sec in enumerate(sections):
        m, s = divmod(sec['start_seconds'], 60)
        heading = esc(sec['heading']) if sec['heading'] else "Section"
        is_extra = sec.get('importance') == 'extra'
        jump_url = esc(_yt_ts_link(source_url, sec['start_seconds']))

        notes_protected, placeholders = _protect_math(sec['notes'])
        notes_html = markdown.markdown(
            notes_protected,
            extensions=['tables', 'fenced_code', 'codehilite', 'nl2br', 'sane_lists'],
            extension_configs={'codehilite': {'guess_lang': False}},
        )
        notes_html = _restore_math(notes_html, placeholders)

        page += f'''
    <div class="card" id="sec-{idx}">
        <div class="badges-row">
            <div class="timestamp-badge">{m:02d}:{s:02d}</div>
            {'<span class="extra-badge">⚡ Extra / Tangent</span>' if is_extra else ''}
            <a class="jump-link" href="{jump_url}" target="_blank" rel="noopener">▶ Watch this part</a>
        </div>
        <h2 data-vne-heading="{idx}">{heading}</h2>
        <div class="content" data-vne-notes="{idx}">{notes_html}</div>
'''
        # Structured table (kept separate from prose per the extraction prompt)
        page += _render_table_html(sec.get('table'))

        # Code dictated/mentioned in the transcript
        for cb in sec.get('code_blocks', []):
            if cb.get('description'):
                page += f'<div class="code-label">{esc(cb["description"])}</div>'
            page += _render_code_html(cb.get('code', ''), cb.get('language', ''))

        # Screenshots + anything extracted from them
        for ss in sec.get('screenshot_timestamps', []):
            ts = ss['seconds']
            full_img = screenshot_map.get(ts)
            if not (full_img and os.path.exists(full_img)):
                continue

            b64 = get_base64_image(full_img)
            caption = ss.get('vision_caption', '')
            shot_url = esc(_yt_ts_link(source_url, ts))
            page += f'<div class="screenshot-wrapper"><img src="data:image/png;base64,{b64}" alt="Screenshot">'
            if caption:
                page += (f'<div style="text-align: center; font-size: 1.05rem; color: var(--text-muted); '
                          f'margin-top: 12px; font-weight: 500; font-style: italic;">{esc(caption)}</div>')
            page += (f'<div class="shot-link"><a href="{shot_url}" target="_blank" rel="noopener">'
                      f'▶ Jump to {ts // 60:02d}:{ts % 60:02d} in video</a></div>')
            page += '</div>'

            # Zoomed-in crops so cluttered/handwritten boards are actually legible
            for crop in cropped_visuals_map.get(ts, []):
                cp = crop.get('path')
                if cp and os.path.exists(cp):
                    cb64 = get_base64_image(cp)
                    cdesc = esc(crop.get('description') or 'Zoomed detail')
                    page += (f'<div class="screenshot-wrapper"><img src="data:image/png;base64,{cb64}" alt="{cdesc}">'
                              f'<div style="text-align:center; font-size:0.92rem; color:var(--text-muted); margin-top:8px;">🔍 {cdesc}</div></div>')

            # Code extracted from the screen itself (Vision or local OCR)
            code_from_screen = vision_code_blocks.get(ts) or local_code_blocks.get(ts)
            if code_from_screen and code_from_screen.get('code'):
                page += f'<div class="code-label">Code on screen at {ts // 60:02d}:{ts % 60:02d}</div>'
                page += _render_code_html(code_from_screen['code'], code_from_screen.get('language', ''))

        # Section flowchart
        fp = flowchart_images.get(idx)
        if fp and os.path.exists(fp):
            b64 = get_base64_image(fp)
            page += (f'<div class="screenshot-wrapper"><h3 style="margin-top: 0; color: var(--text-muted);">'
                      f'Process Flow</h3><img src="data:image/png;base64,{b64}"></div>')
        page += "</div>"

    page += "</div>"  # close .container

    # ── Revision Quiz (optional, --quiz) ──
    if quiz_cards:
        quiz_html = ''
        for i, card in enumerate(quiz_cards):
            quiz_html += f'''
        <div class="quiz-card" onclick="this.classList.toggle('revealed')">
            <div class="q" data-vne-quiz-q="{i}">Q{i + 1}. {esc(card['question'])}</div>
            <div class="a" data-vne-quiz-a="{i}">{esc(card['answer'])}</div>
            <div class="tap-hint">tap to {{ }} answer</div>
        </div>'''.replace('{ }', 'reveal/hide')
        page += f'''
    <div class="quiz-section">
        <h2>🧠 Revision Quiz</h2>
        <div class="quiz-hint">{len(quiz_cards)} question(s) · tap a card to reveal the answer · also saved as <a href="flashcards.txt">flashcards.txt</a> for Anki import</div>
        {quiz_html}
    </div>'''

    page += '''
<script>
function toggleDrawer(){
    document.getElementById('side-drawer').classList.toggle('open');
    document.getElementById('drawer-backdrop').classList.toggle('open');
}
function closeDrawer(){
    document.getElementById('side-drawer').classList.remove('open');
    document.getElementById('drawer-backdrop').classList.remove('open');
}
function toggleTheme(){
    var root = document.documentElement;
    var current = root.getAttribute('data-theme');
    if (!current) current = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    var next = current === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    try { localStorage.setItem('vne-theme', next); } catch(e) {}
}
if (typeof VNE_ORIGINAL_HTML !== 'undefined') {
    document.querySelectorAll('[data-vne-heading]').forEach(function(el){
        VNE_ORIGINAL_HTML.headings[el.dataset.vneHeading] = el.textContent;
    });
    document.querySelectorAll('[data-vne-notes]').forEach(function(el){
        VNE_ORIGINAL_HTML.notes[el.dataset.vneNotes] = el.innerHTML;
    });
}
</script>
</body></html>'''
    _safe_write_text(html_path, page)
    return html_path

def build_markdown_report(video_dir, video_title, channel, source_url,
                          analysis, screenshot_map, cropped_visuals_map,
                          vision_code_blocks, local_code_blocks,
                          flowchart_images, key_takeaways=None, quiz_cards=None):
    """Build Markdown report — great for code copy-paste."""
    md_path = video_dir / 'report.md'
    vtype = analysis.get('video_type', 'general')
    sections = analysis.get('sections', [])

    type_labels = {
        'coding': 'Coding Tutorial', 'slides': 'Presentation',
        'lecture': 'Lecture', 'tutorial': 'Tutorial',
        'trading': 'Trading/Finance', 'whiteboard': 'Whiteboard',
        'general': 'General',
    }

    L = []
    L.append(f"# {video_title}\n")
    if channel:
        L.append(f"**Channel:** {channel}  ")
    L.append(f"**URL:** [{source_url}]({source_url})  ")
    L.append(f"**Type:** {type_labels.get(vtype, vtype)}  ")
    L.append(f"**Sections:** {len(sections)}  ")
    L.append("")

    if key_takeaways:
        L.append("## 🔑 Key Takeaways\n")
        for t in key_takeaways:
            L.append(f"- {t}")
        L.append("")

    # Overview flowchart
    ofc = flowchart_images.get('overview')
    if ofc and os.path.exists(ofc):
        L.append("## 📋 Video Overview\n")
        L.append(f"![Overview Flowchart]({os.path.basename(ofc)})\n")

    ofc_mermaid = analysis.get('overview_flowchart')
    if ofc_mermaid:
        L.append("<details><summary>📊 Flowchart source (Mermaid)</summary>\n")
        L.append("```mermaid")
        L.append(ofc_mermaid)
        L.append("```\n</details>\n")

    L.append("---\n")

    for idx, sec in enumerate(sections):
        m, s = divmod(sec['start_seconds'], 60)
        heading = f"## [{m:02d}:{s:02d}]"
        if sec['heading']:
            heading += f" {sec['heading']}"
        if sec.get('importance') == 'extra':
            heading += " *(extra)*"
        L.append(f"{heading}\n")
        L.append(f"[▶ Watch this part]({_yt_ts_link(source_url, sec['start_seconds'])})\n")

        # Notes
        L.append(sec['notes'])
        L.append("")

        # Code blocks (transcript)
        for cb in sec.get('code_blocks', []):
            if cb.get('description'):
                L.append(f"> {cb['description']}\n")
            L.append(f"```{cb.get('language', '')}")
            L.append(cb['code'])
            L.append("```\n")

        # Table
        table = sec.get('table')
        if table and table.get('headers') and table.get('rows'):
            hdr = table['headers']
            L.append("| " + " | ".join(str(h) for h in hdr) + " |")
            L.append("| " + " | ".join("---" for _ in hdr) + " |")
            for row in table['rows']:
                L.append("| " + " | ".join(str(c) for c in row) + " |")
            L.append("")

        # Visual content
        for ss in sec.get('screenshot_timestamps', []):
            ts = ss['seconds']

            # Vision code
            vcb = vision_code_blocks.get(ts)
            if vcb and vcb.get('code'):
                L.append(f"**Code from screen [{ts // 60:02d}:{ts % 60:02d}]:**\n")
                L.append(f"```{vcb.get('language', '')}")
                L.append(vcb['code'])
                L.append("```\n")

            # Local OCR code
            if not vcb:
                lcb = local_code_blocks.get(ts)
                if lcb and lcb.get('code'):
                    L.append(f"**Code detected [{ts // 60:02d}:{ts % 60:02d}]:**\n")
                    L.append(f"```{lcb.get('language', '')}")
                    L.append(lcb['code'])
                    L.append("```\n")

            # Cropped visuals
            crops = cropped_visuals_map.get(ts, [])
            for ci in crops:
                cp = ci.get('path')
                cd = ci.get('description', ci.get('type', 'Visual'))
                if cp and os.path.exists(cp):
                    L.append(f"**{cd}:**\n")
                    L.append(f"![{cd}]({os.path.basename(cp)})\n")

            # Full screenshot — ALWAYS included
            full = screenshot_map.get(ts)
            if full and os.path.exists(full):
                desc = ss.get('description',
                              f'Screenshot at {ts // 60:02d}:{ts % 60:02d}')
                L.append(f"![{desc}]({os.path.basename(full)})")
                L.append(f"[▶ Jump to {ts // 60:02d}:{ts % 60:02d} in video]({_yt_ts_link(source_url, ts)})\n")

        # Section flowchart
        fp = flowchart_images.get(idx)
        if fp and os.path.exists(fp):
            L.append(f"**Process Flow:**\n")
            L.append(f"![Flowchart]({os.path.basename(fp)})\n")

        fc_mermaid = sec.get('flowchart')
        if fc_mermaid:
            L.append("<details><summary>📊 Flowchart source</summary>\n")
            L.append("```mermaid")
            L.append(fc_mermaid)
            L.append("```\n</details>\n")

        L.append("---\n")

    if quiz_cards:
        L.append("## 🧠 Revision Quiz\n")
        L.append("*Also saved as `flashcards.txt` — Anki-importable.*\n")
        for i, card in enumerate(quiz_cards, 1):
            L.append(f"**Q{i}.** {card['question']}  ")
            L.append(f"<details><summary>Show answer</summary>{card['answer']}</details>\n")

    _safe_write_text(md_path, '\n'.join(L))
    return md_path


# ═══════════════════════════════════════════════════════════════════
# PDF MERGING (master index via nested bookmarks)
# ═══════════════════════════════════════════════════════════════════

def merge_pdfs(video_records, output_path):
    """Merge each video's report.pdf into one combined_report.pdf.
    video_records: list of dicts with at least {'title', 'pdf'}.

    Each video gets a top-level bookmark (named after its title) in the
    combined PDF's outline/bookmark panel, and — because build_pdf_report()
    stamps its own per-section bookmarks into every per-video PDF —
    pypdf's import_outline=True carries those in as nested children. Open
    the bookmarks panel in any PDF viewer and you get a full course index:
    video -> section, click to jump straight to that page."""
    writer = PdfWriter()
    for rec in video_records:
        p = rec.get('pdf')
        if not p or not os.path.exists(p):
            continue
        try:
            writer.append(str(p), outline_item=rec.get('title') or os.path.basename(p),
                          import_outline=True)
        except TypeError:
            # Older pypdf without import_outline kwarg — fall back to plain append
            writer.append(str(p), outline_item=rec.get('title') or os.path.basename(p))
        except Exception as e:
            print(f"    [!] Could not merge {p}: {e}")
    if len(writer.pages) == 0:
        return None
    with open(output_path, 'wb') as f:
        writer.write(f)
    return output_path


def build_playlist_index(out_dir, playlist_title, video_records):
    """Build a single master-index HTML page for the whole
    playlist/course — one place with every video's own section TOC,
    linking straight into each video's report.html#sec-N."""
    esc = html_escape_module.escape
    out_dir = Path(out_dir)
    title = playlist_title or "Course Index"

    body = []
    for rec in video_records:
        vid_title = esc(rec.get('title') or rec.get('video_id', ''))
        rel_html = None
        if rec.get('html'):
            try:
                rel_html = os.path.relpath(rec['html'], out_dir)
            except ValueError:
                rel_html = rec['html']
        sections = rec.get('sections') or []
        sec_links = []
        for i, sec in enumerate(sections):
            m, s = divmod(int(sec.get('start_seconds', 0)), 60)
            heading = esc(sec.get('heading') or 'Section')
            tag = ' <span class="extra-tag">extra</span>' if sec.get('importance') == 'extra' else ''
            href = f"{rel_html}#sec-{i}" if rel_html else '#'
            sec_links.append(
                f'<li><a href="{href}">{m:02d}:{s:02d} · {heading}</a>{tag}</li>')
        video_href = rel_html or '#'
        body.append(f'''
    <div class="video-card">
      <h2><a href="{video_href}">{vid_title}</a></h2>
      <ul class="sec-list">{''.join(sec_links)}</ul>
    </div>''')

    page = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{esc(title)}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  :root {{ --bg:#f8fafc; --card:#fff; --text:#1e293b; --muted:#64748b; --accent:#2563eb; --border:#e2e8f0; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family:'Inter',sans-serif; background:var(--bg); color:var(--text); margin:0; padding:40px 20px; }}
  .wrap {{ max-width: 820px; margin: 0 auto; }}
  h1 {{ font-size: 1.9rem; font-weight: 800; margin-bottom: 6px; }}
  .sub {{ color: var(--muted); margin-bottom: 30px; }}
  .video-card {{ background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 22px 28px; margin-bottom: 18px; }}
  .video-card h2 {{ margin: 0 0 10px; font-size: 1.15rem; }}
  .video-card h2 a {{ color: var(--text); text-decoration: none; }}
  .video-card h2 a:hover {{ color: var(--accent); }}
  .sec-list {{ list-style: none; margin: 0; padding: 0; display: grid; gap: 6px; }}
  .sec-list li {{ font-size: 0.92rem; }}
  .sec-list a {{ color: var(--muted); text-decoration: none; }}
  .sec-list a:hover {{ color: var(--accent); }}
  .extra-tag {{ font-size: 0.72rem; color: #b45309; background: #fef3c7; padding: 1px 7px; border-radius: 8px; margin-left: 6px; }}
</style>
</head>
<body>
  <div class="wrap">
    <h1>{esc(title)}</h1>
    <div class="sub">{len(video_records)} video(s) · master index · <a href="search.html" style="color: var(--accent);">🔎 search all notes</a></div>
    {''.join(body)}
  </div>
</body>
</html>'''

    index_path = out_dir / 'index.html'
    _safe_write_text(index_path, page)
    return index_path


# ═══════════════════════════════════════════════════════════════════
# LIBRARY SEARCH — search across every video ever processed
# ═══════════════════════════════════════════════════════════════════

def scan_library(scan_dir):
    """Walk `scan_dir` for every processed video (anywhere a video_info.json
    sits) and return a flat list of section-level records — same shape
    whether the source is this run's out_dir or the whole global cache.
    Used by both build_search_index() and the --ask Q&A mode."""
    scan_dir = Path(scan_dir)
    records = []
    if not scan_dir.exists():
        return records

    for info_path in scan_dir.rglob('video_info.json'):
        video_dir = info_path.parent
        try:
            with open(info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
        except Exception:
            continue

        sections_path = video_dir / 'sections.json'
        if not sections_path.exists():
            continue
        try:
            with open(sections_path, 'r', encoding='utf-8') as f:
                analysis = json.load(f)
        except Exception:
            continue

        html_path = video_dir / 'report.html'
        rel_html = None
        if html_path.exists():
            try:
                rel_html = os.path.relpath(html_path, scan_dir)
            except ValueError:
                rel_html = str(html_path)

        pdf_path = video_dir / 'report.pdf'
        rel_pdf = None
        if pdf_path.exists():
            try:
                rel_pdf = os.path.relpath(pdf_path, scan_dir)
            except ValueError:
                rel_pdf = str(pdf_path)

        md_path = video_dir / 'report.md'
        rel_md = None
        if md_path.exists():
            try:
                rel_md = os.path.relpath(md_path, scan_dir)
            except ValueError:
                rel_md = str(md_path)

        for idx, sec in enumerate(analysis.get('sections', [])):
            records.append({
                'video_id': info.get('video_id', video_dir.name),
                'title': info.get('title', video_dir.name),
                'channel': info.get('channel', ''),
                'source_url': info.get('source_url',
                                       f"https://www.youtube.com/watch?v={video_dir.name}"),
                'start_seconds': sec.get('start_seconds', 0),
                'heading': sec.get('heading', ''),
                'notes': sec.get('notes', ''),
                'importance': sec.get('importance', 'must_know'),
                'section_index': idx,
                'rel_html': rel_html,
                'rel_pdf': rel_pdf,
                'rel_md': rel_md,
                'video_dir': str(video_dir),
            })
    return records


def build_search_index(scan_dir, out_path, title="Search My Video Notes"):
    """Build one self-contained search.html over every video found under
    `scan_dir`. The whole dataset is inlined as JSON so client-side JS
    search works by just double-clicking the file — no server needed."""
    records = scan_library(scan_dir)
    esc = html_escape_module.escape

    payload = [{
        'v': r['title'], 'c': r['channel'], 'h': r['heading'],
        'n': (r['notes'] or '')[:400], 'i': r['importance'],
        's': r['start_seconds'], 'url': r['source_url'],
        'html': r['rel_html'], 'idx': r['section_index'],
    } for r in records]
    data_json = json.dumps(payload, ensure_ascii=False)

    page = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{esc(title)}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  :root {{ --bg:#f8fafc; --card:#fff; --text:#1e293b; --muted:#64748b; --accent:#2563eb; --border:#e2e8f0; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family:'Inter',sans-serif; background:var(--bg); color:var(--text); margin:0; padding:40px 20px; }}
  .wrap {{ max-width: 820px; margin: 0 auto; }}
  h1 {{ font-size: 1.7rem; font-weight: 800; margin-bottom: 6px; }}
  .sub {{ color: var(--muted); margin-bottom: 24px; }}
  input#q {{ width: 100%; padding: 14px 18px; font-size: 1.05rem; border: 1px solid var(--border); border-radius: 12px; margin-bottom: 24px; font-family: inherit; }}
  .result {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 16px 22px; margin-bottom: 12px; }}
  .result .meta {{ font-size: 0.8rem; color: var(--muted); margin-bottom: 4px; }}
  .result h3 {{ margin: 0 0 6px; font-size: 1.02rem; }}
  .result h3 a {{ color: var(--text); text-decoration: none; }}
  .result h3 a:hover {{ color: var(--accent); }}
  .result p {{ margin: 0; font-size: 0.92rem; color: #475569; }}
  .extra-tag {{ font-size: 0.7rem; color: #b45309; background: #fef3c7; padding: 1px 7px; border-radius: 8px; margin-left: 6px; }}
  .empty {{ color: var(--muted); padding: 20px 0; }}
  mark {{ background: #fde68a; padding: 0 2px; border-radius: 3px; }}
</style>
</head>
<body>
  <div class="wrap">
    <h1>🔎 Search My Video Notes</h1>
    <div class="sub" id="count"></div>
    <input id="q" type="text" placeholder="Type to search across every video's notes..." autofocus>
    <div id="results"></div>
  </div>
<script>
const DATA = {data_json};
document.getElementById('count').textContent = DATA.length + " saved section(s) across your library";
const q = document.getElementById('q');
const resultsEl = document.getElementById('results');
function escapeHtml(s) {{
  return s.replace(/[&<>"']/g, m => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}})[m]);
}}
function highlight(text, term) {{
  if (!term) return escapeHtml(text);
  const idx = text.toLowerCase().indexOf(term.toLowerCase());
  if (idx === -1) return escapeHtml(text);
  return escapeHtml(text.slice(0, idx)) + '<mark>' + escapeHtml(text.slice(idx, idx+term.length)) + '</mark>' + escapeHtml(text.slice(idx+term.length));
}}
function render(term) {{
  const t = term.trim().toLowerCase();
  if (!t) {{ resultsEl.innerHTML = '<div class="empty">Start typing to search your notes...</div>'; return; }}
  const matches = DATA.filter(d => (d.v + ' ' + d.h + ' ' + d.n).toLowerCase().includes(t)).slice(0, 60);
  if (!matches.length) {{ resultsEl.innerHTML = '<div class="empty">No matches.</div>'; return; }}
  resultsEl.innerHTML = matches.map(d => {{
    const m = Math.floor(d.s/60), s = d.s%60;
    const ts = String(m).padStart(2,'0') + ':' + String(s).padStart(2,'0');
    const link = d.html ? (d.html + '#sec-' + d.idx) : (d.url + '&t=' + d.s + 's');
    const tag = d.i === 'extra' ? '<span class="extra-tag">extra</span>' : '';
    return '<div class="result"><div class="meta">' + escapeHtml(d.v) + ' · ' + ts + tag + '</div>' +
           '<h3><a href="' + link + '" target="_blank">' + highlight(d.h || 'Section', term) + '</a></h3>' +
           '<p>' + highlight(d.n, term) + '</p></div>';
  }}).join('');
}}
q.addEventListener('input', () => render(q.value));
render('');
</script>
</body>
</html>'''

    _safe_write_text(out_path, page)
    return out_path


# ═══════════════════════════════════════════════════════════════════
# GROUNDED Q&A — ask questions across your saved notes, answers cite
# back to the exact video + timestamp, or say "not in your notes"
# ═══════════════════════════════════════════════════════════════════

_QA_STOPWORDS = set("""a an the is are was were be been being to of in on for and
or but with as at by from this that these those it its into over under about
than then so if not no do does did can could should would will shall may
might have has had i you he she we they them his her our your their what
which who whom how when where why""".split())


def _qa_tokenize(text):
    return [w for w in re.findall(r"[a-zA-Z0-9\u0900-\u097F]+", (text or '').lower())
            if w not in _QA_STOPWORDS and len(w) > 1]


def retrieve_relevant_sections(question, records, top_k=15):
    """Lightweight lexical retrieval — no embeddings/vector-DB dependency.
    Scores every saved section by keyword overlap with the question,
    weighting heading matches higher than body-text matches. This is a
    reasonable match quality for a personal library, and --ask always shows
    which sections it used so you can judge relevance yourself."""
    q_words = _qa_tokenize(question)
    if not q_words:
        return []
    scored = []
    for r in records:
        heading_words = _qa_tokenize(r.get('heading', ''))
        notes_words = _qa_tokenize(r.get('notes', ''))
        score = sum(3 if w in heading_words else notes_words.count(w) for w in q_words)
        if score > 0:
            scored.append((score, r))
    scored.sort(key=lambda x: -x[0])
    return [r for _, r in scored[:top_k]]


def answer_question(question, records, api_key, budget, top_k=15):
    """Grounded Q&A over the whole library: retrieve the most relevant saved
    sections, ask Gemini to answer ONLY from them with citations back to
    video+timestamp, or say the notes don't cover it rather than guessing.
    Returns (found: bool, answer: str, sources: list of section records)."""
    relevant = retrieve_relevant_sections(question, records, top_k=top_k)
    if not relevant:
        return False, "", []

    blocks = []
    for i, r in enumerate(relevant, 1):
        m, s = divmod(int(r.get('start_seconds', 0)), 60)
        blocks.append(f'[{i}] Video: "{r.get("title", "")}" @ {m:02d}:{s:02d}\n'
                      f'Heading: {r.get("heading", "")}\n'
                      f'Notes: {(r.get("notes") or "")[:1500]}\n')
    prompt = f"QUESTION: {question}\n\n" + "\n---\n".join(blocks)
    contents = [{"role": "user", "parts": [{"text": prompt}]}]

    try:
        parsed = _call_gemini_with_fallback(api_key, TEXT_MODEL_FALLBACKS, contents,
                                            system_prompt=QA_PROMPT,
                                            max_tokens=2048, budget=budget)
    except RuntimeError as e:
        return False, f"(Q&A failed: {e})", []

    found = bool(parsed.get('found')) if isinstance(parsed, dict) else False
    answer = str(parsed.get('answer', '')).strip() if isinstance(parsed, dict) else ''
    used_idx = parsed.get('used_excerpts', []) if isinstance(parsed, dict) else []

    sources = []
    for i in used_idx:
        try:
            idx = int(i) - 1
            if 0 <= idx < len(relevant):
                sources.append(relevant[idx])
        except (TypeError, ValueError):
            continue
    if not sources and found:
        sources = relevant[:3]
    return found, answer, sources


# ═══════════════════════════════════════════════════════════════════
# EXPORTS — runnable code files & Anki-importable flashcards
# ═══════════════════════════════════════════════════════════════════

_LANG_EXT = {
    'python': 'py', 'py': 'py', 'javascript': 'js', 'js': 'js',
    'typescript': 'ts', 'ts': 'ts', 'java': 'java', 'c': 'c', 'cpp': 'cpp',
    'c++': 'cpp', 'csharp': 'cs', 'c#': 'cs', 'go': 'go', 'golang': 'go',
    'rust': 'rs', 'ruby': 'rb', 'php': 'php', 'html': 'html', 'css': 'css',
    'sql': 'sql', 'bash': 'sh', 'shell': 'sh', 'sh': 'sh', 'json': 'json',
    'yaml': 'yaml', 'yml': 'yaml', 'kotlin': 'kt', 'swift': 'swift',
    'r': 'r', 'scala': 'scala', 'dart': 'dart',
}


def _slugify(text, maxlen=40):
    text = re.sub(r'[^a-zA-Z0-9]+', '_', (text or '').strip()).strip('_').lower()
    return (text[:maxlen] or 'code')


def save_code_files(video_dir, analysis, vision_code_blocks, local_code_blocks):
    """For coding tutorials: dump every extracted code block to real,
    individually-saved .py/.js/etc files in chronological order, so nothing
    needs to be retyped from the report. Returns list of written paths."""
    entries = []  # (seconds, language, code, description)

    for sec in analysis.get('sections', []):
        base_sec = sec.get('start_seconds', 0)
        for cb in sec.get('code_blocks', []):
            if cb.get('code'):
                entries.append((base_sec, cb.get('language', ''), cb['code'],
                               cb.get('description') or sec.get('heading', '')))

    for ts, cb in vision_code_blocks.items():
        if cb.get('code'):
            entries.append((ts, cb.get('language', ''), cb['code'], 'on-screen code'))

    for ts, cb in local_code_blocks.items():
        if cb.get('code'):
            entries.append((ts, cb.get('language', ''), cb['code'], 'on-screen code (OCR)'))

    if not entries:
        return []

    entries.sort(key=lambda e: e[0])

    code_dir = video_dir / 'code'
    code_dir.mkdir(exist_ok=True)
    written = []
    for i, (ts, lang, code, desc) in enumerate(entries, 1):
        ext = _LANG_EXT.get((lang or '').strip().lower(), 'txt')
        m, s = divmod(int(ts), 60)
        fname = f"{i:02d}_{m:02d}{s:02d}_{_slugify(desc)}.{ext}"
        fpath = code_dir / fname
        if _safe_write_text(fpath, code if code.endswith('\n') else code + '\n'):
            written.append(str(fpath))
    return written


def filter_must_know(analysis):
    """Return a shallow copy of `analysis` with 'extra'-tagged sections
    removed — used for the optional --quick-revision export."""
    filtered = dict(analysis)
    filtered['sections'] = [s for s in analysis.get('sections', [])
                            if s.get('importance') != 'extra']
    return filtered


def save_anki_flashcards(video_dir, cards):
    """Write cards as a tab-separated .txt — Anki's "Basic" note type import
    format is exactly `question<TAB>answer` per line, so this file can be
    dragged straight into Anki (File > Import)."""
    if not cards:
        return None
    path = video_dir / 'flashcards.txt'
    lines = []
    for c in cards:
        q = (c['question'] or '').replace('\t', ' ').replace('\n', '<br>')
        a = (c['answer'] or '').replace('\t', ' ').replace('\n', '<br>')
        lines.append(f"{q}\t{a}")
    return str(path) if _safe_write_text(path, '\n'.join(lines) + '\n') else None


# ═══════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════

def process_video(video_id, title, channel, base_dir, args, budget):
    """Full pipeline for one video."""
    video_id = _require_safe_video_id(video_id)
    safe_title = title.encode('ascii', 'replace').decode('ascii')
    print()
    if ui is not None and ui.console.is_terminal:
        ui.divider(f" {safe_title[:70]}  ({video_id}) ")
    else:
        print(f"{'=' * 60}")
        print(f"  {safe_title}")
        print(f"  ({video_id})")
        print(f"{'=' * 60}")

    video_dir = base_dir / video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    source_url = f"https://www.youtube.com/watch?v={video_id}"

    # Small metadata file every downstream tool (search index, Q&A, combined
    # flashcards, master index) reads instead of re-deriving title/channel —
    # also what survives into the global cache write-through.
    _safe_write_json(video_dir / 'video_info.json',
                     {'video_id': video_id, 'title': title, 'channel': channel,
                      'source_url': source_url})

    # ── Resume / checkpoint setup ──────────────────────────────────────
    # If a previous run of this exact video was interrupted (lost internet,
    # hit the API budget, was killed, etc.), we don't want to redo work
    # that already finished — transcript fetches and API calls already
    # spent are cached on disk and reused, so the video only needs to
    # pick up from wherever it stopped. Pass --fresh to disable this and
    # force a full reprocess.
    resume = not getattr(args, 'fresh', False)
    transcript_cache_path = video_dir / 'transcript_cache.json'
    meta_cache_path = video_dir / 'video_meta_cache.json'
    sections_path = video_dir / 'sections.json'
    vision_cache_path = video_dir / 'vision_cache.json'

    def _outputs_already_built():
        needed = []
        if args.format in ('pdf', 'both'):
            needed.append(video_dir / 'report.pdf')
        if args.format in ('html', 'both'):
            needed.append(video_dir / 'report.html')
        if args.format in ('md', 'both'):
            needed.append(video_dir / 'report.md')
        return bool(needed) and all(p.exists() for p in needed)

    def _load_index_record():
        """Build the lightweight record main() needs for combined_report.pdf
        / the master index, from whatever's already on disk (used on both
        the local-resume and global-cache-hit paths, so neither skips the
        video's entry in the playlist index)."""
        sec_list = []
        try:
            with open(sections_path, 'r', encoding='utf-8') as f:
                for s in json.load(f).get('sections', []):
                    sec_list.append({'start_seconds': s.get('start_seconds', 0),
                                     'heading': s.get('heading', ''),
                                     'importance': s.get('importance', 'must_know')})
        except Exception as e:
            _log.debug("Couldn't read sections from %s: %s", sections_path, e)
        html_p = video_dir / 'report.html'
        pdf_p = video_dir / 'report.pdf'
        md_p = video_dir / 'report.md'
        quiz_cards = None
        try:
            with open(video_dir / 'quiz.json', 'r', encoding='utf-8') as f:
                quiz_cards = json.load(f)
        except Exception as e:
            _log.debug("Couldn't read quiz.json in %s: %s", video_dir, e)
        return {
            'video_id': video_id, 'title': title, 'channel': channel,
            'html': str(html_p.resolve()) if html_p.exists() else None,
            'pdf': str(pdf_p.resolve()) if pdf_p.exists() else None,
            'md': str(md_p.resolve()) if md_p.exists() else None,
            'sections': sec_list,
            'quiz_cards': quiz_cards,
        }

    # ── Global cache: same video seen in another playlist/run already? ──
    global_cache_dir = getattr(args, 'global_cache_dir', None)
    global_video_dir = Path(global_cache_dir) / video_id if global_cache_dir else None
    if (resume and global_video_dir and global_video_dir.exists()
            and (global_video_dir / 'sections.json').exists()):
        print(f"  [global-cache] '{title[:50]}' already processed "
              f"(seen in another playlist/run) — reusing, 0 API calls.")
        for item in global_video_dir.iterdir():
            try:
                if item.is_file():
                    shutil.copy2(item, video_dir / item.name)
                elif item.is_dir():
                    shutil.copytree(item, video_dir / item.name, dirs_exist_ok=True)
            except Exception as e:
                _log.debug("Couldn't copy cached item %s into %s: %s", item, video_dir, e)
        if _outputs_already_built():
            return _load_index_record()
        # Cached data exists but not in the format requested this run —
        # fall through and rebuild reports from the cached analysis/frames.

    if resume and _outputs_already_built():
        print("  [resume] Report already exists for this video — skipping "
              "(use --fresh to reprocess).")
        return _load_index_record()

    # ── 1. Transcript + lightweight metadata (cached — a later failure never
    #      forces a re-fetch). These two are independent network calls, so
    #      they run concurrently instead of one after another — free speedup,
    #      no quality impact. ──
    print("  [1/9] Fetching transcript + metadata...")
    transcript = None
    if resume and transcript_cache_path.exists():
        try:
            with open(transcript_cache_path, 'r', encoding='utf-8') as f:
                transcript = json.load(f)
            print(f"    [resume] Using cached transcript ({len(transcript)} lines).")
        except Exception:
            transcript = None

    video_meta = None
    if resume and meta_cache_path.exists():
        try:
            with open(meta_cache_path, 'r', encoding='utf-8') as f:
                video_meta = json.load(f)
        except Exception:
            video_meta = None

    jobs = {}
    with ThreadPoolExecutor(max_workers=2) as fetch_pool:
        if transcript is None:
            jobs['transcript'] = fetch_pool.submit(get_transcript, video_id, language=args.language)
        if video_meta is None:
            jobs['meta'] = fetch_pool.submit(fetch_video_metadata, video_id)
        for name, fut in jobs.items():
            try:
                result = fut.result()
            except Exception:
                result = None
            if name == 'transcript':
                transcript = result
            else:
                video_meta = result

    if transcript:
        _safe_write_json(transcript_cache_path, transcript)

    if not transcript and not getattr(args, 'no_whisper_fallback', False):
        transcript = transcribe_with_whisper(video_id, video_dir,
                                             model_size=getattr(args, 'whisper_model', 'base'))
        if transcript:
            _safe_write_json(transcript_cache_path, transcript)

    if not transcript:
        raise VideoProcessingError(
            "No transcript available — YouTube captions are missing/disabled "
            "and the Whisper fallback is unavailable or failed. Re-running "
            "won't help unless that changes (try --language, install/enable "
            "Whisper, or pass --no-whisper-fallback off).")

    video_meta = video_meta or {}
    _safe_write_json(meta_cache_path, video_meta)

    # ── 2. Pass 1: AI text analysis (1 API call) — reused if already done ──
    analysis = None
    if resume and sections_path.exists():
        try:
            with open(sections_path, 'r', encoding='utf-8') as f:
                analysis = json.load(f)
            print(f"  [2/9] [resume] Using cached analysis "
                  f"({len(analysis.get('sections', []))} sections) — 0 API calls.")
        except Exception:
            analysis = None

    if analysis is None:
        if args.skip_ai:
            print("  [2/9] --skip-ai: keyword heuristics (0 API calls)")
            analysis = analyze_with_regex(transcript)
        else:
            print("  [2/9] Pass 1: AI transcript analysis (1 API call)...")
            text, truncated = build_timestamped_text(transcript,
                                                      max_chars=args.max_chars)
            if truncated:
                print(f"    ⚠ Transcript truncated at {args.max_chars} chars")
            try:
                analysis = analyze_with_gemini_v4(text, args.api_key, budget,
                                                  chapters=video_meta.get('chapters'))
                print(f"    [OK] Type: {analysis['video_type']} | "
                      f"{len(analysis['sections'])} sections")
                _warn_if_language_drifted(text, analysis['sections'])
            except RuntimeError as e:
                if getattr(args, 'degrade_on_error', False):
                    print(f"    [X] AI failed: {e}")
                    print("    --degrade-on-error is on: falling back to "
                          "keyword heuristics (lower quality notes).")
                    analysis = analyze_with_regex(transcript)
                    analysis['degraded'] = True
                else:
                    raise VideoProcessingError(
                        f"Pass 1 AI analysis failed: {e} — The transcript "
                        f"is already cached, so nothing is lost; re-run the "
                        f"same command to retry just this step (or pass "
                        f"--degrade-on-error to get lower-quality "
                        f"keyword-based notes instead of stopping).")

        _safe_write_json(sections_path, analysis, indent=2)

    video_type = analysis.get('video_type', 'general')

    # ── 2b. Optional extra passes: key takeaways (--summary), quiz (--quiz) ──
    takeaways_path = video_dir / 'key_takeaways.json'
    key_takeaways = None
    if getattr(args, 'summary', False):
        if resume and takeaways_path.exists():
            try:
                with open(takeaways_path, 'r', encoding='utf-8') as f:
                    key_takeaways = json.load(f)
            except Exception:
                key_takeaways = None
        if key_takeaways is None and not args.skip_ai and budget.available():
            key_takeaways = generate_key_takeaways(analysis, args.api_key, budget)
            _safe_write_json(takeaways_path, key_takeaways)

    quiz_path = video_dir / 'quiz.json'
    quiz_cards = None
    if getattr(args, 'quiz', False):
        if resume and quiz_path.exists():
            try:
                with open(quiz_path, 'r', encoding='utf-8') as f:
                    quiz_cards = json.load(f)
            except Exception:
                quiz_cards = None
        if quiz_cards is None and not args.skip_ai and budget.available():
            quiz_cards = generate_quiz(analysis, args.api_key, budget)
            _safe_write_json(quiz_path, quiz_cards)

    # ── 3. Download video (skipped if a previous run already downloaded it —
    #      yt-dlp itself resumes partial/interrupted downloads by default) ──
    quality = args.quality
    if quality == 'auto':
        quality = '720' if video_type in ('coding', 'slides', 'whiteboard') else '480'

    all_timestamps = collect_screenshot_timestamps(
        analysis, max_screenshots=args.max_screenshots)

    keep_video = getattr(args, 'keep_video', False)
    existing_video_path = video_dir / f"{video_id}.mp4"

    if not all_timestamps and video_type != 'slides':
        print("  [3/9] No screenshots needed, skip download.")
        video_path = None
    elif resume and existing_video_path.exists() and existing_video_path.stat().st_size > 0:
        print("  [3/9] [resume] Using already-downloaded video file.")
        video_path = str(existing_video_path)
    else:
        mode = "video+audio, keeping file" if keep_video else "video-only, faster"
        print(f"  [3/9] Downloading video ({quality}p, {mode})...")
        try:
            video_path = download_video(video_id, video_dir, quality=quality,
                                        keep_audio=keep_video)
        except Exception as e:
            if getattr(args, 'degrade_on_error', False):
                print(f"    [X] Download failed: {e}")
                print("    --degrade-on-error is on: continuing with a "
                      "report that has no screenshots.")
                video_path = None
            else:
                raise VideoProcessingError(
                    f"Video download failed: {e} — Transcript and AI "
                    f"analysis are already cached; re-run the same command "
                    f"to retry just the download (or pass --degrade-on-error "
                    f"to get a report with no screenshots instead of "
                    f"stopping).")

    # ── 4. Slide detection (FREE, local) ──
    slide_transitions = []
    if video_path and video_type in ('slides', 'lecture', 'whiteboard'):
        print("  [4/9] Slide detection (local)...")
        with _spinner("Scanning frames for slide transitions…"):
            slide_transitions = detect_slide_transitions(video_path)
        all_timestamps = collect_screenshot_timestamps(
            analysis, slide_transitions=slide_transitions,
            max_screenshots=args.max_screenshots)
    else:
        print("  [4/9] Slide detection: N/A")

    print(f"    Total timestamps: {len(all_timestamps)}")

    # ── 5. Extract frames ──
    screenshot_map = {}
    ts_meta = {}  # seconds → (visual_type, description)

    if video_path and all_timestamps:
        print(f"  [5/9] Extracting {len(all_timestamps)} frame(s) (clean board mode)...")
        resumed_frames = 0
        # One shared decoder handle for every screenshot in this video —
        # opening/closing the file per-frame was the main bottleneck for
        # videos with many screenshots. Same algorithm, same output
        # quality, just far fewer file (re)opens.
        #
        # This whole loop runs inside try/finally so shared_cap.release()
        # always happens, even if a single frame throws partway through
        # (e.g. a corrupted frame, a disk-full write). Without the
        # finally, an exception here would leak the open handle on
        # video_path — on Windows in particular that can leave the file
        # locked, so the later cleanup step that deletes the downloaded
        # video (when --keep-video isn't passed) would silently fail.
        shared_cap = cv2.VideoCapture(video_path)
        if not shared_cap.isOpened():
            shared_cap.release()
            shared_cap = None
        try:
            for ts, vtype_tag, desc in _progress_iter(
                    all_timestamps, len(all_timestamps), "Extracting frames"):
                m, s = divmod(ts, 60)
                frame_path = str(video_dir / f"frame_{m:02d}_{s:02d}.png")

                # Reuse a frame already extracted in a previous (interrupted) run
                if resume and os.path.exists(frame_path):
                    screenshot_map[ts] = frame_path
                    ts_meta[ts] = (vtype_tag, desc)
                    resumed_frames += 1
                    continue

                # Extract the best possible exact frame
                extract_best_unaltered_frame(video_path, ts, frame_path, cap=shared_cap)

                if not os.path.exists(frame_path):
                    continue

                img = cv2.imread(frame_path)
                if img is not None:
                    overlaid = add_timestamp_overlay(img, f"{m:02d}:{s:02d}")
                    cv2.imwrite(frame_path, overlaid)
                    screenshot_map[ts] = frame_path
                    ts_meta[ts] = (vtype_tag, desc)
        finally:
            if shared_cap is not None:
                shared_cap.release()
        if resumed_frames:
            print(f"    [resume] Reused {resumed_frames} already-extracted frame(s).")
    else:
        print("  [5/9] No frames to extract.")

    # ── 6. Deduplicate frames (FREE, local) ──
    if len(screenshot_map) > 1:
        print(f"  [6/9] Deduplicating {len(screenshot_map)} frame(s) (local)...")
        screenshot_map = deduplicate_frames(screenshot_map)
        print(f"    Kept: {len(screenshot_map)} unique frame(s)")
    else:
        print("  [6/9] Dedup: N/A")

    # ── 7. Local OCR + Vision analysis ──
    vision_code_blocks = {}
    local_code_blocks = {}
    cropped_visuals_map = {}

    # 7a. Local OCR first (FREE — saves API calls)
    if HAS_TESSERACT and screenshot_map:
        print(f"  [7/9] Local OCR (Tesseract, FREE)...")
        ocr_count = 0
        for ts, img_path in _progress_iter(
                sorted(screenshot_map.items()), len(screenshot_map), "Running local OCR"):
            ocr_text = local_ocr_extract(img_path)
            if ocr_text:
                is_code, lang = detect_code_in_text(ocr_text)
                if is_code:
                    local_code_blocks[ts] = {'language': lang, 'code': ocr_text}
                    ocr_count += 1
        if ocr_count:
            print(f"    [OK] Extracted code from {ocr_count} frame(s) locally (0 API calls)")
    elif not HAS_TESSERACT:
        print("  [7/9] Local OCR: pytesseract not installed (optional)")
    else:
        print("  [7/9] Local OCR: no frames")

    # 7b. Vision analysis (batched — saves ~85% API calls)
    need_vision = (screenshot_map and not args.no_vision and not args.skip_ai
                   and budget.available())

    if need_vision:
        # Prepare items for batch analysis
        ts_context = {}
        for sec in analysis.get('sections', []):
            ctx = f"Section: {sec.get('heading', '')}\nNotes: {sec.get('notes', '')[:500]}"
            for ss in sec.get('screenshot_timestamps', []):
                ts_context[ss['seconds']] = ctx
        items = [(ts, path, ts_context.get(ts, '')) for ts, path in sorted(screenshot_map.items())]
        batch_count = (len(items) + VISION_BATCH_SIZE - 1) // VISION_BATCH_SIZE
        print(f"    Vision analysis: {len(items)} images in "
              f"~{batch_count} batch(es) ({VISION_BATCH_SIZE}/batch)...")

        if not resume and vision_cache_path.exists():
            try:
                vision_cache_path.unlink()
            except Exception as e:
                _log.debug("Couldn't remove stale vision cache %s: %s", vision_cache_path, e)
        vision_results = analyze_screenshots_batch(
            items, args.api_key, budget, cache_path=str(vision_cache_path))

        # Process vision results
        dropped_not_useful = []
        for ts, result in vision_results.items():
            if not result:
                continue

            # Remove useless frames (talking head / blank)
            if not result.get('is_useful', True):
                m, s = divmod(ts, 60)
                try:
                    os.remove(screenshot_map[ts])
                    del screenshot_map[ts]
                    dropped_not_useful.append(f"{m:02d}:{s:02d}")
                except Exception as e:
                    _log.debug("Couldn't remove frame flagged not-useful at ts=%s: %s", ts, e)
                continue

            caption = result.get('caption', '')
            if caption:
                for sec in analysis.get('sections', []):
                    for ss in sec.get('screenshot_timestamps', []):
                        if ss['seconds'] == ts:
                            ss['vision_caption'] = caption

            # NOTE: Cropping is intentionally disabled — screenshots are kept
            # as the full, uncropped frame. cropped_visuals_map stays empty so
            # reports only ever show the complete frame (no zoomed sub-crops).

            # Extract code from vision (supplements local OCR)
            ext_code = result.get('extracted_code')
            if (ext_code and isinstance(ext_code, dict) and ext_code.get('code')
                    and ts not in local_code_blocks):
                vision_code_blocks[ts] = ext_code

        useful_vision = sum(1 for r in vision_results.values()
                           if r and r.get('is_useful', True))
        missing_after_all = len(items) - len(vision_results)
        print(f"    [OK] Vision: {len(items)} sent → {useful_vision} kept with AI "
              f"analysis, {len(dropped_not_useful)} dropped (AI: not useful — "
              f"talking-head/blank), "
              f"{sum(len(v) for v in cropped_visuals_map.values())} crops, "
              f"{len(vision_code_blocks)} code extractions")
        if dropped_not_useful:
            print(f"        dropped at: {', '.join(dropped_not_useful)}")
        if missing_after_all:
            print(f"    ⚠ {missing_after_all} screenshot(s) still have no AI "
                  f"analysis after retries — they'll appear in the report as "
                  f"plain images, without a caption/description.")
    else:
        reason = ("--no-vision" if args.no_vision
                  else "--skip-ai" if args.skip_ai
                  else "no budget" if not budget.available()
                  else "no frames")
        print(f"    Vision: skipped ({reason})")

    # ── 8. Flowcharts (FREE — mermaid.ink) ──
    flowchart_images = {}
    if not args.no_flowchart:
        print("  [8/9] Flowcharts (mermaid.ink, FREE)...")

        ofc = analysis.get('overview_flowchart')
        if ofc:
            fp = str(video_dir / 'overview_flowchart.png')
            if generate_flowchart_image(ofc, fp):
                flowchart_images['overview'] = fp
                print("    [OK] Overview flowchart")

        for i, sec in enumerate(analysis.get('sections', [])):
            fc = sec.get('flowchart')
            if fc:
                fp = str(video_dir / f'flowchart_section_{i}.png')
                if generate_flowchart_image(fc, fp):
                    flowchart_images[i] = fp

        if flowchart_images:
            print(f"    [OK] {len(flowchart_images)} flowchart(s) rendered")
    else:
        print("  [8/9] Flowcharts: --no-flowchart")

    # ── 9. Build reports ──
    print("  [9/9] Building reports...")
    output_paths = {}

    if args.format in ('html', 'both'):
        with _spinner("Building HTML report…"):
            html_path = build_html_report(
                video_dir, title, channel, source_url,
                analysis, screenshot_map, flowchart_images,
                video_meta=video_meta, cropped_visuals_map=cropped_visuals_map,
                vision_code_blocks=vision_code_blocks,
                local_code_blocks=local_code_blocks,
                key_takeaways=key_takeaways, quiz_cards=quiz_cards)
        output_paths['html'] = html_path
        print(f"    [OK] HTML: {html_path.resolve()}")

    if args.format in ('pdf', 'both'):
        with _spinner("Building PDF report…"):
            pdf_path = build_pdf_report(
                video_dir, title, channel, source_url,
                analysis, screenshot_map, cropped_visuals_map,
                vision_code_blocks, local_code_blocks, flowchart_images,
                key_takeaways=key_takeaways, quiz_cards=quiz_cards)
        output_paths['pdf'] = pdf_path
        print(f"    [OK] PDF:  {pdf_path.resolve()}")

    if args.format in ('md', 'both'):
        with _spinner("Building Markdown report…"):
            md_path = build_markdown_report(
                video_dir, title, channel, source_url,
                analysis, screenshot_map, cropped_visuals_map,
                vision_code_blocks, local_code_blocks, flowchart_images,
                key_takeaways=key_takeaways, quiz_cards=quiz_cards)
        output_paths['md'] = md_path
        print(f"    [OK] MD:  {md_path.resolve()}")

    # Runnable code files (coding tutorials) — always on, no extra API cost
    code_files = save_code_files(video_dir, analysis, vision_code_blocks, local_code_blocks)
    if code_files:
        print(f"    [OK] Code: {len(code_files)} runnable file(s) in {video_dir / 'code'}")

    # Anki-importable flashcards, if --quiz was on
    if quiz_cards:
        fc_path = save_anki_flashcards(video_dir, quiz_cards)
        if fc_path:
            print(f"    [OK] Flashcards: {fc_path} (Anki-importable)")

    # Quick-revision export: same reports, must-know sections only
    if getattr(args, 'quick_revision', False):
        extra_count = sum(1 for s in analysis['sections'] if s.get('importance') == 'extra')
        if extra_count == 0:
            print("    [skip] --quick-revision: no 'extra' sections to filter out.")
        else:
            qr_analysis = filter_must_know(analysis)
            if args.format in ('html', 'both'):
                qr_html = build_html_report(
                    video_dir, f"{title} (Quick Revision)", channel, source_url,
                    qr_analysis, screenshot_map, flowchart_images,
                    video_meta=video_meta, cropped_visuals_map=cropped_visuals_map,
                    vision_code_blocks=vision_code_blocks,
                    local_code_blocks=local_code_blocks, output_name='quick_revision')
                print(f"    [OK] Quick revision HTML: {qr_html} "
                      f"({len(qr_analysis['sections'])}/{len(analysis['sections'])} sections)")
            if args.format in ('pdf', 'both'):
                qr_pdf = build_pdf_report(
                    video_dir, f"{title} (Quick Revision)", channel, source_url,
                    qr_analysis, screenshot_map, cropped_visuals_map,
                    vision_code_blocks, local_code_blocks, flowchart_images,
                    output_name='quick_revision')
                print(f"    [OK] Quick revision PDF: {qr_pdf}")

    # Cleanup video file — kept on disk if --keep-video was passed
    if video_path:
        if keep_video:
            print(f"    [OK] Video kept: {video_path}")
        else:
            try:
                os.remove(video_path)
            except Exception as e:
                _log.debug("Couldn't remove downloaded video %s: %s", video_path, e)

    # Write-through to the global cache so this video is never reprocessed
    # again in ANY playlist/run that shares the same cache dir.
    if global_video_dir:
        try:
            global_video_dir.mkdir(parents=True, exist_ok=True)
            for item in video_dir.iterdir():
                if item.name.endswith('.mp4'):
                    continue  # never cache the raw video, only derived outputs
                try:
                    if item.is_file():
                        shutil.copy2(item, global_video_dir / item.name)
                    elif item.is_dir():
                        shutil.copytree(item, global_video_dir / item.name, dirs_exist_ok=True)
                except Exception as e:
                    _log.debug("Couldn't cache %s into global cache: %s", item, e)
        except Exception as e:
            _log.debug("Couldn't write through to global cache %s: %s", global_video_dir, e)

    # Summary
    total_code = (sum(len(s.get('code_blocks', [])) for s in analysis['sections'])
                  + len(vision_code_blocks) + len(local_code_blocks))
    total_crops = sum(len(v) for v in cropped_visuals_map.values())
    print(f"\n  +-- Summary --------------------------------+")
    print(f"  | Sections:    {len(analysis['sections'])}")
    print(f"  | Screenshots: {len(screenshot_map)}")
    print(f"  | Code blocks: {total_code}")
    print(f"  | Flowcharts:  {len(flowchart_images)}")
    print(f"  | {budget}")
    print(f"  +--------------------------------------------+")

    return {
        'video_id': video_id, 'title': title, 'channel': channel,
        'html': str(output_paths['html'].resolve()) if output_paths.get('html') else None,
        'pdf': str(output_paths['pdf'].resolve()) if output_paths.get('pdf') else None,
        'md': str(output_paths['md'].resolve()) if output_paths.get('md') else None,
        'sections': [{'start_seconds': s.get('start_seconds', 0),
                     'heading': s.get('heading', ''),
                     'importance': s.get('importance', 'must_know')}
                    for s in analysis.get('sections', [])],
        'quiz_cards': quiz_cards,
    }


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def _hdr(label, value):
    """Formats one 'Label:   value' line for the startup config box, padded
    so every value starts in the same column no matter how long the label
    is (previously 'AI provider:' and 'Global cache:' — the two longest
    labels — didn't get enough padding and landed one/two columns off from
    everything else)."""
    return f"  {label:<14}{value}"


def main():
    global TEXT_MODEL, VISION_MODEL
    parser = argparse.ArgumentParser(
        description="MARROW — Extract EVERYTHING from "
                    "YouTube videos into PDF + Markdown notes. "
                    "Optimized for Gemini free tier.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "https://www.youtube.com/watch?v=XXXX"
  %(prog)s "https://www.youtube.com/playlist?list=XXXX" --max-videos 5
  %(prog)s "URL" --no-vision          # 1 API call per video
  %(prog)s "URL" --skip-ai            # 0 API calls (local only)
  %(prog)s "URL" --format md           # Markdown only
  %(prog)s "URL" --max-api-calls 15    # Hard budget limit
  %(prog)s "URL" --fresh               # Ignore saved progress, start over
  %(prog)s "URL" --open                # Open the report when done

Interrupted runs (lost internet, hit API limit, killed process) resume
automatically — just re-run the same command. Pass --fresh to disable this.

If a video hits an unrecoverable error (no transcript, Pass 1 AI failure,
download failure) it STOPS right there instead of saving a broken report —
re-run the same command to retry just that step. Pass --degrade-on-error to
instead keep going with lower-quality output.

Ask questions across everything you've already processed (no URL needed):
  %(prog)s --ask "what is a segment tree?"
  %(prog)s --ask "binary search variations" --library-dir output

Free tier costs per video:
  Pass 1 (transcript): 1 API call
  Pass 2 (vision):     ~3-6 calls (batched, 8 imgs/call)
  Flowcharts:          0 calls (mermaid.ink, free)
  Slide detection:     0 calls (OpenCV, local)
  Frame dedup:         0 calls (OpenCV, local)
  OCR (if Tesseract):  0 calls (local)
""",
    )
    parser.add_argument('url', nargs='?', default=None,
                        help="Video, playlist, or channel URL (omit when using --ask)")
    parser.add_argument('output_dir', nargs='?', default='output',
                        help="Output directory (default: output)")
    parser.add_argument('max_videos', nargs='?', type=int, default=10,
                        help="Max videos for playlist/channel (default: 10)")

    qa = parser.add_argument_group('Ask your library (no URL needed)')
    qa.add_argument('--ask', default=None, metavar='QUESTION',
                    help="Answer a question using ONLY your already-saved "
                         "notes (searches --library-dir), with citations "
                         "back to the source video + timestamp")
    qa.add_argument('--library-dir', default=None,
                    help="Where to search for --ask (default: the global "
                         "cache dir, so it covers everything you've ever "
                         "processed; pass an --output-dir to search just "
                         "one run/playlist instead)")

    api = parser.add_argument_group('API (Gemini free tier)')
    api.add_argument('--api-key',
                     default=os.environ.get('GEMINI_API_KEY'),
                     help="Gemini API key (or GEMINI_API_KEY env)")
    api.add_argument('--model', default=None,
                     help=f"Override model (default: {TEXT_MODEL})")
    api.add_argument('--skip-ai', action='store_true',
                     help="Zero API calls — keyword heuristics only")
    api.add_argument('--max-api-calls', type=int, default=None,
                     help="Hard limit on total API calls")

    out = parser.add_argument_group('Output')
    out.add_argument('--format', choices=['pdf', 'html', 'md', 'both'],
                     default='both',
                     help="Output format — 'both' builds pdf+html+md "
                          "(default: both)")
    out.add_argument('--open', action='store_true',
                     help="Automatically open the finished report (or the "
                          "master index, for playlists) in your browser "
                          "when the run finishes")

    feat = parser.add_argument_group('Features')
    feat.add_argument('--no-vision', action='store_true',
                      help="Skip Vision (Pass 2) — only 1 API call/video")
    feat.add_argument('--no-flowchart', action='store_true',
                      help="Skip flowchart generation")
    feat.add_argument('--language', default=None,
                      help="Preferred transcript language (e.g., hi, en)")
    feat.add_argument('--fresh', action='store_true',
                      help="Ignore any saved progress and reprocess every "
                           "video from scratch (by default, a video that "
                           "was interrupted — e.g. by a lost connection or "
                           "a hit API limit — resumes from where it left "
                           "off instead of starting over)")
    feat.add_argument('--summary', action='store_true',
                      help="Add a 'Key Takeaways' box at the top of each "
                           "report (1 extra API call/video)")
    feat.add_argument('--quiz', action='store_true',
                      help="Generate a revision quiz + Anki-importable "
                           "flashcards.txt per video (1 extra API call/video)")
    feat.add_argument('--quick-revision', action='store_true',
                      help="Also build a filtered quick_revision.{pdf,html} "
                           "containing only 'must-know' sections (skips "
                           "tangents/extras) — for last-minute revision")
    feat.add_argument('--no-whisper-fallback', action='store_true',
                      help="Don't fall back to local Whisper transcription "
                           "for videos with no YouTube captions at all "
                           "(default: fallback is on, but silently skips "
                           "if no Whisper package is installed)")
    feat.add_argument('--degrade-on-error', action='store_true',
                      help="If Pass 1 AI analysis or the video download "
                           "fails, fall back to lower-quality output "
                           "(keyword heuristics / no screenshots) and keep "
                           "going, instead of the default: stop that video "
                           "right there so a plain re-run retries it "
                           "properly instead of quietly saving a broken "
                           "report")
    feat.add_argument('--whisper-model', default='base',
                      choices=['tiny', 'base', 'small', 'medium', 'large-v3'],
                      help="Local Whisper model size for the no-captions "
                           "fallback — bigger is more accurate but slower "
                           "on CPU (default: base)")

    dl = parser.add_argument_group('Video download')
    dl.add_argument('--quality', default='auto',
                    choices=['auto', '480', '720', '1080', '1440', '2160', 'best'],
                    help="Screenshot source resolution cap — 'auto' picks "
                         "720 for coding/slides/whiteboard videos and 480 "
                         "otherwise; use '1080'/'1440'/'2160'/'best' for "
                         "sharper screenshots on dense-text videos "
                         "(default: auto)")
    dl.add_argument('--keep-video', action='store_true',
                    help="Keep the downloaded video file (with audio) in "
                         "each video's output folder instead of deleting "
                         "it after processing")

    perf = parser.add_argument_group('Performance')
    perf.add_argument('--parallel', type=int, default=1, metavar='N',
                      help="Process N videos from a playlist at once "
                           "(2-3 is reasonable; note log lines from "
                           "different videos may interleave). Default: 1 "
                           "(sequential, same as before)")
    perf.add_argument('--fallback-models', default=None,
                      help="Comma-separated backup models to roll over to "
                           "if the primary model gets rate-limited, e.g. "
                           "'gemini-3.1-flash-lite,gemini-flash-latest'")
    cache = parser.add_argument_group('Caching')
    cache.add_argument('--global-cache-dir', default=DEFAULT_GLOBAL_CACHE_DIR,
                       help="A video already processed here (in ANY "
                            f"playlist/run) is never reprocessed "
                            f"(default: {DEFAULT_GLOBAL_CACHE_DIR})")
    cache.add_argument('--no-global-cache', action='store_true',
                       help="Disable the cross-playlist/run cache above")

    lim = parser.add_argument_group('Limits')
    lim.add_argument('--max-videos', type=int, default=None,
                     dest='max_videos_flag',
                     help="Override max videos (flag version)")

    args = parser.parse_args()

    if args.max_videos_flag is not None:
        args.max_videos = args.max_videos_flag

    if args.model:
        TEXT_MODEL = args.model
        VISION_MODEL = args.model
        TEXT_MODEL_FALLBACKS[0] = args.model
        VISION_MODEL_FALLBACKS[0] = args.model
    if args.fallback_models:
        extra = [m.strip() for m in args.fallback_models.split(',') if m.strip()]
        TEXT_MODEL_FALLBACKS[:] = [TEXT_MODEL_FALLBACKS[0]] + extra
        VISION_MODEL_FALLBACKS[:] = [VISION_MODEL_FALLBACKS[0]] + extra

    args.global_cache_dir = None if args.no_global_cache else args.global_cache_dir

    if not args.ask and not args.url:
        parser.error('a URL is required (or use --ask "question" to search '
                     'your existing notes without one)')

    # ── --ask mode: answer from already-saved notes, no video processing ──
    if args.ask:
        if not args.api_key and not (mrp is not None and mrp.rotation_enabled()):
            print("\n" + "=" * 55)
            print("  No AI provider configured (needed to answer questions).")
            print("  Run marrow.py once with no arguments to add a key via /keys,")
            print("  or paste a bare Gemini key below to continue this run only.")
            print("  Get a FREE Gemini key at: https://aistudio.google.com/app/apikey")
            print("=" * 55)
            args.api_key = input("Please paste your Gemini API key here: ").strip()
            if not args.api_key:
                print("No key provided. Exiting.")
                sys.exit(1)

        library_dir = args.library_dir or args.global_cache_dir or DEFAULT_GLOBAL_CACHE_DIR
        print(f"\nSearching your library at: {library_dir}")
        records = scan_library(library_dir)
        if not records:
            print("No processed videos found there yet — process some "
                  "videos first, or pass --library-dir to point at the "
                  "right folder.")
            sys.exit(1)
        print(f"Loaded {len(records)} saved section(s) across your library.\n")

        qa_budget = APIBudget(max_calls=args.max_api_calls)
        found, answer, sources = answer_question(args.ask, records, args.api_key, qa_budget)
        print(f"Q: {args.ask}\n")
        if not found or not answer:
            print("A: Not covered in your saved notes.")
        else:
            print(f"A: {answer}\n")
            print("Sources:")
            for s in sources:
                m, sec = divmod(int(s.get('start_seconds', 0)), 60)
                link = _yt_ts_link(s.get('source_url', ''), s.get('start_seconds', 0))
                print(f"  - \"{s.get('title', '')}\" @ {m:02d}:{sec:02d} — {link}")
        return

    rotation_ready = mrp is not None and mrp.rotation_enabled()
    if not args.skip_ai and not args.api_key and not rotation_ready:
        print("\n" + "=" * 55)
        print("  No AI provider configured.")
        print("  Run marrow.py once with no arguments to add a key via /keys,")
        print("  or paste a bare Gemini key below to continue this run only.")
        print("  Get a FREE Gemini key at: https://aistudio.google.com/app/apikey")
        print("=" * 55)
        args.api_key = input("Please paste your Gemini API key here: ").strip()
        if not args.api_key:
            print("No key provided. Exiting.")
            sys.exit(1)
        print("Key accepted!\n")
    budget = APIBudget(max_calls=args.max_api_calls)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    args.max_chars = 9999999
    args.max_screenshots = 999

    ai_line = _hdr("AI provider:", mrp.current_status()) if rotation_ready else \
              _hdr("Text model:", TEXT_MODEL_FALLBACKS) + "\n" + _hdr("Vis. model:", VISION_MODEL_FALLBACKS)

    print(f"\n{'=' * 56}")
    print(f"  MARROW")
    print(f"  Video -> comprehensive notes")
    print(f"{'-' * 56}")
    print(_hdr("Output:", out_dir.resolve()))
    print(_hdr("Format:", args.format))
    print(_hdr("Quality:", f"{args.quality}{' (video kept after run)' if args.keep_video else ''}"))
    print(_hdr("Vision:", 'off' if args.no_vision else 'on (batched)'))
    print(_hdr("Flowcharts:", 'off' if args.no_flowchart else 'on (mermaid.ink)'))
    print(_hdr("Summary:", 'on (+1 API call/video)' if args.summary else 'off'))
    print(_hdr("Quiz:", 'on (+1 API call/video)' if args.quiz else 'off'))
    print(_hdr("Local OCR:", '[OK] pytesseract' if HAS_TESSERACT else 'X (pip install pytesseract)'))
    print(_hdr("Budget:", budget))
    print(ai_line)
    print(_hdr("Parallel:", f"{args.parallel} video(s) at a time"))
    print(_hdr("Global cache:", 'off' if not args.global_cache_dir else args.global_cache_dir))
    print(_hdr("Resume:", 'off (--fresh: reprocessing everything)' if args.fresh else 'on (interrupted videos continue where they stopped)'))
    print(f"{'=' * 56}")

    print(f"\nFetching video list: {args.url}")
    videos, playlist_title = get_video_list(args.url, max_videos=args.max_videos)
    print(f"Found {len(videos)} video(s)")

    total = len(videos)
    results = [None] * total
    statuses = [None] * total   # 'ok' | 'stopped' | 'error'
    messages = [None] * total
    completed = 0
    progress_lock = threading.Lock()
    is_parallel = args.parallel > 1 and total > 1

    def _run_one(idx, video_id, title, channel):
        nonlocal completed
        if is_parallel:
            # Each worker thread gets its own contextvars Context, so this
            # tag only ever prefixes this thread's own log lines.
            _video_tag_ctx.set(video_id)
            # ...and, for the same reason, this only disables spinners/
            # progress bars/live countdowns for THIS thread — Rich allows
            # only one live-updating region per terminal at a time, so
            # several parallel workers each trying to animate their own
            # would corrupt the output. Sequential (non-parallel) runs are
            # unaffected and keep the full live experience.
            if ui is not None:
                ui.set_parallel_mode(True)
        rec = None
        status = 'ok'
        msg = None
        try:
            rec = process_video(video_id, title, channel, out_dir, args, budget)
        except VideoProcessingError as e:
            status, msg = 'stopped', str(e)
            print(f"\n  [STOPPED] '{title[:60]}'")
            print(f"    Reason: {msg}")
            print(f"    -> Nothing lost — re-run the exact same command "
                  f"later and this video resumes right here.")
        except Exception as e:
            status, msg = 'error', str(e)
            print(f"\n  [X] Unexpected error on '{title[:60]}': {e}")
            import traceback
            traceback.print_exc()
        with progress_lock:
            completed += 1
            pct = completed * 100 // total if total else 100
            print(f"\n[Progress] {completed}/{total} videos done ({pct}%)\n")
        return idx, rec, status, msg

    if is_parallel:
        print(f"\nProcessing with {args.parallel} video(s) in parallel "
              f"(each video's log lines are tagged with its [video_id])...\n")
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = [pool.submit(_run_one, i, vid, t, ch)
                      for i, (vid, t, ch) in enumerate(videos)]
            for fut in as_completed(futures):
                idx, rec, status, msg = fut.result()
                results[idx] = rec
                statuses[idx] = status
                messages[idx] = msg
    else:
        for i, (video_id, title, channel) in enumerate(videos):
            print(f"\n[>] Video {i + 1}/{total}")
            _, rec, status, msg = _run_one(i, video_id, title, channel)
            results[i] = rec
            statuses[i] = status
            messages[i] = msg

    per_video_records = [r for r in results if r]
    entry_point = None  # single best file to open with --open

    if len(per_video_records) > 1:
        pdf_records = [r for r in per_video_records if r.get('pdf')]
        if pdf_records:
            combined = out_dir / 'combined_report.pdf'
            if merge_pdfs(pdf_records, combined):
                print(f"\n[OK] Combined PDF: {combined.resolve()}")
        else:
            print("\n[!] No per-video PDFs to combine — pass --format pdf "
                  "or --format both to get combined_report.pdf")

        index_html = build_playlist_index(out_dir, playlist_title, per_video_records)
        index_html = Path(index_html).resolve()
        print(f"[OK] Master index: {index_html}")
        entry_point = index_html

        # Combined flashcards — every video's cards in one Anki-importable
        # deck, each question tagged with its source video for context.
        all_cards = []
        for r in per_video_records:
            for c in (r.get('quiz_cards') or []):
                all_cards.append({
                    'question': f"[{(r.get('title') or '')[:40]}] {c['question']}",
                    'answer': c['answer'],
                })
        if all_cards:
            combined_fc = save_anki_flashcards(out_dir, all_cards)
            if combined_fc:
                # save_anki_flashcards always writes 'flashcards.txt' — give
                # the playlist-wide deck its own distinct name.
                combined_fc_final = out_dir / 'combined_flashcards.txt'
                try:
                    shutil.move(combined_fc, str(combined_fc_final))
                    print(f"[OK] Combined flashcards: "
                          f"{combined_fc_final.resolve()} "
                          f"({len(all_cards)} cards, Anki-importable)")
                except Exception as e:
                    _log.debug("Couldn't move combined flashcards to %s: %s", combined_fc_final, e)

    elif len(per_video_records) == 1:
        # Single video, not a playlist: open its own polished report
        # directly. (Previously this always fell through to the generic
        # search-index tool below instead, since that got assigned first
        # and the per-video-report fallback further down never got a
        # chance to run once entry_point was already set.)
        for kind in ('html', 'pdf', 'md'):
            if per_video_records[0].get(kind):
                entry_point = Path(per_video_records[0][kind])
                break

    # Search-across-all-notes: one for this run's own output folder...
    if per_video_records:
        run_search = build_search_index(out_dir, out_dir / 'search.html',
                                        title=f"Search: {playlist_title or 'My Notes'}")
        run_search = Path(run_search).resolve()
        print(f"[OK] Search this run's notes: {run_search}")
        if entry_point is None:
            entry_point = run_search

        # ...and one for the WHOLE global cache, so it covers every video
        # you've ever processed across every playlist/run, not just this one.
        if args.global_cache_dir:
            try:
                global_search = build_search_index(
                    Path(args.global_cache_dir), Path(args.global_cache_dir) / 'search.html',
                    title="Search My Entire Video Notes Library")
                print(f"[OK] Search your WHOLE library: {Path(global_search).resolve()}")
            except Exception as e:
                print(f"[!] Could not update global search index: {e}")

    # ── Final summary — impossible to miss, tells you exactly which
    #    video(s) (if any) still need a re-run, and exactly where every
    #    file that DID finish actually lives on disk. ──
    ok_idx = [i for i in range(total) if statuses[i] == 'ok']
    stopped_idx = [i for i in range(total) if statuses[i] == 'stopped']
    error_idx = [i for i in range(total) if statuses[i] == 'error']

    print(f"\n{'=' * 56}")
    print(f"  [OK] {len(ok_idx)}/{total} video(s) completed successfully")
    if stopped_idx or error_idx:
        print(f"  [!]  {len(stopped_idx) + len(error_idx)}/{total} video(s) "
              f"did NOT finish:")
        for i in stopped_idx:
            print(f"       - '{videos[i][1][:55]}'")
            print(f"         {messages[i]}")
        for i in error_idx:
            print(f"       - '{videos[i][1][:55]}' — unexpected error: "
                  f"{messages[i]}")
        print(f"\n  Nothing already finished is lost. Run the exact same "
              f"command again and only the video(s) above will be "
              f"(re)worked on — everything else is skipped instantly.")

    if per_video_records:
        print(f"\n  YOUR NOTES:")
        for r in per_video_records:
            print(f"    '{(r.get('title') or '')[:55]}'")
            for kind, label in (('pdf', 'PDF'), ('html', 'HTML'), ('md', 'MD')):
                if r.get(kind):
                    print(f"       {label}: {r[kind]}")
            if entry_point is None:
                for kind in ('html', 'pdf', 'md'):
                    if r.get(kind):
                        entry_point = Path(r[kind])
                        break

    print(f"\n  All output is saved under: {out_dir.resolve()}")
    print(f"  {budget}")
    if not HAS_TESSERACT:
        print(f"  💡 Tip: pip install pytesseract for FREE local OCR")
    print(f"{'=' * 56}")

    if args.open:
        if entry_point and Path(entry_point).exists():
            if open_file_for_viewing(entry_point):
                print(f"  Opening: {entry_point}")
            else:
                print(f"  [!] Couldn't auto-open — open this path "
                      f"yourself: {entry_point}")
        else:
            print(f"  [!] Nothing finished to open yet.")

    # Non-zero exit code so scripts/automation can tell a re-run is needed,
    # without taking anything away from the summary already printed above.
    if stopped_idx or error_idx:
        sys.exit(1)


if __name__ == '__main__':
    main()
