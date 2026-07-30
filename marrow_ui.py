"""
marrow_ui.py — MARROW's look and feel.

Same anatomy as the Gemini CLI screenshot this was modeled on (block-letter
logo → tips → bordered input box → bottom status bar), redone with:
  - a MARROW wordmark instead of a GEMINI one
  - a warm ember/rust gradient instead of blue-green (marrow ~ bone marrow)
  - a bottom status bar that reports *our* state: working folder, which
    provider/model is currently active in the rotation, and calls used
    this session (Gemini's showed dir / sandbox / model / context-used;
    ours shows dir / provider·model / API calls this session)

This module also carries the "live progress" toolkit — spinners, ticking
countdowns, a download progress bar, and an auto-styled line printer —
used by the engine (see marrow_engine.py) so long, previously-silent
waits (rate-limit pauses, video downloads, AI calls, local Whisper
transcription) always show visible, live-updating feedback instead of
looking like the tool has frozen.

Every one of these degrades gracefully to plain text (or nothing,
matching the old behaviour) when stdout isn't a real terminal, and never
raises even if something about the terminal/Rich version is unusual — a
cosmetic progress bar failing to render must never be the reason a video
fails to process. See `live_ok()` / `_try_start()` below for how that's
enforced, including the case where some OTHER spinner further up the call
stack (e.g. the "Thinking…" status /ask already shows) is active: Rich
only supports one live-updating region per console at a time, and Rich's
behaviour on a second, nested one varies by version (older releases raise
LiveError, newer ones silently do nothing) — `_try_start()` checks first
and catches broadly, so either way we just skip our own widget instead of
crashing or double-rendering.
"""
import contextvars
import os
import re
import sys
import time
from contextlib import contextmanager

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.align import Align

console = Console()

# ─────────────────────────────────────────────────────────────────────
# Themes — each a 3-stop RGB gradient (left -> mid -> right) plus an accent
# color used for borders/prompts elsewhere in the UI.
# ─────────────────────────────────────────────────────────────────────

THEMES = {
    "ember":  {"stops": [(139, 0, 20), (216, 82, 24), (255, 184, 28)], "accent": "#e0703a"},
    "aurora": {"stops": [(88, 24, 168), (168, 40, 176), (64, 200, 220)], "accent": "#b446c8"},
    "mono":   {"stops": [(90, 90, 90), (170, 170, 170), (235, 235, 235)], "accent": "#aaaaaa"},
}
DEFAULT_THEME = "ember"


def resolve_theme():
    """The theme to render engine-side output in. marrow.py sets the
    MARROW_THEME env var to whatever the person picked before handing off
    to marrow_engine.py as a subprocess, so a video processed from the
    interactive shell keeps looking like the same tool instead of
    dropping back to plain defaults the moment processing starts. Falls
    back to the default theme for standalone `python marrow_engine.py`
    use, where there's no shell session to inherit a theme from."""
    env_theme = os.environ.get("MARROW_THEME")
    return env_theme if env_theme in THEMES else DEFAULT_THEME


def _lerp(a, b, t):
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _gradient_color(theme, t):
    """t in [0,1] across the whole banner width."""
    stops = THEMES.get(theme, THEMES[DEFAULT_THEME])["stops"]
    if t <= 0.5:
        return _lerp(stops[0], stops[1], t / 0.5)
    return _lerp(stops[1], stops[2], (t - 0.5) / 0.5)


def _rgb_hex(rgb):
    return "#%02x%02x%02x" % rgb


# ─────────────────────────────────────────────────────────────────────
# 5x7 block font — 1 = filled pixel. Minimal, legible, matches the blocky
# style of the reference screenshot.
# ─────────────────────────────────────────────────────────────────────

_FONT = {
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "W": ["10001", "10001", "10001", "10101", "10101", "11011", "10001"],
}
_WORD = "MARROW"
_SHADOW_ROWS = (5, 6)  # bottom two rows rendered as a textured shadow, not a solid fill


def render_banner(theme=DEFAULT_THEME):
    letters = [_FONT[ch] for ch in _WORD]
    letter_w = 5
    gap = 1
    total_cols = len(letters) * letter_w + (len(letters) - 1) * gap

    for row in range(7):
        line = Text()
        col = 0
        for li, letter in enumerate(letters):
            bits = letter[row]
            for bit in bits:
                t = col / max(1, total_cols - 1)
                rgb = _gradient_color(theme, t)
                if bit == "1":
                    if row in _SHADOW_ROWS:
                        line.append("▓▓", style=_rgb_hex(tuple(c // 2 + 40 for c in rgb)))
                    else:
                        line.append("  ", style=f"on {_rgb_hex(rgb)}")
                else:
                    line.append("  ")
                col += 1
            if li != len(letters) - 1:
                line.append("  ")
                col += gap
        console.print(Align.center(line))
    console.print()


def render_tips(theme=DEFAULT_THEME):
    # Deliberately NOT numbered — numbers here read as a clickable menu
    # ("type 1 to do the first thing"), but this box is just examples.
    # A bullet makes it unambiguous that these are things to try, not
    # choices to pick by number.
    accent = THEMES.get(theme, THEMES[DEFAULT_THEME])["accent"]
    console.print(f"[bold]Tips for getting started:[/bold]")
    tips = [
        "Paste a YouTube URL to turn it into notes (PDF + web page).",
        "[bold]/keys[/bold]      manage your AI provider API keys",
        "[bold]/settings[/bold]  fallback models, key priority, theme",
        "[bold]/library[/bold]   browse videos you've already processed",
        "[bold]/ask[/bold] <q>   ask a question across your saved notes",
        "[bold]/help[/bold]      see every command",
    ]
    for tip in tips:
        console.print(f"[{accent}]•[/{accent}] {tip}")
    console.print()


def status_line(theme=DEFAULT_THEME):
    """A plain, one-shot readout of provider/model + calls used — printed
    fresh above the input box each time, instead of a persistent bottom
    toolbar. (An earlier version pinned this to the terminal's last row
    via prompt_toolkit's bottom_toolbar; on some terminals — Termux in
    particular — that reserved-row redraw fights with plain print() output
    from the processing engine and garbles the screen. A plain line that
    scrolls normally like everything else avoids that entirely.)"""
    import marrow_providers as mrp
    dim = "#888888"
    status = mrp.current_status() or "no provider configured — run /keys"
    console.print(f"[{dim}]{status}[/{dim}]")


def status_ok(msg):
    console.print(f"[green]✓[/green] {msg}")


def status_err(msg):
    console.print(f"[bold red]✗[/bold red] {msg}")


def status_info(msg, theme=DEFAULT_THEME):
    accent = THEMES.get(theme, THEMES[DEFAULT_THEME])["accent"]
    console.print(f"[{accent}]›[/{accent}] {msg}")


def status_warn(msg):
    console.print(f"[yellow]⚠[/yellow] {msg}")


# ─────────────────────────────────────────────────────────────────────
# Live-progress toolkit — spinners, ticking countdowns, download bars.
# ─────────────────────────────────────────────────────────────────────
# Used by marrow_engine.py at the specific points that used to be
# completely silent for several seconds to several minutes at a time
# (rate-limit pauses, the actual AI request, video downloads, local
# Whisper transcription) — the single biggest source of "did this
# freeze?" during a run. Every helper here is safe to call unconditionally
# (they check for themselves whether it's safe to render anything).

_parallel_ctx = contextvars.ContextVar("marrow_parallel_run", default=False)


def set_parallel_mode(flag=True):
    """Call this from inside a --parallel worker thread (not the main
    thread) before it starts processing its video. Each worker thread
    gets its own contextvars.Context by default, so this only ever
    affects that one thread's own calls to live_ok() — other threads,
    and a non-parallel run, are unaffected.

    Needed because Rich allows only ONE live-updating region (spinner /
    progress bar) per Console at a time; several worker threads racing to
    each open their own would corrupt the terminal output. Parallel runs
    fall back to plain text feedback instead — still informative, just
    not animated."""
    _parallel_ctx.set(bool(flag))


def live_ok():
    """Whether it's currently safe to open a new spinner/progress bar:
    stdout must be a real terminal (not piped/redirected — animated
    output there just spams a log file with redraw junk) and this must
    not be a --parallel worker thread sharing the console with others."""
    try:
        return bool(console.is_terminal) and not _parallel_ctx.get()
    except Exception:
        return False


def _console_has_live():
    """Best-effort check for whether some OTHER Live display (a spinner
    opened elsewhere in the call stack — e.g. /ask's "Thinking…" status)
    is already active on this console. Rich's internal attribute for this
    has changed name across versions (_live_stack vs _live), so both are
    tried; if neither exists, we can't tell and rely entirely on
    _try_start()'s try/except instead."""
    try:
        stack = getattr(console, "_live_stack", None)
        if stack is not None:
            return len(stack) > 0
        return getattr(console, "_live", None) is not None
    except Exception:
        return False


def _try_start(live_obj):
    """Starts a Rich Live-based object (a Status or a Progress) as
    defensively as possible. Returns True if it actually started (safe to
    use), False if a live display is already active elsewhere (skip ours
    rather than fight over the terminal) or starting it failed for any
    other reason. Different Rich versions handle a second, nested Live
    differently — older ones raise LiveError, newer ones silently no-op —
    so this checks first AND catches broadly, covering both without
    needing to know which version is installed."""
    if _console_has_live():
        return False
    try:
        live_obj.start()
        return True
    except Exception:
        return False


def _safe_stop(live_obj):
    try:
        live_obj.stop()
    except Exception:
        pass


@contextmanager
def spinner(label, theme=None):
    """A themed spinner around one blocking call, e.g.:
        with ui.spinner("Calling Gemini…"):
            resp = requests.post(...)
    Falls back to doing nothing (just runs the code) when it's not safe
    to render one right now — never blocks, never raises."""
    theme = theme or resolve_theme()
    accent = THEMES.get(theme, THEMES[DEFAULT_THEME])["accent"]
    if not live_ok():
        yield
        return
    # A literal Text (not an f-string of markup) — `label` can be dynamic
    # (a model name, a quality string) and Rich's markup parser would
    # silently swallow any bracketed substring in it otherwise.
    status = console.status(Text(label, style=accent), spinner="dots")
    if _try_start(status):
        try:
            yield
        finally:
            _safe_stop(status)
    else:
        yield


def live_wait(seconds, label, theme=None):
    """Blocks for `seconds`, showing a live ticking countdown bar instead
    of a single static line that sits unchanged until the whole wait is
    over (which is what made rate-limit pauses and API-busy backoffs feel
    like MARROW had frozen). Falls back to one static line + a plain
    sleep when a countdown can't be shown right now."""
    theme = theme or resolve_theme()
    accent = THEMES.get(theme, THEMES[DEFAULT_THEME])["accent"]
    if seconds <= 0:
        return
    if not live_ok():
        console.print(Text(f"⏳ {label} — {seconds:.0f}s…", style=accent))
        time.sleep(seconds)
        return

    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

    progress = Progress(
        TextColumn("⏳ {task.description}", style=accent, markup=False),
        BarColumn(bar_width=18, complete_style=accent, finished_style=accent),
        TimeRemainingColumn(),
        console=console, transient=True,
    )
    if not _try_start(progress):
        console.print(Text(f"⏳ {label} — {seconds:.0f}s…", style=accent))
        time.sleep(seconds)
        return
    try:
        task = progress.add_task(label, total=seconds)
        step = 0.2
        elapsed = 0.0
        while elapsed < seconds:
            sl = min(step, seconds - elapsed)
            time.sleep(sl)
            elapsed += sl
            try:
                progress.update(task, completed=elapsed)
            except Exception:
                pass
    finally:
        _safe_stop(progress)


def progress_iter(iterable, total, label, theme=None, min_total=3):
    """Wraps a plain local loop (frame extraction, OCR passes, etc.) with
    a live 'N/M' progress bar. Falls back to iterating with no visual
    change at all — same as before this existed — when total is too
    small to be worth it or a bar can't be shown right now."""
    theme = theme or resolve_theme()
    accent = THEMES.get(theme, THEMES[DEFAULT_THEME])["accent"]
    if not live_ok() or total < min_total:
        for item in iterable:
            yield item
        return

    from rich.progress import Progress, BarColumn, TextColumn, MofNCompleteColumn, TimeElapsedColumn

    progress = Progress(
        TextColumn("{task.description}", style=accent, markup=False),
        BarColumn(bar_width=24, complete_style=accent, finished_style=accent),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console, transient=True,
    )
    if not _try_start(progress):
        for item in iterable:
            yield item
        return
    try:
        task = progress.add_task(label, total=total)
        for item in iterable:
            yield item
            try:
                progress.advance(task)
            except Exception:
                pass
    finally:
        _safe_stop(progress)


class DownloadProgress:
    """A live download bar (percent / size / speed / ETA) driven by
    yt-dlp's progress_hooks — replaces what used to be total silence for
    however long a video download took (yt-dlp runs with quiet=True so
    its own console output never appears). Usage:

        with ui.DownloadProgress("Downloading video") as dp:
            ydl_opts['progress_hooks'] = [dp.hook]
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

    Safe to use unconditionally — becomes a total no-op (hook does
    nothing, no bar shown) exactly when spinner()/live_wait() would."""

    def __init__(self, label="Downloading", theme=None):
        self.label = label
        self.theme = theme or resolve_theme()
        self._progress = None
        self._task = None

    def __enter__(self):
        if not live_ok():
            return self
        accent = THEMES.get(self.theme, THEMES[DEFAULT_THEME])["accent"]
        from rich.progress import (
            Progress, BarColumn, TextColumn, DownloadColumn,
            TransferSpeedColumn, TimeRemainingColumn,
        )
        progress = Progress(
            TextColumn("{task.description}", style=accent, markup=False),
            BarColumn(bar_width=22, complete_style=accent, finished_style=accent),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console, transient=True,
        )
        if _try_start(progress):
            self._progress = progress
            try:
                self._task = progress.add_task(self.label, total=None)
            except Exception:
                self._progress = None
        return self

    def hook(self, d):
        """Pass this straight to yt-dlp's progress_hooks list."""
        if self._progress is None:
            return
        try:
            status = d.get("status")
            if status == "downloading":
                total = d.get("total_bytes") or d.get("total_bytes_estimate")
                done = d.get("downloaded_bytes", 0)
                if total:
                    self._progress.update(self._task, total=total, completed=done)
                else:
                    self._progress.update(self._task, completed=done)
            elif status == "finished":
                self._progress.update(self._task, description=f"{self.label} — finalizing")
        except Exception:
            pass

    def __exit__(self, exc_type, exc, tb):
        if self._progress is not None:
            _safe_stop(self._progress)
        return False


def divider(label="", theme=None):
    """A themed horizontal rule — a nicer, less clip-arty stand-in for
    the old '=' * 56 ASCII banners. `label` is wrapped in a literal
    rich.text.Text before being handed to Rule() — Rule() markup-parses
    a plain str title by default, which silently drops anything that
    looks like a tag (e.g. a video title containing "[Full Course]"
    would lose that substring entirely), so a Text object (rendered
    as-is, no parsing) is used instead to stay safe on arbitrary
    dynamic content such as video titles."""
    from rich.rule import Rule
    theme = theme or resolve_theme()
    accent = THEMES.get(theme, THEMES[DEFAULT_THEME])["accent"]
    try:
        console.print(Rule(Text(label), style=accent, characters="─"))
    except Exception:
        console.print(f"── {label} " + "─" * 40)


def kv_panel(pairs, title=None, theme=None):
    """A bordered panel of aligned label/value rows — replacement for the
    old hand-padded '  Label:      value' plain-text block."""
    from rich.table import Table
    theme = theme or resolve_theme()
    accent = THEMES.get(theme, THEMES[DEFAULT_THEME])["accent"]
    table = Table.grid(padding=(0, 2, 0, 0))
    table.add_column(style=f"bold {accent}", no_wrap=True)
    table.add_column()
    for k, v in pairs:
        table.add_row(k, str(v))
    panel(table, title=title, theme=theme)


# ─────────────────────────────────────────────────────────────────────
# Auto-styled line printer for the engine's plain print() call sites —
# adds colour to the ~100 existing "[OK] ...", "[!] ...", "✗ ..." etc.
# messages throughout marrow_engine.py without having to touch every
# call site individually. Only ever styles a short, fixed-vocabulary
# PREFIX matched at the very start of the line — never the free-form
# text after it (which can contain a video title, an error message, etc.
# with its own brackets/punctuation) — so this can never misinterpret
# arbitrary dynamic content as a style tag the way raw Rich markup
# parsing on untrusted text could.
# ─────────────────────────────────────────────────────────────────────

_ENGINE_PREFIX_STYLES = [
    (re.compile(r'^\[OK\]'), "bold green"),
    (re.compile(r'^\[X\]'), "bold red"),
    (re.compile(r'^\[STOPPED\]'), "bold red"),
    (re.compile(r'^\[!\]'), "yellow"),
    (re.compile(r'^\[resume\]'), "cyan"),
    (re.compile(r'^\[global-cache\]'), "cyan"),
    (re.compile(r'^\[skip\]'), "dim"),
    (re.compile(r'^\[Progress\]'), "__accent_bold__"),
    (re.compile(r'^\[>\]'), "__accent_bold__"),
    (re.compile(r'^\[\d+/\d+\]'), "__accent_bold__"),
    (re.compile(r'^✓'), "bold green"),
    (re.compile(r'^✗'), "bold red"),
    (re.compile(r'^⚠'), "yellow"),
    (re.compile(r'^⏳'), "yellow"),
    (re.compile(r'^↻'), "magenta"),
    (re.compile(r'^💡'), "cyan"),
]


def print_engine_line(text, tag=None, theme=None):
    """Prints one line from the processing engine with light, safe
    styling: a dim [video_id] tag (only present under --parallel) plus
    colour on any recognized status prefix at the start of the message.
    Built from a plain rich.text.Text (never markup-parsed), so it can
    never crash or misrender on a video title, error string, or anything
    else with brackets/special characters in it."""
    theme = theme or resolve_theme()
    accent = THEMES.get(theme, THEMES[DEFAULT_THEME])["accent"]

    t = Text()
    if tag:
        t.append(f"[{tag}] ", style="dim")

    stripped = text.lstrip()
    indent_len = len(text) - len(stripped)
    style, match_end = None, 0
    for pattern, sty in _ENGINE_PREFIX_STYLES:
        m = pattern.match(stripped)
        if m:
            style = f"bold {accent}" if sty == "__accent_bold__" else sty
            match_end = m.end()
            break

    if style and match_end:
        t.append(text[:indent_len])
        t.append(text[indent_len:indent_len + match_end], style=style)
        t.append(text[indent_len + match_end:])
    else:
        t.append(text)

    try:
        console.print(t)
    except Exception:
        # A rendering hiccup here must never be the reason a video fails
        # to process — fall back to the plainest possible output.
        print(text if not tag else f"[{tag}] {text}")


def panel(body, title=None, theme=DEFAULT_THEME, style=None):
    accent = style or THEMES.get(theme, THEMES[DEFAULT_THEME])["accent"]
    console.print(Panel(body, title=title, border_style=accent, expand=False))


# ─────────────────────────────────────────────────────────────────────
# Bordered input + bottom status bar
# ─────────────────────────────────────────────────────────────────────

_PT_SESSION = None


def _get_pt_session():
    global _PT_SESSION
    if _PT_SESSION is None:
        from prompt_toolkit import PromptSession
        _PT_SESSION = PromptSession()
    return _PT_SESSION


def get_input(prompt_label="›", theme=DEFAULT_THEME, hint=None, show_status=True):
    """Draws a rounded box like the reference screenshot's input field, then
    reads a line. Falls back to plain input() when stdin/stdout isn't a
    real terminal (e.g. piped input, some IDEs).

    hint: an optional one-line instruction printed in dim text just above
    the box (e.g. "Paste a YouTube URL, ask a question, or type /help") —
    so the person always knows what's expected before they type, instead
    of guessing from a blank prompt.

    show_status: prints the current provider/model line above the box.
    This used to live in a prompt_toolkit `bottom_toolbar` pinned to the
    terminal's last row; that reserved-row redraw doesn't play well with
    plain print() output from the processing engine on some terminals
    (garbled/overlapping lines on Termux in particular), so it's now a
    normal line that scrolls with everything else."""
    accent = THEMES.get(theme, THEMES[DEFAULT_THEME])["accent"]

    if show_status:
        status_line(theme)
    if hint:
        console.print(f"[dim]{hint}[/dim]")

    if not sys.stdin.isatty():
        return input(f"{prompt_label} ")

    try:
        from prompt_toolkit.formatted_text import HTML

        width = min(console.width, 100)
        console.print(f"[{accent}]╭{'─' * (width - 2)}╮[/{accent}]")
        session = _get_pt_session()
        text = session.prompt(HTML(f"<style fg='{accent}'>{prompt_label} </style>"))
        console.print(f"[{accent}]╰{'─' * (width - 2)}╯[/{accent}]")
        return text
    except (EOFError, KeyboardInterrupt):
        raise
    except Exception:
        # Any prompt_toolkit environment quirk (unsupported terminal, etc.)
        # — degrade to a plain prompt rather than crash the whole tool.
        return input(f"{prompt_label} ")


def menu(options, title=None, theme=DEFAULT_THEME, cancel_label="cancel / back"):
    """Prints a numbered list of options and reads a validated choice.
    Returns the 0-based index of the chosen option, or None if the person
    cancelled (typed 0, blank, or an unrecognized value — always re-prompts
    once on an unrecognized value before giving up, rather than silently
    misinterpreting a typo as a different option).

    Always tells the person exactly what to type, so a menu never shows
    up as a bare '› ' with no explanation."""
    accent = THEMES.get(theme, THEMES[DEFAULT_THEME])["accent"]
    if title:
        console.print(f"\n[bold]{title}[/bold]")
    for i, label in enumerate(options, 1):
        console.print(f"  [{accent}]{i}.[/{accent}] {label}")
    console.print(f"  [dim]0. {cancel_label}[/dim]")

    for _ in range(2):  # one retry on an unrecognized value, then give up cleanly
        choice = input(f"\n  Type a number (1-{len(options)}, or 0 to {cancel_label}): ").strip()
        if choice in ("", "0"):
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return int(choice) - 1
        status_err(f"'{choice}' isn't one of the options — pick a number from 0 to {len(options)}.")
    return None
