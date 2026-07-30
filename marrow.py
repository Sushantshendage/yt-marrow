#!/usr/bin/env python3
"""
marrow.py — MARROW's interactive shell.

    $ python marrow.py

Launches the chat-style REPL (banner, tips, bordered input, bottom status
bar — modeled on Gemini CLI's interface). Paste a YouTube URL to process
it; type a plain question to search your saved notes; use /commands for
everything else.

Old-school scripted use still works exactly as before — any arguments
are forwarded straight to the batch engine, unchanged:

    $ python marrow.py "https://youtube.com/watch?v=XXXX" --no-vision
"""
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import marrow_config as cfg
import marrow_ui as ui
from rich.text import Text

VERSION = "1.2.0"
ENGINE = str(Path(__file__).with_name("marrow_engine.py"))

HELP_TEXT = """\
[bold]Commands[/bold]
  /keys              manage AI provider API keys (add / remove / verify)
  /settings          fallback models per key, key priority, theme
  /library           list videos you've already processed
  /ask <question>    ask a question across your saved notes
  /theme <name>      ember (default) · aurora · mono
  /clear             clear the screen
  /version           show the installed MARROW version
  /help              this message
  /quit, /exit       leave MARROW

[bold]Processing a video[/bold]
  Just paste a YouTube URL (video or playlist). Any of the classic flags
  still work if you type them after it, e.g.:
    https://youtube.com/watch?v=XXXX --no-vision --quiz
  Full flag list: /raw --help

[dim]Note: the numbered tips shown at startup are examples, not a menu —
typing "1" is treated as a search over your notes, same as anything else
that isn't a URL or a /command.[/dim]
"""

MAIN_HINT = "Paste a YouTube URL for notes, ask a question about your saved notes, or type /help"


def _looks_like_url(text):
    t = text.strip().lower()
    return t.startswith("http://") or t.startswith("https://") or "youtube.com" in t or "youtu.be" in t


def _run_engine(argv_str, theme):
    ui.divider(" Processing ", theme)
    ui.status_info("Handing off to the processing engine…", theme)
    ui.console.print()
    started = time.time()
    try:
        args = shlex.split(argv_str)
        if "--open" not in args:
            args.append("--open")
        # MARROW_THEME lets the engine (a separate process) render its own
        # spinners/progress bars/output in the same colour the person
        # picked in the shell — without this, processing would visually
        # drop back to plain defaults the moment it started, which was
        # part of why it used to feel like a different, less-finished
        # tool once a video was actually running.
        env = os.environ.copy()
        env["MARROW_THEME"] = theme
        result = subprocess.run([sys.executable, ENGINE, *args], env=env)
        elapsed = time.time() - started
        m, s = divmod(int(elapsed), 60)
        elapsed_str = f"{m}m {s}s" if m else f"{s}s"
        ui.console.print()
        if result.returncode == 0:
            ui.status_ok(f"Done in {elapsed_str}.")
        else:
            ui.status_warn(f"Engine exited with code {result.returncode} after {elapsed_str} "
                            f"(see output above for details).")
    except FileNotFoundError:
        ui.status_err(f"Couldn't find the engine at {ENGINE}.")
    except KeyboardInterrupt:
        ui.console.print()
        ui.status_warn("Cancelled.")


def _cmd_library(theme):
    import marrow_engine as engine
    from rich.table import Table

    library_dir = cfg.load().get("library_dir") or engine.DEFAULT_GLOBAL_CACHE_DIR
    with ui.console.status("Scanning your library…", spinner="dots"):
        records = engine.scan_library(library_dir)
    if not records:
        ui.status_info(f"No processed videos found yet in {library_dir}.")
        return

    videos = {}
    for r in records:
        key = r.get("source_url") or r.get("title")
        videos.setdefault(key, r)
    videos = list(videos.values())

    accent = ui.THEMES.get(theme, ui.THEMES[ui.DEFAULT_THEME])["accent"]
    table = Table(
        title=f"{len(videos)} video(s) in your library",
        caption=str(library_dir),
        border_style=accent, header_style=f"bold {accent}",
        show_lines=False, expand=False,
    )
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("Title")
    table.add_column("Reports", style="dim")
    for i, v in enumerate(videos, 1):
        files = [label for label, field in (("HTML", "rel_html"), ("PDF", "rel_pdf"), ("MD", "rel_md"))
                 if v.get(field)]
        files_str = ", ".join(files) if files else "none on disk"
        # Text(), not an f-string through console.print — a video title is
        # untrusted dynamic content and Rich's markup parser would
        # silently drop anything in it that looks like a tag (e.g. a
        # title containing "[Full Course]").
        table.add_row(str(i), Text(v.get('title') or 'untitled'), files_str)
    ui.console.print()
    ui.console.print(table)

    choice = input(f"\n  Type a number (1-{len(videos)}) to open that video's report, "
                    f"or press Enter to go back: ").strip()
    if not choice:
        return
    if not (choice.isdigit() and 1 <= int(choice) <= len(videos)):
        ui.status_err(f"'{choice}' isn't one of the videos listed — pick a number from 1 to {len(videos)}.")
        return

    v = videos[int(choice) - 1]
    video_dir = Path(v["video_dir"])
    candidates = [(label, video_dir / fname) for label, field, fname in (
        ("HTML (interactive, recommended)", "rel_html", "report.html"),
        ("PDF", "rel_pdf", "report.pdf"),
        ("Markdown (plain text)", "rel_md", "report.md"),
    ) if v.get(field)]

    if not candidates:
        ui.status_err("No report files found on disk for this video.")
        return

    if len(candidates) == 1:
        target = candidates[0][1]
    else:
        idx = ui.menu([label for label, _ in candidates],
                       title=f"Open '{(v.get('title') or '')[:50]}' as:", theme=theme)
        if idx is None:
            return
        target = candidates[idx][1]

    if engine.open_file_for_viewing(target):
        ui.status_ok(f"Opened {target.name}.")
    else:
        ui.status_warn(f"Couldn't auto-open — open this path yourself: {target}")


def _cmd_ask(question, theme):
    import marrow_providers as mrp
    import marrow_engine as engine

    if not mrp.rotation_enabled():
        ui.status_err("No AI provider configured yet. Run /keys first.")
        return
    library_dir = cfg.load().get("library_dir") or engine.DEFAULT_GLOBAL_CACHE_DIR
    with ui.console.status("Searching your notes…", spinner="dots"):
        records = engine.scan_library(library_dir)
    if not records:
        ui.status_info("No processed videos found yet — process one first.")
        return
    budget = engine.APIBudget(max_calls=None)
    try:
        with ui.console.status("Thinking…", spinner="dots"):
            found, answer, sources = engine.answer_question(question, records, "rotation", budget)
    except RuntimeError as e:
        ui.status_err(str(e))
        return
    ui.console.print()
    if not found or not answer:
        ui.status_info("Not covered in your saved notes.")
        return
    # Text(), not a bare string through console.print() — this is
    # AI-generated text and almost certain to contain square brackets at
    # some point (citations like "[1]", asides like "[note]", markdown-
    # style links); Rich's default markup parsing would silently drop
    # anything it doesn't recognize as a valid tag.
    ui.console.print(Text(answer))
    if sources:
        ui.console.print(Text("\nSources:", style="dim"))
        for s in sources:
            m, sec = divmod(int(s.get("start_seconds", 0)), 60)
            ui.console.print(Text(f"  - {s.get('title', '')} @ {m:02d}:{sec:02d}", style="dim"))
    ui.console.print()


def _open_latest_report(theme):
    """Finds and opens the most-recently-produced report — a manual
    backstop for when auto-open (--open, on by default) couldn't find a
    working opener in this particular environment."""
    import marrow_engine as engine
    library_dir = cfg.load().get("library_dir") or engine.DEFAULT_GLOBAL_CACHE_DIR
    base = Path(library_dir)
    info_files = list(base.rglob("video_info.json")) if base.exists() else []
    if not info_files:
        ui.status_info("Nothing processed yet.")
        return
    latest_dir = max(info_files, key=lambda p: p.stat().st_mtime).parent
    for fname in ("report.html", "report.pdf", "report.md"):
        target = latest_dir / fname
        if target.exists():
            if engine.open_file_for_viewing(target):
                ui.status_ok(f"Opened {target.name}.")
            else:
                ui.status_warn(f"Couldn't auto-open — open this path yourself: {target}")
            return
    ui.status_err("No report files found for the most recent video.")


def _post_process_menu(theme):
    """Shown right after any video run (URL or /raw) finishes, so the
    person always has an explicit next step instead of a bare prompt.
    Loops until they choose to go back to the main prompt — 'process
    another video' stays in this loop rather than dropping back out, so
    a back-to-back batch of videos doesn't need re-navigating each time.
    Returns the (possibly changed, via Settings) theme."""
    while True:
        idx = ui.menu(
            [
                "Open the report you just made",
                "Process another video",
                "Ask a question about your notes",
                "Open your library",
                "Settings (fallback models, key priority, theme)",
            ],
            title="What next?",
            theme=theme,
            cancel_label="back to the main prompt",
        )
        if idx is None:
            return theme
        if idx == 0:
            _open_latest_report(theme)
            continue
        if idx == 1:
            url = input("\n  Paste the YouTube URL to process (any classic flags "
                        "work after it too): ").strip()
            if url:
                _run_engine(url, theme)
            continue
        if idx == 2:
            q = input("\n  What do you want to ask about your saved notes? ").strip()
            if q:
                _cmd_ask(q, theme)
            continue
        if idx == 3:
            _cmd_library(theme)
            continue
        if idx == 4:
            import marrow_settings as settings
            theme = settings.run_settings_command(theme)
            continue


def repl():
    theme = cfg.get_theme()
    ui.console.clear()
    ui.render_banner(theme)
    ui.render_tips(theme)

    seen = cfg.load().get("last_version_seen")
    if seen != VERSION:
        c = cfg.load()
        c["last_version_seen"] = VERSION
        cfg.save(c)
        if seen:  # None means this is a first install, not an upgrade — no banner needed
            ui.panel(f"Updated {seen} → {VERSION}. Your keys and library are unchanged.",
                     title="[bold]MARROW upgraded[/bold]", theme=theme)

    if not cfg.list_keys():
        import marrow_setup_wizard as wizard
        wizard.run_first_time_setup(theme)

    while True:
        try:
            line = ui.get_input("›", theme, hint=MAIN_HINT)
        except (EOFError, KeyboardInterrupt):
            ui.console.print("\nBye!")
            break

        line = line.strip()
        if not line:
            continue

        if line in ("/quit", "/exit", "quit", "exit"):
            ui.console.print("Bye!")
            break
        if line == "/version":
            ui.status_info(f"MARROW v{VERSION}", theme)
            continue
        if line == "/help":
            ui.panel(HELP_TEXT, title="MARROW help", theme=theme)
            continue
        if line == "/clear":
            ui.console.clear()
            ui.render_banner(theme)
            continue
        if line.startswith("/theme"):
            parts = line.split(maxsplit=1)
            name = parts[1].strip() if len(parts) > 1 else ""
            if name in ui.THEMES:
                cfg.set_theme(name)
                theme = name
                ui.console.clear()
                ui.render_banner(theme)
                ui.status_ok(f"Theme set to {name}.")
            else:
                ui.status_err(f"Usage: /theme <name> — choices: {', '.join(ui.THEMES)}")
            continue
        if line.startswith("/settings"):
            import marrow_settings as settings
            theme = settings.run_settings_command(theme)
            continue
        if line.startswith("/keys"):
            import marrow_setup_wizard as wizard
            wizard.run_keys_command(line[len("/keys"):], theme)
            continue
        if line == "/library":
            _cmd_library(theme)
            continue
        if line.startswith("/ask"):
            q = line[len("/ask"):].strip()
            if not q:
                ui.status_err("Usage: /ask <your question>")
                continue
            _cmd_ask(q, theme)
            continue
        if line.startswith("/raw"):
            _run_engine(line[len("/raw"):].strip(), theme)
            theme = _post_process_menu(theme)
            continue

        if _looks_like_url(line):
            _run_engine(line, theme)
            theme = _post_process_menu(theme)
            continue

        # Plain text that isn't a command or URL — treat it like a chat
        # message: search the library, same as /ask.
        _cmd_ask(line, theme)


def main():
    if len(sys.argv) > 1:
        # Scripted/batch use — forward straight to the engine, unchanged.
        import marrow_installer
        if not marrow_installer.ensure_dependencies():
            sys.exit(1)
        result = subprocess.run([sys.executable, ENGINE, *sys.argv[1:]])
        sys.exit(result.returncode)

    import marrow_installer
    if not marrow_installer.ensure_dependencies():
        ui.status_err("Setup couldn't finish — fix the error above and re-run marrow.py.")
        sys.exit(1)
    marrow_installer.check_ffmpeg()

    try:
        repl()
    except KeyboardInterrupt:
        print("\nBye!")


if __name__ == "__main__":
    main()
