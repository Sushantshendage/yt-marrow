"""
marrow_installer.py — makes 'first run on a brand new machine' painless.

Checks every third-party import MARROW's engine needs, installs anything
missing with pip, and checks for the one non-pip dependency (ffmpeg,
needed by yt-dlp for audio/video muxing). Every failure is caught and
shown as a plain-English message instead of a raw traceback, because the
one thing worse than "it doesn't work" is "it doesn't work and I can't
tell why."
"""
import importlib
import shutil
import subprocess
import sys

from rich.console import Console
from rich.panel import Panel

console = Console()

# (import name, pip package name)
REQUIRED = [
    ("requests", "requests"),
    ("PIL", "Pillow"),
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("markdown", "markdown"),
    ("pypdf", "pypdf"),
    ("reportlab", "reportlab"),
    ("youtube_transcript_api", "youtube-transcript-api"),
    ("yt_dlp", "yt-dlp"),
    ("rich", "rich"),
    ("prompt_toolkit", "prompt_toolkit"),
]


def _missing_packages():
    missing = []
    for import_name, pip_name in REQUIRED:
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(pip_name)
    return missing


def _pip_install(packages, theme):
    cmd = [sys.executable, "-m", "pip", "install", "--break-system-packages", "-q", *packages]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return False, "pip install timed out after 10 minutes — check your internet connection."
    except FileNotFoundError:
        return False, "Couldn't find pip. Is Python installed correctly (python -m ensurepip)?"
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-8:])
        return False, tail or "pip install failed with no error output."
    return True, None


def ensure_dependencies(theme="cyan"):
    """Call once at startup. Silently returns if everything's already
    installed; otherwise installs what's missing with visible progress.
    Returns True if MARROW is safe to proceed, False if something
    unrecoverable happened."""
    missing = _missing_packages()
    if not missing:
        return True

    console.print(Panel(
        f"First run on this machine — installing {len(missing)} missing "
        f"package(s):\n[dim]{', '.join(missing)}[/dim]",
        title="[bold]MARROW setup[/bold]", border_style=theme, expand=False,
    ))
    with console.status(f"[{theme}]Installing…", spinner="dots"):
        ok, err = _pip_install(missing, theme)

    if not ok:
        console.print(Panel(
            f"[bold red]Couldn't install automatically.[/bold red]\n\n"
            f"{err}\n\n"
            f"[dim]Try manually:[/dim] pip install --break-system-packages {' '.join(missing)}",
            title="[bold red]Setup failed[/bold red]", border_style="red", expand=False,
        ))
        return False

    still_missing = _missing_packages()
    if still_missing:
        console.print(Panel(
            f"[yellow]Installed, but Python still can't import:[/yellow] "
            f"{', '.join(still_missing)}\n"
            f"[dim]This usually means multiple Python installs are on this machine. "
            f"Try: {sys.executable} -m pip install --break-system-packages {' '.join(still_missing)}[/dim]",
            title="[bold yellow]Partial setup[/bold yellow]", border_style="yellow", expand=False,
        ))
        return False

    console.print(f"[{theme}]✓[/{theme}] All packages installed.\n")
    return True


def check_ffmpeg():
    """yt-dlp needs ffmpeg on PATH for merging separate video/audio
    streams. Not pip-installable, so we can only detect and explain."""
    if shutil.which("ffmpeg"):
        return True
    console.print(Panel(
        "[yellow]ffmpeg not found on PATH.[/yellow] Some videos (especially "
        "high-resolution ones) need it to merge video+audio after download.\n\n"
        "[bold]Install it:[/bold]\n"
        "  macOS:    brew install ffmpeg\n"
        "  Windows:  winget install ffmpeg   (or choco install ffmpeg)\n"
        "  Linux:    sudo apt install ffmpeg   (or your distro's package manager)\n\n"
        "[dim]MARROW will still run without it — some downloads may just fail "
        "until it's installed.[/dim]",
        title="[bold yellow]Optional dependency missing[/bold yellow]",
        border_style="yellow", expand=False,
    ))
    return False
