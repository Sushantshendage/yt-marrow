"""
marrow_logging.py — one shared, best-effort file logger for the whole tool.

Why this exists: several places in the pipeline (metadata fetch, channel
branding, screenshot cleanup, resume-state parsing, etc.) are intentionally
"best effort" — a failure there shouldn't crash a video that's otherwise
fine, so it's caught and swallowed. Before this module, "swallowed" meant
*completely* silent: nothing printed, nothing written anywhere, so a
person hitting a weird recurring failure had no way to find out why short
of adding print() statements themselves.

Now those spots log to ~/.marrow/logs/marrow.log (rotated, so it can't
grow unbounded) instead of vanishing into nothing. Console output is
unchanged — this never prints to stdout/stderr on its own, so it doesn't
clutter the CLI's UI. Nothing here ever raises: a logging failure must
never be the thing that crashes a video that was otherwise fine.
"""
import logging
import logging.handlers
import os
from pathlib import Path

_LOG_DIR = Path(os.environ.get("MARROW_HOME", str(Path.home() / ".marrow"))) / "logs"
_LOGGER = None


def get_logger():
    """Returns the shared MARROW logger, creating it (and its log dir) on
    first call. Falls back to a disabled (no-op) logger if the log
    directory can't be created for some reason (read-only filesystem,
    permissions, etc.) — logging must never be why MARROW crashes."""
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER

    logger = logging.getLogger("marrow")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            _LOG_DIR / "marrow.log", maxBytes=2_000_000, backupCount=3,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)
    except OSError:
        logger.addHandler(logging.NullHandler())

    _LOGGER = logger
    return logger
