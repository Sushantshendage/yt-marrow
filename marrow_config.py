"""
marrow_config.py — persistent config for MARROW.

Everything MARROW remembers between runs lives in one JSON file:

    ~/.marrow/config.json

    {
      "theme": "aurora",
      "keys": [
        {"provider": "gemini", "key": "AIza...", "verified": true, "added": "2026-07-28T10:00:00"},
        {"provider": "openrouter", "key": "sk-or-...", "verified": true, "added": "..."},
        ...
      ],
      "rotation_index": 3,
      "cooldowns": {"gemini::gemini-3.5-flash": 1753600000.0}
    }

Design notes:
- A user can add *multiple keys for the same provider* (e.g. two Gemini
  keys from two Google accounts) — each is tracked separately so rotation
  can cycle through all of them, not just one per provider.
- Keys are stored in plain JSON with 0600 permissions (like ~/.ssh files,
  ~/.aws/credentials, gh's config.yml, etc.) — MARROW is a local CLI tool,
  not a networked service, so there's no separate secrets backend to hand
  it off to. The temp file used for the atomic write is created with 0600
  from the moment it's opened (not chmod'd after the fact), so there's no
  window where a partially-written file containing key material is
  readable by anyone but the owner.
- All read-modify-write helpers below (add_key, set_cooldown, etc.) run
  under a module-level lock. MARROW's `--parallel` mode processes several
  videos at once on separate threads, and every one of them can call the
  AI rotation logic concurrently — without this lock, two threads doing
  load() -> mutate -> save() at the same time can stomp on each other's
  update (e.g. one thread's cooldown write getting silently lost), which
  would let rotation retry a slot that's actually still rate-limited.
  This lock only protects against races *within one Python process*
  (which is what --parallel actually spawns — threads, not processes);
  it does not add cross-process file locking for two separate `marrow`
  invocations run at the same time in two terminals. That's a much
  rarer scenario and, worst case, self-heals on the next write.
- `cooldowns` remembers which (provider, model) pairs recently hit a rate
  limit and roughly when they'll likely be free again, so a fresh launch
  of MARROW doesn't immediately retry something that just got throttled.
"""
import copy
import json
import os
import stat
import threading
import time
from pathlib import Path

CONFIG_DIR = Path(os.environ.get("MARROW_HOME", str(Path.home() / ".marrow")))
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULT_CONFIG = {
    "theme": "aurora",
    "keys": [],            # list of {provider, key, verified, added, label}
    "rotation_index": 0,
    "cooldowns": {},       # "provider::model" -> unix timestamp when it's worth retrying
    "created": None,
}

# Guards every read-modify-write sequence below against concurrent threads
# (see module docstring). Re-entrant so a helper that internally calls
# another locked helper doesn't deadlock against itself.
_LOCK = threading.RLock()


def _secure_dir():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(CONFIG_DIR, stat.S_IRWXU)  # 0700, owner only
    except OSError:
        pass  # best-effort — e.g. some Windows filesystems don't support chmod bits


def _secure_file():
    try:
        os.chmod(CONFIG_FILE, stat.S_IRUSR | stat.S_IWUSR)  # 0600
    except OSError:
        pass


def exists():
    return CONFIG_FILE.exists()


def load():
    """Returns the config dict, or a fresh default dict if none exists yet."""
    with _LOCK:
        if not CONFIG_FILE.exists():
            cfg = copy.deepcopy(DEFAULT_CONFIG)
            cfg["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            return cfg
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except (json.JSONDecodeError, OSError):
            cfg = copy.deepcopy(DEFAULT_CONFIG)
        for k, v in DEFAULT_CONFIG.items():
            cfg.setdefault(k, copy.deepcopy(v))
        return cfg


def save(cfg):
    """Atomic write, same pattern the rest of the tool uses for caches.

    The temp file is opened with mode 0600 from the moment it's created
    (via os.open, before any content — including API keys — is written),
    instead of being chmod'd to 0600 only after the fact. That closes a
    small but real window where a newly-created file holding key material
    would briefly inherit the process umask (often world- or group-
    readable) before being locked down.
    """
    with _LOCK:
        _secure_dir()
        tmp = CONFIG_FILE.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        os.replace(tmp, CONFIG_FILE)
        _secure_file()  # belt-and-suspenders: re-assert 0600 on the final path too


def add_key(provider, key, verified=False, label=None):
    with _LOCK:
        cfg = load()
        cfg["keys"].append({
            "provider": provider,
            "key": key,
            "verified": verified,
            "label": label or provider,
            "added": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        save(cfg)
        return cfg


def remove_key(index):
    with _LOCK:
        cfg = load()
        if 0 <= index < len(cfg["keys"]):
            cfg["keys"].pop(index)
            save(cfg)
        return cfg


def list_keys():
    return load().get("keys", [])


def set_key_models(index, models):
    """Sets the fallback model pool + priority order for one key.
    models: list of {"id": ..., "vision": bool|None} dicts, in the exact
    order they should be tried. Pass an empty list to reset back to the
    provider's default (all curated models, in registry order)."""
    with _LOCK:
        cfg = load()
        if 0 <= index < len(cfg["keys"]):
            if models:
                cfg["keys"][index]["enabled_models"] = models
            else:
                cfg["keys"][index].pop("enabled_models", None)
            save(cfg)
        return cfg


def reorder_keys(new_order):
    """Changes which key gets tried first/second/etc. new_order is a list
    containing every current key index exactly once, in the desired new
    order — e.g. [2, 0, 1] moves key 2 to the front."""
    with _LOCK:
        cfg = load()
        keys = cfg["keys"]
        if sorted(new_order) != list(range(len(keys))):
            raise ValueError("new_order must contain every key index exactly once")
        cfg["keys"] = [keys[i] for i in new_order]
        save(cfg)
        return cfg


def set_cooldown(provider, model, seconds):
    with _LOCK:
        cfg = load()
        cfg.setdefault("cooldowns", {})[f"{provider}::{model}"] = time.time() + seconds
        save(cfg)


def is_cooling_down(provider, model):
    cfg = load()
    until = cfg.get("cooldowns", {}).get(f"{provider}::{model}")
    return bool(until and until > time.time())


def get_rotation_index():
    return load().get("rotation_index", 0)


def set_rotation_index(i):
    with _LOCK:
        cfg = load()
        cfg["rotation_index"] = i
        save(cfg)


def set_theme(name):
    with _LOCK:
        cfg = load()
        cfg["theme"] = name
        save(cfg)


def get_theme():
    return load().get("theme", "aurora")
