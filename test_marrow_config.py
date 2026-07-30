"""
Tests for marrow_config.py.

These exist because the shipped README claimed rotation/config behavior
was "covered by scripted tests" while no tests/ directory was actually
included in the delivered zip — this file (and test_marrow_providers.py)
makes that claim true instead of aspirational.

Every test gets an isolated ~/.marrow via the `isolated_config` fixture,
so running the suite never touches a real user's actual config.json.
"""
import json
import os
import stat
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import marrow_config as cfg


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    """Points marrow_config at a throwaway directory for the duration of
    one test, so tests never read or write a real ~/.marrow."""
    config_dir = tmp_path / ".marrow"
    monkeypatch.setattr(cfg, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(cfg, "CONFIG_FILE", config_dir / "config.json")
    yield config_dir


def test_load_with_no_file_returns_defaults(isolated_config):
    c = cfg.load()
    assert c["keys"] == []
    assert c["theme"] == "aurora"
    assert c["rotation_index"] == 0
    assert c["cooldowns"] == {}


def test_save_then_load_roundtrips(isolated_config):
    c = cfg.load()
    c["theme"] = "mono"
    cfg.save(c)
    reloaded = cfg.load()
    assert reloaded["theme"] == "mono"


def test_load_survives_corrupted_json(isolated_config):
    isolated_config.mkdir(parents=True, exist_ok=True)
    (isolated_config / "config.json").write_text("{not valid json", encoding="utf-8")
    c = cfg.load()  # must not raise
    assert c["keys"] == []


def test_config_file_is_owner_only_permissions(isolated_config):
    if os.name != "posix":
        pytest.skip("POSIX permission bits only")
    cfg.add_key("gemini", "fake-key-123")
    mode = stat.S_IMODE(os.stat(cfg.CONFIG_FILE).st_mode)
    assert mode == 0o600, f"expected 0600, got {oct(mode)}"


def test_config_dir_is_owner_only_permissions(isolated_config):
    if os.name != "posix":
        pytest.skip("POSIX permission bits only")
    cfg.add_key("gemini", "fake-key-123")
    mode = stat.S_IMODE(os.stat(cfg.CONFIG_DIR).st_mode)
    assert mode == 0o700, f"expected 0700, got {oct(mode)}"


def test_no_world_readable_window_during_save(isolated_config, monkeypatch):
    """Regression test for the permission race: the temp file used for
    the atomic write must never be created with permissive default
    permissions and chmod'd afterward — it must be born at 0600."""
    if os.name != "posix":
        pytest.skip("POSIX permission bits only")
    seen_modes = []
    real_open = os.open

    def spying_open(path, flags, mode=0o777):
        fd = real_open(path, flags, mode)
        if str(path).endswith(".tmp"):
            seen_modes.append(stat.S_IMODE(os.fstat(fd).st_mode))
        return fd

    monkeypatch.setattr(os, "open", spying_open)
    cfg.add_key("gemini", "fake-key-123")
    assert seen_modes, "expected the tmp file path to be created via os.open"
    assert all(m == 0o600 for m in seen_modes), seen_modes


def test_add_key_and_list_keys(isolated_config):
    cfg.add_key("gemini", "key-a", verified=True, label="my gemini")
    cfg.add_key("openrouter", "key-b")
    keys = cfg.list_keys()
    assert len(keys) == 2
    assert keys[0]["provider"] == "gemini"
    assert keys[0]["label"] == "my gemini"
    assert keys[1]["provider"] == "openrouter"


def test_remove_key(isolated_config):
    cfg.add_key("gemini", "key-a")
    cfg.add_key("openrouter", "key-b")
    cfg.remove_key(0)
    keys = cfg.list_keys()
    assert len(keys) == 1
    assert keys[0]["provider"] == "openrouter"


def test_remove_key_out_of_range_is_noop(isolated_config):
    cfg.add_key("gemini", "key-a")
    cfg.remove_key(5)  # must not raise
    assert len(cfg.list_keys()) == 1


def test_reorder_keys(isolated_config):
    cfg.add_key("gemini", "key-a")
    cfg.add_key("openrouter", "key-b")
    cfg.add_key("groq", "key-c")
    cfg.reorder_keys([2, 0, 1])
    providers = [k["provider"] for k in cfg.list_keys()]
    assert providers == ["groq", "gemini", "openrouter"]


def test_reorder_keys_rejects_invalid_order(isolated_config):
    cfg.add_key("gemini", "key-a")
    cfg.add_key("openrouter", "key-b")
    with pytest.raises(ValueError):
        cfg.reorder_keys([0, 0])  # doesn't contain every index exactly once


def test_set_key_models_and_reset(isolated_config):
    cfg.add_key("gemini", "key-a")
    models = [{"id": "gemini-3.5-flash", "vision": True}]
    cfg.set_key_models(0, models)
    assert cfg.list_keys()[0]["enabled_models"] == models
    cfg.set_key_models(0, [])  # empty list resets to provider default
    assert "enabled_models" not in cfg.list_keys()[0]


def test_cooldown_set_and_check(isolated_config):
    assert cfg.is_cooling_down("gemini", "gemini-3.5-flash") is False
    cfg.set_cooldown("gemini", "gemini-3.5-flash", 60)
    assert cfg.is_cooling_down("gemini", "gemini-3.5-flash") is True


def test_cooldown_expires(isolated_config):
    cfg.set_cooldown("gemini", "gemini-3.5-flash", -1)  # already expired
    assert cfg.is_cooling_down("gemini", "gemini-3.5-flash") is False


def test_rotation_index_roundtrip(isolated_config):
    assert cfg.get_rotation_index() == 0
    cfg.set_rotation_index(4)
    assert cfg.get_rotation_index() == 4


def test_theme_roundtrip(isolated_config):
    assert cfg.get_theme() == "aurora"
    cfg.set_theme("ember")
    assert cfg.get_theme() == "ember"


def test_concurrent_add_key_from_many_threads_loses_nothing(isolated_config):
    """Regression test for the --parallel race: many threads hammering
    add_key() at once must not lose any writes and must never leave the
    config file corrupted (invalid JSON) at the end."""
    n_threads = 20
    errors = []

    def worker(i):
        try:
            cfg.add_key("gemini", f"key-{i}")
        except Exception as e:  # pragma: no cover - failure path
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    # File must be valid, parseable JSON with every key present.
    with open(cfg.CONFIG_FILE, encoding="utf-8") as f:
        data = json.load(f)
    assert len(data["keys"]) == n_threads
    labels = {k["key"] for k in data["keys"]}
    assert labels == {f"key-{i}" for i in range(n_threads)}


def test_concurrent_cooldown_writes_are_not_lost(isolated_config):
    """Same race, but on set_cooldown — the function most directly hit by
    concurrent call_ai_rotating() calls in --parallel mode."""
    n_threads = 15
    threads = [
        threading.Thread(target=cfg.set_cooldown, args=("provider", f"model-{i}", 60))
        for i in range(n_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    final = cfg.load()
    assert len(final["cooldowns"]) == n_threads
    for i in range(n_threads):
        assert f"provider::model-{i}" in final["cooldowns"]
