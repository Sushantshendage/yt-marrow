"""
Tests for marrow_providers.py's rotation manager — the core value prop of
MARROW (per README_MARROW.md: "sticky-until-limited, rotate-on-429,
cross-provider wraparound, vision-only filtering all confirmed working").

No real network calls are made: _build_and_send is monkeypatched per test
to simulate success/rate-limit/invalid-key/generic-failure responses from
providers, which is what "scripted tests with mocked HTTP responses" in
the README's honest-limitations section should have shipped with.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import marrow_config as cfg
import marrow_providers as mrp


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    config_dir = tmp_path / ".marrow"
    monkeypatch.setattr(cfg, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(cfg, "CONFIG_FILE", config_dir / "config.json")
    yield config_dir


@pytest.fixture
def two_provider_setup(isolated_config):
    """gemini key with 2 models (1 vision, 1 not) + groq key with 1
    non-vision model — enough to exercise same-key rollover,
    cross-provider wraparound, and vision filtering."""
    monkeypatch_providers = {
        "gemini": {
            "display": "Google Gemini", "api_style": "gemini",
            "base_url": "https://example.invalid/gemini",
            "models": [
                {"id": "gemini-vision-model", "vision": True},
                {"id": "gemini-text-model", "vision": False},
            ],
        },
        "groq": {
            "display": "Groq", "api_style": "openai",
            "base_url": "https://example.invalid/groq",
            "models": [{"id": "groq-text-model", "vision": False}],
        },
    }
    cfg.add_key("gemini", "gemini-key-1", verified=True)
    cfg.add_key("groq", "groq-key-1", verified=True)
    return monkeypatch_providers


def _simple_contents(text="hello"):
    return [{"role": "user", "parts": [{"text": text}]}]


class TestFlatSlots:
    def test_single_key_expands_to_its_models_in_order(self, isolated_config, monkeypatch, two_provider_setup):
        monkeypatch.setattr(mrp, "PROVIDERS", two_provider_setup)
        cfg.remove_key(1)  # keep only the gemini key
        slots = mrp._flat_slots()
        assert [m["id"] for _, _, m in slots] == ["gemini-vision-model", "gemini-text-model"]

    def test_two_keys_expand_key1_then_key2(self, isolated_config, monkeypatch, two_provider_setup):
        monkeypatch.setattr(mrp, "PROVIDERS", two_provider_setup)
        slots = mrp._flat_slots()
        assert [(p, m["id"]) for _, p, m in slots] == [
            ("gemini", "gemini-vision-model"),
            ("gemini", "gemini-text-model"),
            ("groq", "groq-text-model"),
        ]

    def test_unverified_keys_are_skipped(self, isolated_config, monkeypatch, two_provider_setup):
        monkeypatch.setattr(mrp, "PROVIDERS", two_provider_setup)
        cfg.add_key("groq", "bad-groq-key", verified=False)
        slots = mrp._flat_slots()
        # still only 3 slots — the unverified 4th key contributes nothing
        assert len(slots) == 3

    def test_rotation_enabled_reflects_key_presence(self, isolated_config):
        assert mrp.rotation_enabled() is False
        cfg.add_key("gemini", "some-key")
        assert mrp.rotation_enabled() is True


class TestCallAiRotating:
    def test_no_keys_raises_clear_error(self, isolated_config):
        with pytest.raises(RuntimeError, match="No AI provider configured"):
            mrp.call_ai_rotating(_simple_contents())

    def test_sticky_until_limited_does_not_move_cursor_on_success(
        self, isolated_config, monkeypatch, two_provider_setup
    ):
        monkeypatch.setattr(mrp, "PROVIDERS", two_provider_setup)
        calls = []

        def fake_send(provider_id, model_id, api_key, turns, system_prompt, max_tokens, json_mode):
            calls.append((provider_id, model_id))
            return {"ok": True}

        monkeypatch.setattr(mrp, "_build_and_send", fake_send)

        for _ in range(5):
            result = mrp.call_ai_rotating(_simple_contents())
            assert result == {"ok": True}

        # Every call landed on the same (first) slot — no unnecessary rotation.
        assert calls == [("gemini", "gemini-vision-model")] * 5
        assert cfg.get_rotation_index() == 0

    def test_rate_limit_advances_to_next_model_on_same_key(
        self, isolated_config, monkeypatch, two_provider_setup
    ):
        monkeypatch.setattr(mrp, "PROVIDERS", two_provider_setup)
        calls = []

        def fake_send(provider_id, model_id, api_key, turns, system_prompt, max_tokens, json_mode):
            calls.append((provider_id, model_id))
            if model_id == "gemini-vision-model":
                raise mrp._RateLimited("HTTP 429")
            return {"ok": True}

        monkeypatch.setattr(mrp, "_build_and_send", fake_send)
        result = mrp.call_ai_rotating(_simple_contents())

        assert result == {"ok": True}
        assert calls == [("gemini", "gemini-vision-model"), ("gemini", "gemini-text-model")]
        assert cfg.is_cooling_down("gemini", "gemini-vision-model") is True

    def test_rate_limit_wraps_across_providers(self, isolated_config, monkeypatch, two_provider_setup):
        monkeypatch.setattr(mrp, "PROVIDERS", two_provider_setup)
        calls = []

        def fake_send(provider_id, model_id, api_key, turns, system_prompt, max_tokens, json_mode):
            calls.append((provider_id, model_id))
            if provider_id == "gemini":
                raise mrp._RateLimited("HTTP 429")
            return {"ok": True}

        monkeypatch.setattr(mrp, "_build_and_send", fake_send)
        result = mrp.call_ai_rotating(_simple_contents())

        assert result == {"ok": True}
        # Both gemini slots exhausted before falling through to groq.
        assert calls == [
            ("gemini", "gemini-vision-model"),
            ("gemini", "gemini-text-model"),
            ("groq", "groq-text-model"),
        ]

    def test_vision_requirement_skips_non_vision_models(
        self, isolated_config, monkeypatch, two_provider_setup
    ):
        monkeypatch.setattr(mrp, "PROVIDERS", two_provider_setup)
        calls = []

        def fake_send(provider_id, model_id, api_key, turns, system_prompt, max_tokens, json_mode):
            calls.append((provider_id, model_id))
            return {"ok": True}

        monkeypatch.setattr(mrp, "_build_and_send", fake_send)
        result = mrp.call_ai_rotating(_simple_contents(), require_vision=True)

        assert result == {"ok": True}
        # Only the vision-capable model was ever attempted.
        assert calls == [("gemini", "gemini-vision-model")]

    def test_no_vision_capable_model_available_raises_clear_error(
        self, isolated_config, monkeypatch, two_provider_setup
    ):
        monkeypatch.setattr(mrp, "PROVIDERS", two_provider_setup)
        cfg.remove_key(0)  # drop the only key with a vision-capable model
        with pytest.raises(RuntimeError, match="vision-capable"):
            mrp.call_ai_rotating(_simple_contents(), require_vision=True)

    def test_invalid_key_moves_on_without_permanently_disabling(
        self, isolated_config, monkeypatch, two_provider_setup
    ):
        monkeypatch.setattr(mrp, "PROVIDERS", two_provider_setup)
        calls = []

        def fake_send(provider_id, model_id, api_key, turns, system_prompt, max_tokens, json_mode):
            calls.append((provider_id, model_id))
            if model_id == "gemini-vision-model":
                raise mrp._InvalidKey("HTTP 401")
            return {"ok": True}

        monkeypatch.setattr(mrp, "_build_and_send", fake_send)
        result = mrp.call_ai_rotating(_simple_contents())

        assert result == {"ok": True}
        assert cfg.is_cooling_down("gemini", "gemini-vision-model") is True

    def test_all_slots_failing_raises_with_last_error(
        self, isolated_config, monkeypatch, two_provider_setup
    ):
        monkeypatch.setattr(mrp, "PROVIDERS", two_provider_setup)

        def fake_send(provider_id, model_id, api_key, turns, system_prompt, max_tokens, json_mode):
            raise RuntimeError(f"boom from {model_id}")

        monkeypatch.setattr(mrp, "_build_and_send", fake_send)
        with pytest.raises(RuntimeError, match="All configured providers/models failed"):
            mrp.call_ai_rotating(_simple_contents())

    def test_cooling_down_slots_are_skipped(self, isolated_config, monkeypatch, two_provider_setup):
        monkeypatch.setattr(mrp, "PROVIDERS", two_provider_setup)
        cfg.set_cooldown("gemini", "gemini-vision-model", 999)
        calls = []

        def fake_send(provider_id, model_id, api_key, turns, system_prompt, max_tokens, json_mode):
            calls.append((provider_id, model_id))
            return {"ok": True}

        monkeypatch.setattr(mrp, "_build_and_send", fake_send)
        mrp.call_ai_rotating(_simple_contents())
        assert calls[0] == ("gemini", "gemini-text-model")

    def test_budget_exhausted_raises_before_any_call(self, isolated_config, monkeypatch, two_provider_setup):
        monkeypatch.setattr(mrp, "PROVIDERS", two_provider_setup)

        class FakeBudget:
            def available(self):
                return False

        called = []
        monkeypatch.setattr(mrp, "_build_and_send", lambda *a, **k: called.append(1))
        with pytest.raises(RuntimeError, match="budget exhausted"):
            mrp.call_ai_rotating(_simple_contents(), budget=FakeBudget())
        assert not called
