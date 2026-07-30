"""
marrow_providers.py — multi-provider AI backend with automatic rotation.

Supported providers (curated July 2026 — free/cheap tiers, at least one
vision-capable model each so MARROW's screenshot analysis keeps working
regardless of which provider is currently serving a request):

    gemini      Google Gemini            (native API)
    openrouter  OpenRouter                (OpenAI-compatible; one key unlocks
                                            ~20 different free models, several
                                            vision-capable — the deepest
                                            single-key rotation pool)
    groq        Groq                      (OpenAI-compatible; very fast LPU
                                            inference, generous free RPD)
    openai      OpenAI                    (OpenAI-compatible; paid, include
                                            for people who already have a key)
    anthropic   Anthropic Claude          (native API)
    mistral     Mistral                   (OpenAI-compatible; free tier)

Everything funnels through call_ai_rotating(), which is a drop-in
replacement for the old single-provider _call_gemini_with_fallback():
same input shape (Gemini-style `contents`), same output shape (parsed
JSON dict), same RuntimeError-on-failure contract — so marrow_engine.py
needed only a small hook, not a rewrite.

ROTATION RULE (as specified): stick with the current (key, model) slot
until it gets rate-limited, then advance to the next model on that same
key; once every model on that key is exhausted, advance to the next key
(which may be a different provider); once *every* key+model combination
in the whole list has been tried, wrap back around to the first key's
first model. The cursor is persisted in ~/.marrow/config.json so it
survives restarts instead of always hammering the same first provider.

TO ADD A NEW PROVIDER LATER: add one entry to PROVIDERS below with the
right api_style ("gemini" | "openai" | "anthropic") — if it's an
OpenAI-compatible endpoint (most new providers are), that's the entire
integration; the request/response adapters are shared.
"""
import base64
import json
import re
import time
import requests
from contextlib import nullcontext

import marrow_config as cfg

try:
    import marrow_ui as ui
except ImportError:
    ui = None  # this module still works standalone (plain, no spinner) without it

# ─────────────────────────────────────────────────────────────────────
# PROVIDER REGISTRY
# ─────────────────────────────────────────────────────────────────────

PROVIDERS = {
    "gemini": {
        "display": "Google Gemini",
        "api_style": "gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "key_link": "https://aistudio.google.com/apikey",
        "models": [
            {"id": "gemini-3.5-flash", "vision": True},
            {"id": "gemini-3.1-flash-lite", "vision": True},
            {"id": "gemini-2.5-flash", "vision": True},
        ],
    },
    "openrouter": {
        "display": "OpenRouter (many free models, 1 key)",
        "api_style": "openai",
        "base_url": "https://openrouter.ai/api/v1",
        "key_link": "https://openrouter.ai/keys",
        "models": [
            {"id": "google/gemma-4-31b-it:free", "vision": True},
            {"id": "google/gemma-4-26b-a4b-it:free", "vision": True},
            {"id": "nvidia/nemotron-3-super-120b-a12b:free", "vision": False},
            {"id": "openai/gpt-oss-20b:free", "vision": False},
            {"id": "meta-llama/llama-3.3-70b-instruct:free", "vision": False},
        ],
    },
    "groq": {
        "display": "Groq (fastest inference, generous free tier)",
        "api_style": "openai",
        "base_url": "https://api.groq.com/openai/v1",
        "key_link": "https://console.groq.com/keys",
        "models": [
            {"id": "openai/gpt-oss-20b", "vision": False, "context": 131072, "free": False},
            {"id": "openai/gpt-oss-120b", "vision": False, "context": 131072, "free": False},
            {"id": "llama-3.1-8b-instant", "vision": False, "context": 131072, "free": False},
            {"id": "llama-3.3-70b-versatile", "vision": False, "context": 131072, "free": False},
            {"id": "qwen/qwen3.6-27b", "vision": True, "context": 131072, "free": False},
        ],
    },
    "openai": {
        "display": "OpenAI",
        "api_style": "openai",
        "base_url": "https://api.openai.com/v1",
        "key_link": "https://platform.openai.com/api-keys",
        "models": [
            {"id": "gpt-4o-mini", "vision": True},
            {"id": "gpt-4.1-mini", "vision": True},
        ],
    },
    "anthropic": {
        "display": "Anthropic Claude",
        "api_style": "anthropic",
        "base_url": "https://api.anthropic.com/v1",
        "key_link": "https://console.anthropic.com/settings/keys",
        "models": [
            {"id": "claude-haiku-4-5-20251001", "vision": True},
        ],
    },
    "mistral": {
        "display": "Mistral",
        "api_style": "openai",
        "base_url": "https://api.mistral.ai/v1",
        "key_link": "https://console.mistral.ai/api-keys",
        "models": [
            {"id": "mistral-small-latest", "vision": True},
        ],
    },
}

RATE_LIMIT_COOLDOWN_SECONDS = 90   # how long we skip a throttled slot before retrying it


# ─────────────────────────────────────────────────────────────────────
# Gemini-shaped `contents` -> generic turns  (the one shape the rest of
# marrow_engine.py already speaks, so this is the only converter needed)
# ─────────────────────────────────────────────────────────────────────

def _generic_turns_from_gemini_contents(contents):
    """contents: [{"role": "user"/"model", "parts": [{"text":...} | {"inline_data":{mime_type,data}}]}]
    -> [{"role": "user"/"assistant", "parts": [{"type":"text","text":...} | {"type":"image","mime":...,"data":...}]}]
    """
    turns = []
    for turn in contents:
        role = "assistant" if turn.get("role") == "model" else "user"
        parts = []
        for p in turn.get("parts", []):
            if "text" in p:
                parts.append({"type": "text", "text": p["text"]})
            elif "inline_data" in p:
                d = p["inline_data"]
                parts.append({"type": "image", "mime": d.get("mime_type", "image/jpeg"), "data": d["data"]})
        turns.append({"role": role, "parts": parts})
    return turns


def _has_images(turns):
    return any(p["type"] == "image" for t in turns for p in t["parts"])


# ─────────────────────────────────────────────────────────────────────
# Per-style request builders + response parsers
# ─────────────────────────────────────────────────────────────────────

def _build_and_send(provider_id, model_id, api_key, turns, system_prompt, max_tokens, json_mode):
    style = PROVIDERS[provider_id]["api_style"]
    base = PROVIDERS[provider_id]["base_url"]
    label = f"Calling {PROVIDERS[provider_id]['display']} · {model_id}…"
    # ui.spinner() is a no-op when it's not safe to show one right now
    # (piped output, a --parallel worker thread, or — importantly — when
    # some OTHER spinner further up the call stack is already active,
    # e.g. /ask's "Thinking…" status) rather than crashing or fighting
    # over the terminal, so this is safe to wrap unconditionally.
    cm = ui.spinner(label) if ui is not None else nullcontext()
    with cm:
        if style == "gemini":
            return _send_gemini(base, model_id, api_key, turns, system_prompt, max_tokens, json_mode)
        if style == "openai":
            return _send_openai_style(base, model_id, api_key, turns, system_prompt, max_tokens, json_mode)
        if style == "anthropic":
            return _send_anthropic(base, model_id, api_key, turns, system_prompt, max_tokens, json_mode)
        raise RuntimeError(f"Unknown api_style for provider {provider_id}")


class _RateLimited(RuntimeError):
    pass


class _InvalidKey(RuntimeError):
    pass


def _extract_json(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        repaired = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", cleaned)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError as e:
            # last resort: grab the largest {...} or [...] span in the text
            m = re.search(r'(\{.*\}|\[.*\])', cleaned, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass
            raise RuntimeError(f"Invalid JSON response: {e}")


def _send_gemini(base, model, api_key, turns, system_prompt, max_tokens, json_mode):
    url = f"{base}/models/{model}:generateContent"
    contents = []
    for t in turns:
        parts = []
        for p in t["parts"]:
            if p["type"] == "text":
                parts.append({"text": p["text"]})
            else:
                parts.append({"inline_data": {"mime_type": p["mime"], "data": p["data"]}})
        contents.append({"role": "model" if t["role"] == "assistant" else "user", "parts": parts})
    body = {
        "contents": contents,
        "generationConfig": {"temperature": 0, "maxOutputTokens": max_tokens},
    }
    if json_mode:
        body["generationConfig"]["response_mime_type"] = "application/json"
    if system_prompt:
        body["system_instruction"] = {"parts": [{"text": system_prompt}]}

    resp = requests.post(url, headers={"x-goog-api-key": api_key}, json=body, timeout=180)
    _raise_for_status(resp)
    data = resp.json()
    candidate = (data.get("candidates") or [{}])[0]
    finish_reason = candidate.get("finishReason")
    parts = candidate.get("content", {}).get("parts", [])
    text = parts[0].get("text") if parts else None
    if not text:
        if finish_reason == "MAX_TOKENS":
            raise RuntimeError("Output truncated (MAX_TOKENS).")
        raise RuntimeError(f"No text in response (finishReason={finish_reason}).")
    return _extract_json(text) if json_mode else text


def _send_openai_style(base, model, api_key, turns, system_prompt, max_tokens, json_mode):
    url = f"{base}/chat/completions"
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for t in turns:
        if len(t["parts"]) == 1 and t["parts"][0]["type"] == "text":
            messages.append({"role": t["role"], "content": t["parts"][0]["text"]})
            continue
        content = []
        for p in t["parts"]:
            if p["type"] == "text":
                content.append({"type": "text", "text": p["text"]})
            else:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{p['mime']};base64,{p['data']}"},
                })
        messages.append({"role": t["role"], "content": content})

    body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    resp = requests.post(url, headers={"Authorization": f"Bearer {api_key}"}, json=body, timeout=180)
    if resp.status_code == 400 and json_mode:
        # Some free/open-source models on OpenRouter reject response_format —
        # retry once without it; our _extract_json already strips fences.
        body.pop("response_format", None)
        resp = requests.post(url, headers={"Authorization": f"Bearer {api_key}"}, json=body, timeout=180)
    _raise_for_status(resp)
    data = resp.json()
    choice = (data.get("choices") or [{}])[0]
    text = (choice.get("message") or {}).get("content")
    if not text:
        raise RuntimeError(f"No text in response (finish_reason={choice.get('finish_reason')}).")
    return _extract_json(text) if json_mode else text


def _send_anthropic(base, model, api_key, turns, system_prompt, max_tokens, json_mode):
    url = f"{base}/messages"
    messages = []
    for t in turns:
        content = []
        for p in t["parts"]:
            if p["type"] == "text":
                content.append({"type": "text", "text": p["text"]})
            else:
                content.append({"type": "image", "source": {"type": "base64", "media_type": p["mime"], "data": p["data"]}})
        messages.append({"role": t["role"], "content": content})

    body = {"model": model, "max_tokens": max_tokens, "temperature": 0, "messages": messages}
    if system_prompt:
        body["system"] = system_prompt
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
    resp = requests.post(url, headers=headers, json=body, timeout=180)
    _raise_for_status(resp)
    data = resp.json()
    blocks = data.get("content", [])
    text = next((b.get("text") for b in blocks if b.get("type") == "text"), None)
    if not text:
        raise RuntimeError(f"No text in response (stop_reason={data.get('stop_reason')}).")
    return _extract_json(text) if json_mode else text


def _raise_for_status(resp):
    if resp.status_code in (429, 503):
        raise _RateLimited(f"HTTP {resp.status_code}")
    if resp.status_code in (401, 403):
        raise _InvalidKey(f"HTTP {resp.status_code}: {resp.text[:200]}")
    if not resp.ok:
        raise RuntimeError(f"API error {resp.status_code}: {resp.text[:400]}")


# ─────────────────────────────────────────────────────────────────────
# Key verification — one cheap call per provider, used by the setup wizard
# and by /keys verify
# ─────────────────────────────────────────────────────────────────────

def verify_key(provider_id, api_key):
    """Returns (ok: bool, message: str)."""
    p = PROVIDERS.get(provider_id)
    if not p:
        return False, f"Unknown provider '{provider_id}'"
    style = p["api_style"]
    try:
        if style == "gemini":
            r = requests.get(f"{p['base_url']}/models", headers={"x-goog-api-key": api_key}, timeout=20)
        elif style == "openai":
            r = requests.get(f"{p['base_url']}/models", headers={"Authorization": f"Bearer {api_key}"}, timeout=20)
        elif style == "anthropic":
            r = requests.get(f"{p['base_url']}/models", headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"}, timeout=20)
        else:
            return False, "Unknown api_style"
    except requests.RequestException as e:
        return False, f"Network error: {e}"

    if r.status_code == 200:
        return True, "Key verified"
    if r.status_code in (401, 403):
        return False, "Key rejected (unauthorized) — double-check you copied it fully"
    if r.status_code == 404:
        # A couple of providers don't expose /models publicly on older SDKs —
        # not fatal, treat as "probably fine, will confirm on first real call".
        return True, "Key format accepted (this provider doesn't confirm at signup — verified on first use)"
    return False, f"Unexpected response: HTTP {r.status_code}"


# ─────────────────────────────────────────────────────────────────────
# Live model discovery — used by /settings to show what a key can
# *actually* reach right now, instead of only the curated PROVIDERS list
# above (which can go stale as providers rename/retire models).
# ─────────────────────────────────────────────────────────────────────

def _curated_lookup(provider_id):
    return {m["id"]: m for m in PROVIDERS.get(provider_id, {}).get("models", [])}


def list_live_models(provider_id, api_key, timeout=20):
    """Fetches the live list of models this specific key can reach, and
    enriches each with whatever pricing/context/vision metadata the
    provider's own API hands back — backfilled from the curated PROVIDERS
    table above for anything the live response leaves out.

    Returns (models, error):
      models — list of {"id", "vision": bool|None, "context": int|None,
                "free": bool|None, "known": bool} in the order the
                provider returned them.
      error  — None on success, or a short human-readable reason it
               fell back to the curated static list (network issue,
               unexpected response shape, etc.) — 'models' is never
               empty just because the live call failed, so the caller
               always has something to show.
    """
    provider = PROVIDERS.get(provider_id)
    if not provider:
        return [], f"Unknown provider '{provider_id}'"
    curated = _curated_lookup(provider_id)
    fallback = [dict(m) for m in provider["models"]]
    style = provider["api_style"]
    base = provider["base_url"]

    try:
        if style == "openai":
            r = requests.get(f"{base}/models",
                              headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout)
            r.raise_for_status()
            raw = r.json().get("data", [])
            out = []
            for m in raw:
                mid = m.get("id")
                if not mid:
                    continue
                known = curated.get(mid)
                context = m.get("context_length") or m.get("context_window") or (known or {}).get("context")
                arch = m.get("architecture") or {}
                modality = arch.get("modality") or ""
                input_mods = arch.get("input_modalities") or []
                if modality or input_mods:
                    vision = ("image" in modality) or ("image" in input_mods)
                else:
                    vision = (known or {}).get("vision")
                pricing = m.get("pricing")
                free = None
                if pricing:
                    try:
                        free = float(pricing.get("prompt", "0") or 0) == 0 and float(pricing.get("completion", "0") or 0) == 0
                    except (TypeError, ValueError):
                        free = None
                elif mid.endswith(":free"):
                    free = True
                elif known is not None:
                    free = known.get("free")
                out.append({"id": mid, "vision": vision, "context": context,
                            "free": free, "known": known is not None})
            return (out or fallback), None

        if style == "gemini":
            r = requests.get(f"{base}/models", headers={"x-goog-api-key": api_key}, timeout=timeout)
            r.raise_for_status()
            raw = r.json().get("models", [])
            out = []
            for m in raw:
                if "generateContent" not in m.get("supportedGenerationMethods", []):
                    continue  # skip embedding-only / non-chat models
                mid = (m.get("name") or "").split("/")[-1]
                if not mid:
                    continue
                known = curated.get(mid)
                context = m.get("inputTokenLimit") or (known or {}).get("context")
                vision = (known or {}).get("vision", True)  # Gemini chat models are multimodal by default
                free = (known or {}).get("free", "flash" in mid or None)
                out.append({"id": mid, "vision": vision, "context": context,
                            "free": free, "known": known is not None})
            return (out or fallback), None

        if style == "anthropic":
            r = requests.get(f"{base}/models",
                              headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
                              timeout=timeout)
            r.raise_for_status()
            raw = r.json().get("data", [])
            out = []
            for m in raw:
                mid = m.get("id")
                if not mid:
                    continue
                known = curated.get(mid)
                out.append({"id": mid, "vision": (known or {}).get("vision", True),
                            "context": (known or {}).get("context"), "free": False,
                            "known": known is not None})
            return (out or fallback), None

        return fallback, f"Unknown api_style '{style}'"

    except requests.RequestException as e:
        return fallback, f"Couldn't reach {provider['display']} ({e}) — showing the built-in list instead."
    except (ValueError, KeyError) as e:
        return fallback, f"Unexpected response from {provider['display']} ({e}) — showing the built-in list instead."


# ─────────────────────────────────────────────────────────────────────
# Rotation manager
# ─────────────────────────────────────────────────────────────────────

def _flat_slots():
    """[(key_entry, provider_id, model_dict), ...] in priority order —
    each key expands into its own model list (see below), then the next
    key, in the order keys are stored in config (/settings can reorder
    that list to change which key gets tried first).

    A key normally expands into every model curated for its provider, in
    the order declared in PROVIDERS. If the person picked a specific
    fallback pool for that key via /settings (stored as
    key_entry['enabled_models']), that hand-picked list + order is used
    instead."""
    slots = []
    for key_entry in cfg.list_keys():
        if key_entry.get("verified") is False:
            continue  # skip keys that have failed verification
        provider_id = key_entry["provider"]
        provider = PROVIDERS.get(provider_id)
        if not provider:
            continue
        chosen = key_entry.get("enabled_models")
        for model in (chosen if chosen else provider["models"]):
            slots.append((key_entry, provider_id, model))
    return slots


def rotation_enabled():
    return len(cfg.list_keys()) > 0


def current_status():
    """Human-readable 'provider/model' for the status bar, or None."""
    slots = _flat_slots()
    if not slots:
        return None
    idx = cfg.get_rotation_index() % len(slots)
    _, provider_id, model = slots[idx]
    return f"{PROVIDERS[provider_id]['display']} · {model['id']}"


def call_ai_rotating(contents, system_prompt=None, max_tokens=8192, budget=None,
                     json_mode=True, require_vision=False, on_switch=None):
    """Drop-in replacement for the old single-provider Gemini call.
    contents: Gemini-style turns (see _generic_turns_from_gemini_contents).
    on_switch: optional callback(provider_display, model_id) fired whenever
               rotation actually moves to a different slot than last time —
               used by the UI to print '↻ switched to Groq · llama-3.3-70b'.
    """
    if budget and not budget.available():
        raise RuntimeError("API budget exhausted")

    turns = _generic_turns_from_gemini_contents(contents)
    need_vision = require_vision or _has_images(turns)

    slots = _flat_slots()
    if not slots:
        raise RuntimeError(
            "No AI provider configured yet. Run /keys add to connect at least one."
        )

    n = len(slots)
    start_idx = cfg.get_rotation_index() % n
    last_err = None
    tried_any = False

    for step in range(n):
        idx = (start_idx + step) % n
        key_entry, provider_id, model = slots[idx]

        if need_vision and not model.get("vision"):
            continue
        if cfg.is_cooling_down(provider_id, model["id"]):
            continue

        tried_any = True
        try:
            result = _build_and_send(
                provider_id, model["id"], key_entry["key"], turns,
                system_prompt, max_tokens, json_mode,
            )
            if idx != cfg.get_rotation_index():
                cfg.set_rotation_index(idx)
            if on_switch and idx != start_idx:
                on_switch(PROVIDERS[provider_id]["display"], model["id"])
            if budget:
                budget.use()
            return result

        except _InvalidKey as e:
            last_err = e
            # Don't permanently disable on one bad response — some providers
            # 401 transiently. We just move on for this call.
            cfg.set_cooldown(provider_id, model["id"], 30)
            continue
        except _RateLimited as e:
            last_err = e
            cfg.set_cooldown(provider_id, model["id"], RATE_LIMIT_COOLDOWN_SECONDS)
            cfg.set_rotation_index((idx + 1) % n)
            continue
        except RuntimeError as e:
            last_err = e
            cfg.set_rotation_index((idx + 1) % n)
            continue

    if not tried_any:
        kind = "vision-capable " if need_vision else ""
        raise RuntimeError(
            f"No {kind}model available right now (all configured providers are "
            f"unverified or cooling down). Try /keys, or wait a minute and retry."
        )
    raise RuntimeError(f"All configured providers/models failed. Last error: {last_err}")
