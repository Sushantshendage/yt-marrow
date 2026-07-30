"""
marrow_settings.py — the /settings command.

Everything that changes *how* MARROW's AI rotation behaves lives here,
separate from marrow_setup_wizard.py (which only handles adding/removing/
verifying keys themselves):

  - which models are in the fallback pool for a given key, and the order
    they're tried in (picked from a LIVE list fetched from the provider,
    not just the hardcoded PROVIDERS table in marrow_providers.py)
  - which key gets tried first/second/etc. (key priority)
  - theme

Reachable any time via /settings. New keys added through the wizard also
get offered the model-picker from here right after they're verified, so
picking fallback models is part of the normal "add a key" flow, not a
separate thing you have to remember to go configure.
"""
import marrow_config as cfg
import marrow_providers as mrp
import marrow_ui as ui


# ─────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────

def _fmt_context(n):
    if not n:
        return "—"
    if n >= 1_000_000:
        v = n / 1_000_000
        return (f"{v:.1f}M").replace(".0M", "M")
    if n >= 1_000:
        return f"{round(n / 1_000)}K"
    return str(n)


def _fmt_bool(b):
    if b is True:
        return "Yes"
    if b is False:
        return "No"
    return "—"


def _fmt_cost(free):
    if free is True:
        return "Free"
    if free is False:
        return "Paid"
    return "—"


def _print_model_table(models, current_ids=None):
    from rich.table import Table
    current_ids = current_ids or set()
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2, 0, 0))
    table.add_column("#", justify="right")
    table.add_column("Model")
    table.add_column("Vision")
    table.add_column("Context")
    table.add_column("Cost")
    for i, m in enumerate(models, 1):
        mark = "✓ " if m["id"] in current_ids else "  "
        table.add_row(str(i), f"{mark}{m['id']}", _fmt_bool(m.get("vision")),
                      _fmt_context(m.get("context")), _fmt_cost(m.get("free")))
    ui.console.print(table)


# ─────────────────────────────────────────────────────────────────────
# Fallback-model picker for one key
# ─────────────────────────────────────────────────────────────────────

def pick_models_for_key(theme, index):
    keys = cfg.list_keys()
    if not (0 <= index < len(keys)):
        ui.status_err("No key with that number. Run /keys to see the list.")
        return
    key = keys[index]
    provider = mrp.PROVIDERS.get(key["provider"])
    if not provider:
        ui.status_err(f"Unknown provider '{key['provider']}' on this key.")
        return

    with ui.console.status(f"Fetching live model list from {provider['display']}…", spinner="dots"):
        models, err = mrp.list_live_models(key["provider"], key["key"])
    if err:
        ui.status_warn(err)
    if not models:
        ui.status_err(f"{provider['display']} didn't return any usable models for this key.")
        return

    current = key.get("enabled_models")
    current_ids = {m["id"] for m in current} if current else {m["id"] for m in models}

    ui.console.print(f"\n[bold]Fallback models for {key['label']}[/bold] "
                      f"([dim]✓ = currently enabled[/dim]):")
    _print_model_table(models, current_ids)

    ui.console.print(
        "\n  Type the numbers of the models MARROW should use for this key, in the\n"
        "  order you want them tried — e.g. \"1,3,2\" tries model 1 first, then 3,\n"
        "  then 2, skipping the rest. Type \"all\" to use every model shown above\n"
        "  in that order, or press Enter to leave this key's selection unchanged."
    )

    for _ in range(2):  # one retry on bad input, then bail out without changing anything
        raw = input("\n  Models to use (numbers, \"all\", or Enter to skip): ").strip().lower()
        if raw == "":
            ui.status_info("Left unchanged.")
            return
        if raw == "all":
            chosen = list(range(len(models)))
            break
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        try:
            idxs = [int(p) - 1 for p in parts]
        except ValueError:
            ui.status_err(f"'{raw}' isn't a valid list of numbers. Example: 1,3,2")
            continue
        if any(not (0 <= i < len(models)) for i in idxs):
            ui.status_err(f"Use numbers between 1 and {len(models)} only.")
            continue
        if not idxs:
            ui.status_err("Pick at least one model, or press Enter to leave it unchanged.")
            continue
        # de-dupe while keeping the person's first-mentioned priority order
        seen = set()
        chosen = [i for i in idxs if not (i in seen or seen.add(i))]
        break
    else:
        ui.status_warn("Couldn't make sense of that after a couple of tries — nothing changed.")
        return

    enabled = [{"id": models[i]["id"], "vision": bool(models[i].get("vision"))} for i in chosen]
    cfg.set_key_models(index, enabled)
    order_str = " → ".join(m["id"] for m in enabled)
    ui.status_ok(f"Saved for {key['label']}: {order_str}")


def _choose_key(theme, prompt_title):
    keys = cfg.list_keys()
    if not keys:
        ui.status_info("No keys configured yet. Try: /keys add")
        return None
    if len(keys) == 1:
        return 0
    labels = [f"{k['label']}  ({'✓ verified' if k.get('verified') else 'unverified'})" for k in keys]
    return ui.menu(labels, title=prompt_title, theme=theme)


# ─────────────────────────────────────────────────────────────────────
# Key priority (which key is tried first)
# ─────────────────────────────────────────────────────────────────────

def reorder_keys_flow(theme):
    keys = cfg.list_keys()
    if len(keys) < 2:
        ui.status_info("You need at least 2 keys before priority order matters. Try: /keys add")
        return

    ui.console.print("\n[bold]Current order (tried top to bottom):[/bold]")
    for i, k in enumerate(keys, 1):
        ui.console.print(f"  {i}. {k['label']}")

    ui.console.print(
        f"\n  Type the new order as numbers separated by commas, using every number\n"
        f"  1-{len(keys)} exactly once — e.g. \"2,1,3\" tries key 2 first, then key 1,\n"
        f"  then key 3. Press Enter to keep the current order."
    )

    for _ in range(2):
        raw = input("\n  New order: ").strip()
        if raw == "":
            ui.status_info("Left unchanged.")
            return
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        try:
            new_order = [int(p) - 1 for p in parts]
        except ValueError:
            ui.status_err(f"'{raw}' isn't a valid list of numbers. Example: 2,1,3")
            continue
        if sorted(new_order) != list(range(len(keys))):
            ui.status_err(f"Use every number from 1 to {len(keys)} exactly once.")
            continue
        cfg.reorder_keys(new_order)
        ui.status_ok("Key priority updated.")
        return
    ui.status_warn("Couldn't make sense of that after a couple of tries — nothing changed.")


# ─────────────────────────────────────────────────────────────────────
# Theme
# ─────────────────────────────────────────────────────────────────────

def _change_theme_flow(theme):
    names = list(ui.THEMES.keys())
    idx = ui.menu(names, title="Pick a theme:", theme=theme)
    if idx is None:
        return theme
    new_theme = names[idx]
    cfg.set_theme(new_theme)
    ui.console.clear()
    ui.render_banner(new_theme)
    ui.status_ok(f"Theme set to {new_theme}.")
    return new_theme


# ─────────────────────────────────────────────────────────────────────
# Top-level /settings menu
# ─────────────────────────────────────────────────────────────────────

def run_settings_command(theme):
    """Returns the (possibly changed) theme, so the caller's REPL loop
    stays in sync if the person changed it from in here."""
    while True:
        idx = ui.menu(
            [
                "Manage AI provider keys (add / remove / verify)",
                "Choose fallback models for a key",
                "Reorder key priority (which key is tried first)",
                "Change theme",
            ],
            title="MARROW settings — what would you like to change?",
            theme=theme,
            cancel_label="done, back to MARROW",
        )
        if idx is None:
            return theme

        if idx == 0:
            import marrow_setup_wizard as wizard
            wizard.run_keys_command("")
            more = input("\n  Add or remove a key? (add / remove N / verify N, or Enter to go back): ").strip()
            if more:
                wizard.run_keys_command(more)
        elif idx == 1:
            key_idx = _choose_key(theme, "Which key should this apply to?")
            if key_idx is not None:
                pick_models_for_key(theme, key_idx)
        elif idx == 2:
            reorder_keys_flow(theme)
        elif idx == 3:
            theme = _change_theme_flow(theme)
