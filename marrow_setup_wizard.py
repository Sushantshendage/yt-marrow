"""
marrow_setup_wizard.py — the "connect your AI provider(s)" flow.

Runs automatically the very first time MARROW is launched on a machine
(no ~/.marrow/config.json yet, or one with zero keys in it) and is also
reachable any time afterwards via the /keys command. A person can add
as many keys as they want — same provider twice, five different
providers, whatever — every one of them becomes a slot in the rotation.
"""
import getpass

import marrow_config as cfg
import marrow_providers as mrp
import marrow_ui as ui


def _provider_menu():
    ui.console.print()
    for i, (pid, p) in enumerate(mrp.PROVIDERS.items(), 1):
        ui.console.print(f"  [bold]{i}.[/bold] {p['display']}")
    ui.console.print(f"  [dim]{len(mrp.PROVIDERS) + 1}. done adding keys[/dim]")
    ui.console.print()
    choice = input(f"  Pick a provider by number (1-{len(mrp.PROVIDERS)}), "
                    f"or press Enter to stop adding keys: ").strip()
    ids = list(mrp.PROVIDERS.keys())
    if choice == "":
        return None
    if choice.isdigit():
        n = int(choice)
        if n == len(ids) + 1:
            return None
        if 1 <= n <= len(ids):
            return ids[n - 1]
    return "__invalid__"


def _add_one_key(theme=None):
    theme = theme or cfg.get_theme()
    pid = _provider_menu()
    if pid is None:
        return False
    if pid == "__invalid__":
        ui.status_err("Didn't recognize that choice.")
        return True

    provider = mrp.PROVIDERS[pid]
    ui.console.print(f"\n  Get a key here: [underline]{provider['key_link']}[/underline]")
    try:
        key = getpass.getpass("  Paste your API key and press Enter (the characters stay hidden): ").strip()
    except Exception:
        key = input("  Paste your API key and press Enter: ").strip()

    if not key:
        ui.status_warn("Empty key, skipped.")
        return True

    with ui.console.status(f"[{theme}]Verifying with {provider['display']}…", spinner="dots"):
        ok, msg = mrp.verify_key(pid, key)

    if ok:
        ui.status_ok(f"{msg} — {provider['display']} is ready.")
    else:
        ui.status_err(msg)
        keep = input("  Save it anyway? MARROW will retry it later. (y/N): ").strip().lower()
        if keep != "y":
            ui.status_info("Not saved.")
            return True

    cfg.add_key(pid, key, verified=ok, label=provider["display"])

    if ok:
        pick = input("\n  Pick which models this key should use as fallbacks, from a live "
                      "list? (Y/n — Enter also means yes): ").strip().lower()
        if pick != "n":
            import marrow_settings as settings
            new_index = len(cfg.list_keys()) - 1
            settings.pick_models_for_key(theme, new_index)
    return True


def run_first_time_setup(theme=None):
    theme = theme or cfg.get_theme()
    ui.panel(
        "Let's connect at least one AI provider so MARROW can analyze videos.\n"
        "You can add more than one — MARROW automatically rotates to the next "
        "one if a provider's free-tier limit runs out.",
        title="[bold]Welcome to MARROW[/bold]",
        theme=theme,
    )
    while True:
        cont = _add_one_key(theme)
        if not cont:
            break
        more = input("\n  Add another key? (y/N): ").strip().lower()
        if more != "y":
            break

    keys = cfg.list_keys()
    if not keys:
        ui.status_warn(
            "No keys configured. MARROW will still start, but video analysis "
            "won't work until you run /keys."
        )
    else:
        verified = sum(1 for k in keys if k.get("verified"))
        ui.status_ok(f"{len(keys)} key(s) saved ({verified} verified). You can add more anytime with /keys.")
    ui.console.print()


def run_keys_command(arg="", theme=None):
    """Handles the /keys command inside the REPL: list, add, remove, verify."""
    theme = theme or cfg.get_theme()
    arg = arg.strip().lower()
    keys = cfg.list_keys()

    if arg in ("", "list"):
        if not keys:
            ui.status_info("No keys configured yet. Try: /keys add")
            return
        ui.console.print()
        for i, k in enumerate(keys):
            mark = "[green]✓ verified[/green]" if k.get("verified") else "[yellow]unverified[/yellow]"
            masked = k["key"][:6] + "…" + k["key"][-4:] if len(k["key"]) > 12 else "…"
            pool = k.get("enabled_models")
            pool_note = f"  [dim]({len(pool)} model(s) picked)[/dim]" if pool else ""
            ui.console.print(f"  [{i}] {k['label']}  {masked}  {mark}{pool_note}")
        ui.console.print("\n  /keys add       — connect another provider\n"
                          "  /keys remove N  — remove key number N\n"
                          "  /keys verify N  — re-check key number N\n"
                          "  /settings       — fallback models, key priority, theme\n")
        return

    if arg == "add":
        while _add_one_key(theme):
            more = input("\n  Add another key? (y/N): ").strip().lower()
            if more != "y":
                break
        return

    parts = arg.split()
    if len(parts) == 2 and parts[0] == "remove" and parts[1].isdigit():
        idx = int(parts[1])
        if 0 <= idx < len(keys):
            label = keys[idx]["label"]
            cfg.remove_key(idx)
            ui.status_ok(f"Removed {label}.")
        else:
            ui.status_err("No key with that number. Run /keys to see the list.")
        return

    if len(parts) == 2 and parts[0] == "verify" and parts[1].isdigit():
        idx = int(parts[1])
        if 0 <= idx < len(keys):
            k = keys[idx]
            with ui.console.status("Verifying…", spinner="dots"):
                ok, msg = mrp.verify_key(k["provider"], k["key"])
            all_keys = cfg.load()
            all_keys["keys"][idx]["verified"] = ok
            cfg.save(all_keys)
            (ui.status_ok if ok else ui.status_err)(msg)
        else:
            ui.status_err("No key with that number. Run /keys to see the list.")
        return

    ui.status_err("Didn't understand that. Try: /keys, /keys add, /keys remove N, /keys verify N, "
                  "or /settings for fallback models and key priority.")
