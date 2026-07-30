# MARROW — production-readiness audit

**Verdict going in: not production-ready.** Solid architecture, but with a
few concrete, reproducible bugs and one unverified claim (tests that
didn't exist). **Verdict now: the concrete issues below are fixed and
tested.** This document is the paper trail — what was found, what was
changed, and what was deliberately left alone and why. Nothing in here is
rounded up; if something is a known trade-off rather than a fix, it says so.

This was a manual code audit (not a scanner run) across all ~6,600 lines,
followed by targeted fixes and a new test suite that exercises the fixes.
Everything below was independently confirmed: I ran the new tests (32,
all passing), installed the pinned dependencies into a clean virtualenv
to confirm they resolve together, and did a plain import of every module.

---

## What was already solid (no changes needed here)

Worth stating plainly, since an audit report that only lists problems is
misleading: this codebase does a lot of things right.

- No `eval`/`exec`, no `shell=True`, no `os.system` anywhere.
- All subprocess calls use argument lists, never shell strings — no
  injection surface even though URLs and titles flow through the pipeline.
- HTML report generation properly escapes everything via `html.escape` —
  a video title containing `<script>` tags can't execute in the generated
  report.
- API keys are read via `getpass` (hidden input), never echoed, never
  logged.
- Config writes were already atomic (temp file + `os.replace`) before this
  pass — the fixes below tightened the permission handling on top of that,
  they didn't have to introduce atomicity from scratch.
- Sensible, specific exception types in the provider layer
  (`_RateLimited`, `_InvalidKey` vs. generic failures) driving real retry
  logic, not one broad `except Exception`.
- `KeyboardInterrupt` handled at every relevant entry point.

## Fixed this pass

| # | Issue | Severity | Where |
|---|---|---|---|
| 1 | Config `load()` fallback did a **shallow** copy of `DEFAULT_CONFIG`, sharing its mutable `keys`/`cooldowns` objects across calls — any write could permanently corrupt the in-process default. Found by the new test suite, not by inspection. | High | `marrow_config.py` |
| 2 | `--parallel` mode's threads all read-modify-write `config.json` (rotation index, cooldowns) with no locking — races could silently corrupt rotation state under real usage of the parallel feature. | High | `marrow_config.py` |
| 3 | API-key config file briefly existed with default (non-0600) permissions before being `chmod`'d — a real, if narrow, local-exposure window. | Medium | `marrow_config.py` |
| 4 | Shared video-decoder handle only released on the happy path; an exception mid-extraction leaked it, which can block later cleanup of the downloaded video file (especially on Windows). | Medium | `marrow_engine.py` |
| 5 | `VERSION` constant stuck at `1.0.0` despite a full v1.1 changelog — silently broke the upgrade-notice feature. | Low (real bug, low impact) | `marrow.py` |
| 6 | 13 `except Exception: pass` blocks with zero logging anywhere in the tool — legitimate "don't crash the run" design, but genuinely undiagnosable if something kept failing. | Low/Medium (diagnosability) | `marrow_engine.py` |
| 7 | `requirements.txt` had no version constraints at all. | Medium | `requirements.txt` |
| 8 | README claimed rotation logic was "covered by scripted tests" — no `tests/` directory shipped. | Trust/honesty issue | (whole repo) |
| 9 | `video_id` used directly to build filesystem paths with no validation, even though it currently always comes from trusted yt-dlp metadata. | Low (defense in depth) | `marrow_engine.py` |
| 10 | No `LICENSE` file for something being handed to other users. | Low | (whole repo) |

Each fix is described in more detail, in the same voice as the project's
own changelog style, in `CHANGELOG_v1.1.1.md`.

### How the fixes were verified, not just written
- `pytest tests/` — 32 tests, 0 failures, no real network calls made.
  Includes a 20-thread concurrent-write stress test against the new
  config lock, and a permission-race regression test that spies on the
  exact `os.open` call used for the atomic write.
- `pip install -r requirements.txt` into a clean virtualenv — resolves
  and installs with no conflicts; every package's actual installed
  version was then imported successfully in one script (not just "pip
  said yes").
- Every module (`marrow_config`, `marrow_logging`, `marrow_providers`,
  `marrow_settings`, `marrow_ui`, `marrow_setup_wizard`,
  `marrow_installer`, `marrow_engine`) imports cleanly on its own.
- The new logger was exercised directly (not just code-reviewed) to
  confirm it actually writes to `~/.marrow/logs/marrow.log`.

## Deliberately left alone (trade-offs, not oversights)

- **Per-process, not cross-process, config locking.** The lock added for
  issue #2 covers `--parallel`'s threads (the actual, reachable race).
  It does not add OS-level file locking for two independent `marrow`
  processes running at the same time in two terminals. That's a rarer
  scenario, and because the write itself is atomic, the worst case is a
  lost update, not a corrupted file — it self-heals on the next write.
  Adding true cross-process locking (`fcntl` on POSIX, `msvcrt` on
  Windows) is a reasonable follow-up if that scenario ever matters to you,
  but it's meaningfully more platform-specific code for a rare case.
- **The first-run auto-installer still doesn't ask for y/n confirmation**
  before installing missing pip packages — it does show exactly what
  it's about to install first. That's how the original author designed
  it (smooth first run over an extra confirmation step), and changing an
  intentional UX decision without being asked isn't "fixing" it — so it
  was left as-is. Easy to add a prompt if you'd rather have one.
- **The provider list's model IDs/limits will drift over time** — this
  is an industry-wide moving target (see the README's own note on this),
  not something a one-time hardening pass can pin down permanently.

## What "production-ready" means here, concretely

For a local, single-user CLI tool like this (not a hosted multi-tenant
service), the bar that matters is: it doesn't lose or corrupt the user's
data, it doesn't leak secrets, it fails loudly enough to be debuggable,
and the claims in its own docs are true. All four now hold, backed by a
real test suite instead of a README claim.
