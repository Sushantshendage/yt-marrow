# MARROW — changes in this pass (production-hardening)

This pass didn't add features — it's a full audit for production-readiness,
plus fixes for everything the audit found. Full writeup in
`PRODUCTION_READINESS.md`; short version below.

## Fixed
- **Version number was stuck at 1.0.0.** `CHANGELOG_v1.1.md` documented a
  full v1.1 feature set, but the `VERSION` constant in `marrow.py` was
  never bumped to match — which silently broke the "Updated vX → vY" note
  MARROW is supposed to show after an upgrade. Now `VERSION = "1.1.1"`.
- **Config corruption bug (found via the new test suite, not by
  inspection).** `marrow_config.load()`'s fallback path used `dict(DEFAULT_CONFIG)`
  — a *shallow* copy. Since `DEFAULT_CONFIG["keys"]` is a list and
  `DEFAULT_CONFIG["cooldowns"]` is a dict, every "no config file yet" or
  "config file corrupted" load shared those exact objects with the
  module-level default. Any later write mutated that shared default
  in place, permanently — so a failed first write, or a corrupted
  config recovery, could leak stale state into what should have been a
  clean slate for the rest of that process's life. Fixed with a real
  `copy.deepcopy`.
- **`--parallel` mode could corrupt rotation state.** Every video
  processed in parallel runs on its own thread, and each one calls into
  the AI rotation logic, which reads-modifies-writes `~/.marrow/config.json`
  (rotation index, cooldowns). None of that was locked, so two threads
  finishing at the same moment could stomp on each other's update — worst
  case, retrying a provider/model slot that's actually still rate-limited.
  Every mutating config helper now runs under a lock.
- **Small window where the API-key file was more permissive than
  intended.** The atomic-write temp file used to be created with default
  permissions and `chmod`'d to 0600 afterward — a brief gap where a
  partially-written file holding key material could be readable by more
  than just the owner. It's now created at 0600 from the moment it's
  opened, so there's no gap.
- **Leaked video-decoder handle on error.** The frame-extraction loop
  shares one `cv2.VideoCapture` handle across every screenshot in a
  video; it was only released on the happy path. An exception partway
  through the loop (a bad frame, a full disk) would leak that handle —
  on Windows in particular, a leaked handle can leave the video file
  locked, so the later "delete the downloaded video" cleanup step would
  quietly fail. Now wrapped in try/finally.
- **13 places that failed completely silently.** Best-effort cleanup and
  optional-metadata steps (temp file cleanup, cache write-through, stale
  cache eviction, etc.) caught and discarded every exception with no
  trace anywhere — by design, so one weird video wouldn't crash a whole
  run. That's still the right call, but "silent" used to mean *actually*
  silent: nothing printed, nothing logged, nowhere to look if the same
  thing kept failing. They now log to the new `~/.marrow/logs/marrow.log`
  (rotated, capped at a few MB) — console output is unchanged.
- **Unpinned dependencies.** `requirements.txt` had zero version
  constraints, so a `pip install` today vs. in six months could pull a
  breaking major release with no warning. Pinned with floors (and, for
  the two packages with an untested major bump sitting on PyPI right
  now, ceilings too) — except `yt-dlp`, which is deliberately left
  unbounded since keeping it current is one of the most common fixes
  for "downloads suddenly stopped working."

## New
- `marrow_logging.py` — the shared rotating file logger mentioned above.
- `tests/test_marrow_config.py`, `tests/test_marrow_providers.py` — 32
  tests, no real network calls, covering persistence, atomic-write
  permissions, the new concurrency lock, and the rotation manager
  (sticky-until-limited, rotate-on-429, cross-provider wraparound,
  vision-only filtering, cooldown skipping, invalid-key handling). The
  README previously *claimed* this coverage existed; now it actually
  does. Run with `pytest tests/`.
- A defense-in-depth check on `video_id` before it's used to build a
  filesystem path. In practice `video_id` always comes straight from
  yt-dlp's own metadata (already validated against YouTube), so this
  isn't fixing an observed exploit — it's making sure a future upstream
  change, corrupted cache entry, or hand-edited resume file can't put
  something like `../..` where a plain video ID is expected.
- `LICENSE` (MIT) — there wasn't one.

## Known trade-offs (not bugs — deliberate, and worth knowing about)
- The config lock added above is per-*process* (covers `--parallel`'s
  threads, which was the real, reachable race). It doesn't add
  cross-process file locking for two separate `marrow` invocations run
  in two terminals at once — a much rarer scenario, and the atomic write
  means the worst case is a lost update, not a corrupted file, so it
  self-heals on the next write.
- The first-run dependency auto-installer still installs without an
  explicit y/n prompt. It does clearly show what it's about to install
  first — this was a deliberate original design choice for a smooth
  first run, not an oversight, so it was left as-is rather than changed
  without being asked.
