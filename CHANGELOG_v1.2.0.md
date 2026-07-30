# MARROW — changes in this pass (screenshot-loss bug fix + live progress UI)

Two independent things prompted this pass: long videos were losing most of
their requested screenshots before a report was ever built, and the CLI went
completely silent (no output at all, for anywhere from seconds to minutes)
at exactly the points where something was actually happening — a video
downloading, an AI call waiting on rate limits, local Whisper transcribing.
Both are fixed below.

## Fixed — the screenshot-loss bug

- **`deduplicate_frames()` was comparing frames with no time limit at all.**
  It walked every extracted screenshot in timestamp order and compared each
  one only to whichever frame it had most recently decided to keep — with
  no cap on how far apart in time two compared frames could be. Screenshot
  timestamps aren't evenly spaced; they're specific moments (an AI-picked
  highlight, a detected slide transition) that can be seconds or many
  *minutes* apart. On any video where slides share one template, or a code
  editor keeps the same theme, or a whiteboard keeps the same framing —
  which is most videos — two screenshots from completely different parts
  of the video, discussing completely different things, would score as
  "the same shot" and the later one was silently deleted before Vision
  analysis ever got to see it. Measured on a synthetic same-template,
  different-content pair: **99.65% similar** under the old metric — a
  hair below a same-image score of 100%, nowhere near enough margin to
  tell them apart. This was very likely the direct cause of "asked for 26
  screenshots, only got a handful back." Fixed by only ever comparing (and
  only ever discarding as a duplicate) frames that land within 8 seconds
  of each other — comfortably enough to still catch its intended case (the
  same slide held on screen long enough that 2-3 timestamps landed on it),
  while never again touching two screenshots from different discussions
  just because they look alike.
- **The similarity metric itself was also upgraded**, from a plain whole-
  frame mean-pixel-difference to windowed SSIM (structural similarity,
  computed via `cv2.boxFilter` local statistics — no new dependency). SSIM
  responds to local structural change (added text, a moved diagram) even
  when it's a small part of the frame, instead of averaging it away against
  a shared background. On the same synthetic pair, this widened the gap
  between "identical" and "different content, same template" from 0.35
  percentage points to roughly 12 — real room for a threshold to work with.
  (The old code's docstring already claimed to use SSIM; it didn't. Now it
  actually does.)
- **Vision-batch results are now matched to screenshots by an explicit
  index the model returns, not by list position**, and any screenshot
  missing from a batch's response — because the model returned fewer
  objects than images sent, merged two into one, or the reply got cut off —
  is automatically retried on its own before the run moves on, so a partial
  or slightly-off AI response no longer means that screenshot silently
  never appears in the final report.

Both were verified with synthetic reproductions in `tests/` — a same-
template/different-content set of 26 screenshots spread across a 40-minute
span (mirroring the reported symptom) all survive `deduplicate_frames()`
now, while a genuine 3-frame "same slide held on screen" duplicate still
correctly collapses to 1.

## New — live progress feedback during processing

Several points in the pipeline used to go completely silent for however
long they took — which, combined with the fix above no longer being
needed as a red herring, was the other half of "did this freeze?":

- **Video downloads** ran with yt-dlp's `quiet: True` and produced zero
  output for the entire download, often the single longest step in the
  whole run. Now shows a live bar (percent / size / speed / ETA).
- **The rate limiter between AI calls** printed one static line then
  blocked on a silent `sleep()` — now a live ticking countdown.
- **AI calls themselves** (vision batches, transcript analysis, `/ask`)
  now show a themed spinner for the duration of the actual network
  request, instead of the terminal just sitting there.
- **Local Whisper transcription, frame extraction, and local OCR** — all
  can run for a while on longer videos with zero prior feedback — now show
  a spinner or a live N/M progress bar.
- All of the above degrade gracefully to plain text (or nothing, matching
  the previous behavior exactly) when stdout isn't a real terminal, and
  are automatically disabled inside `--parallel` worker threads, which
  share one terminal and can't each animate their own widget at once.
  None of this can ever crash a run: every live widget is wrapped so that
  a rendering hiccup — or another spinner already active further up the
  call stack (e.g. `/ask`'s "Thinking…" status) — just silently skips its
  own display rather than erroring out or corrupting the terminal.
- The shell now hands its color theme to the processing engine (a separate
  process) via an environment variable, so a video processed from inside
  the shell keeps looking like the same tool instead of dropping to plain
  defaults the moment processing starts. It also now reports how long a
  run took ("Done in 4m 12s").
- `/library` is now a proper bordered table instead of hand-padded plain
  text lines.

## Fixed — a markup-injection bug found while building the above

- Several places printed dynamic, untrusted text (video titles, `/ask`'s
  AI-generated answers) straight through Rich's `console.print()`/`Rule()`/
  `console.status()`, all of which parse `[...]` in plain strings as style
  tags by default. A title or answer containing something that merely
  *looked* like a tag — `"Learn Python [FULL COURSE]"`, an AI answer with
  a citation like `[1]` or an aside like `[note]` — could have that exact
  substring **silently deleted** from what got printed (confirmed with a
  reproduction: `"...weird[not a real tag] video title"` rendered as
  `"...weird video title"`, no error, no warning). Separately, building a
  progress bar's label from an f-string containing a literal `{` or `}`
  (plausible in a quality string, a model name) hit Python's own
  `str.format()` parsing underneath and would raise `KeyError` outright.
  Fixed everywhere found by constructing a literal `rich.text.Text` (or
  using Rich's `{task.description}` placeholder pattern) for any dynamic
  content instead of interpolating it into a markup or format template.

## Known trade-offs (deliberate, worth knowing about)

- The 8-second window and SSIM threshold in `deduplicate_frames()` are
  reasonable defaults, not values tuned against a large real-video corpus
  (no network access from this sandbox to pull test videos) — if you find
  a video where local dedup still misbehaves, that function is the first
  place to look, and both numbers are named parameters.
- Live progress widgets need a real terminal (`stdout.isatty()`); running
  MARROW with output redirected to a file or through another program's
  pipe gets the old plain-text behavior, not the new bars — this is
  intentional (an animated bar redrawing into a log file is worse than no
  bar), not a bug.
