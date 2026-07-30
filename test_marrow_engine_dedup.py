"""
Regression test for the "asked for 26 screenshots, only got a handful
back" bug: marrow_engine.deduplicate_frames() used to compare every frame
to whichever one it most recently decided to keep, with no limit on how
far apart in time the two could be. On any video where slides/frames
share a visual template (extremely common — same PowerPoint theme, same
code editor, same whiteboard framing), this silently deleted screenshots
from completely different parts of the video before Vision analysis ever
saw them.

Fixed by (a) only ever comparing frames within a short time window of
each other, and (b) using windowed SSIM instead of a whole-frame mean-
pixel-difference, which gives much better separation between "genuinely
the same frame" and "different content, similar background".

These tests use synthetic in-memory images (no real video/network
needed) that specifically reproduce the reported failure shape.
"""
import sys
from pathlib import Path

import numpy as np
import cv2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import marrow_engine as engine


def _make_slide(path, text, bg=(30, 30, 30), pos=(40, 240)):
    img = np.full((480, 640, 3), bg, dtype=np.uint8)
    cv2.putText(img, str(text), pos, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
    cv2.imwrite(str(path), img)


def test_far_apart_screenshots_sharing_a_template_all_survive(tmp_path):
    """The core reported bug: 26 distinct screenshots, same visual
    template throughout (as most real videos are), spread across a
    ~40-minute video. All 26 must survive dedup."""
    screenshot_map = {}
    for i in range(26):
        ts = i * 90 + 10  # every ~90s
        p = tmp_path / f"frame_{ts}.png"
        _make_slide(p, f"Concept #{i + 1} - unique content")
        screenshot_map[ts] = str(p)

    result = engine.deduplicate_frames(dict(screenshot_map))

    assert len(result) == 26, (
        f"expected all 26 distinct far-apart screenshots to survive, "
        f"got {len(result)} — same-template frames are being wrongly "
        f"treated as duplicates again"
    )
    assert set(result.keys()) == set(screenshot_map.keys())


def test_genuine_near_duplicate_still_collapses(tmp_path):
    """The case this function is actually meant to catch: the same slide
    held on screen, producing several extracted frames a couple of
    seconds apart with identical content — these should still collapse
    down to one."""
    screenshot_map = {}
    for ts in (500, 503, 506):
        p = tmp_path / f"frame_{ts}.png"
        _make_slide(p, "Held Slide", bg=(30, 30, 30))
        screenshot_map[ts] = str(p)

    result = engine.deduplicate_frames(dict(screenshot_map))
    assert len(result) == 1, (
        f"expected 3 near-identical, closely-timed frames to collapse "
        f"to 1, got {len(result)}"
    )


def test_different_content_close_in_time_both_survive(tmp_path):
    """A fast slide transition: two DIFFERENT slides just a couple of
    seconds apart (within the time-proximity window) should still both
    survive — being close in time isn't enough on its own, the content
    has to actually be near-identical too."""
    screenshot_map = {}
    _make_slide(tmp_path / "a.png", "First distinct slide content")
    _make_slide(tmp_path / "b.png", "Totally different second slide")
    screenshot_map[100] = str(tmp_path / "a.png")
    screenshot_map[103] = str(tmp_path / "b.png")

    result = engine.deduplicate_frames(dict(screenshot_map))
    assert len(result) == 2, (
        "two genuinely different slides, even close in time, should "
        "both survive — only true near-duplicates should be removed"
    )


def test_single_or_empty_map_is_a_noop():
    assert engine.deduplicate_frames({}) == {}
    m = {10: "/nonexistent/path.png"}
    # a single entry should short-circuit before ever touching disk
    assert engine.deduplicate_frames(m) == m


def test_local_ssim_scores_identical_images_as_1():
    img = np.random.randint(0, 255, (120, 160), dtype=np.uint8)
    assert engine._local_ssim(img, img) == pytest.approx(1.0, abs=1e-6)


def test_local_ssim_separates_similar_from_identical(tmp_path):
    """The measured justification for switching metrics: SSIM should
    give a same-template-different-content pair a noticeably lower score
    than a truly identical pair, with real margin for a threshold."""
    _make_slide(tmp_path / "a.png", "Slide A")
    _make_slide(tmp_path / "b.png", "Slide B - different text entirely")
    _make_slide(tmp_path / "c.png", "Slide A")  # identical to a.png

    def gray(p):
        return cv2.resize(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2GRAY), (160, 120))

    ga, gb, gc = gray(tmp_path / "a.png"), gray(tmp_path / "b.png"), gray(tmp_path / "c.png")
    identical_score = engine._local_ssim(ga, gc)
    different_score = engine._local_ssim(ga, gb)

    assert identical_score > 0.99
    assert different_score < identical_score - 0.05, (
        "expected meaningful separation between identical and "
        "different-content-same-template pairs"
    )
