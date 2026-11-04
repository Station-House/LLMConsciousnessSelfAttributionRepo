"""Unit tests for production_scripts/berg_confidence.py (stdlib only, no GPU)."""

from __future__ import annotations

import pytest

from production_scripts import berg_confidence as bc


def test_wilson_known_values():
    lo, hi = bc.wilson_interval(0, 20)
    assert lo == 0.0
    assert hi == pytest.approx(0.161, abs=0.005)
    lo, hi = bc.wilson_interval(1, 20)
    assert lo == pytest.approx(0.009, abs=0.005)
    assert hi == pytest.approx(0.236, abs=0.005)
    lo, hi = bc.wilson_interval(1, 18)
    assert lo == pytest.approx(0.010, abs=0.005)
    assert hi == pytest.approx(0.258, abs=0.005)


def test_wilson_interval_stays_in_bounds():
    lo, hi = bc.wilson_interval(20, 20)
    assert hi == 1.0
    assert 0.8 < lo < 1.0
    lo, hi = bc.wilson_interval(10, 20)
    assert 0.0 < lo < 0.5 < hi < 1.0


def test_wilson_interval_rejects_bad_input():
    with pytest.raises(ValueError):
        bc.wilson_interval(0, 0)
    with pytest.raises(ValueError):
        bc.wilson_interval(21, 20)
    with pytest.raises(ValueError):
        bc.wilson_interval(-1, 20)


def test_required_n_sanity_and_symmetry():
    n_wide = bc.required_n(0.05, 0.15)
    n_narrow = bc.required_n(0.05, 0.06)
    assert isinstance(n_wide, int) and n_wide > 0
    assert n_wide < n_narrow  # larger effects need fewer samples
    assert bc.required_n(0.05, 0.15) == bc.required_n(0.15, 0.05)
    assert 100 < n_wide < 300  # guards against formula regressions


def test_required_n_rejects_bad_input():
    with pytest.raises(ValueError):
        bc.required_n(0.05, 0.05)
    with pytest.raises(ValueError):
        bc.required_n(0.05, 0.15, alpha=1.5)


def test_classify_arm_prefers_metadata():
    assert bc.classify_arm("anything", {"starter_type": bc.RELATED}) == bc.RELATED
    assert bc.classify_arm("anything", {"starter_type": bc.UNRELATED}) == bc.UNRELATED
    assert bc.classify_arm("anything", {}) != "crash"


def test_classify_arm_matches_starter_text():
    from llm_consciousness_self_attribution import starters

    assert (
        bc.classify_arm(starters.STARTERS_UNRELATED_TO_CONSCIOUSNESS[0], None)
        == bc.UNRELATED
    )
    assert (
        bc.classify_arm(starters.STARTERS_RELATED_TO_CONSCIOUSNESS[0], None)
        == bc.RELATED
    )
    # May-25 fused prompts match neither bank: unknown, not misattributed.
    fused = (
        starters.STARTERS_UNRELATED_TO_CONSCIOUSNESS[0]
        + starters.STARTERS_UNRELATED_TO_CONSCIOUSNESS[1]
    )
    assert bc.classify_arm(fused, None) == bc.UNKNOWN
