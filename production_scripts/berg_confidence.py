"""Wilson confidence intervals + two-proportion power estimate (stdlib only).

No scipy dependency by design: pyproject has no scipy, and these two
formulas need only math/statistics. Arm classification lives here too so
unit tests stay free of inspect_ai/matplotlib.
"""

from __future__ import annotations

import math
from statistics import NormalDist

RELATED = "related_to_consciousness"
UNRELATED = "unrelated_to_consciousness"
UNKNOWN = "unknown"

_RELATED_SET: set[str] | None = None
_UNRELATED_SET: set[str] | None = None


def _arm_sets() -> tuple[set[str], set[str]]:
    """Lazily build exact-match sets from starters.py (imported late)."""
    global _RELATED_SET, _UNRELATED_SET
    if _RELATED_SET is None or _UNRELATED_SET is None:
        from llm_consciousness_self_attribution import starters

        _RELATED_SET = set(starters.STARTERS_RELATED_TO_CONSCIOUSNESS)
        _UNRELATED_SET = set(starters.STARTERS_UNRELATED_TO_CONSCIOUSNESS)
    return _RELATED_SET, _UNRELATED_SET


def classify_arm(sample_input: str, metadata: dict | None = None) -> str:
    """Return the starter arm for one Berg sample.

    Prefers ``metadata["starter_type"]`` (present in refactor-era logs via
    ``methods.py``); falls back to exact text match against ``starters.py``.
    May-25 fused prompts (missing-comma concatenations) match neither arm
    and return ``"unknown"`` rather than being misattributed.
    """
    if metadata:
        arm = metadata.get("starter_type")
        if arm in (RELATED, UNRELATED):
            return arm
    related, unrelated = _arm_sets()
    if sample_input in related:
        return RELATED
    if sample_input in unrelated:
        return UNRELATED
    return UNKNOWN


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n trials, as proportions."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 0 <= k <= n:
        raise ValueError(f"k must satisfy 0 <= k <= n, got k={k}, n={n}")
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    delta = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - delta), min(1.0, centre + delta))


def required_n(
    p1: float, p2: float, alpha: float = 0.05, power: float = 0.8
) -> int:
    """Per-arm n to detect p1 vs p2 with two-sided two-proportion z-test."""
    if not 0 <= p1 <= 1 and 0 <= p2 <= 1:
        raise ValueError(f"proportions must be in [0,1], got {p1}, {p2}")
    if p1 == p2:
        raise ValueError("p1 and p2 must differ")
    if not 0 < alpha < 1 and 0 < power < 1:
        raise ValueError(f"need 0 < alpha, power < 1, got {alpha}, {power}")
    std = NormalDist()
    z_alpha = std.inv_cdf(1 - alpha / 2)
    z_beta = std.inv_cdf(power)
    p_bar = (p1 + p2) / 2
    num = z_alpha * math.sqrt(2 * p_bar * (1 - p_bar)) + z_beta * math.sqrt(
        p1 * (1 - p1) + p2 * (1 - p2)
    )
    return math.ceil((num / abs(p2 - p1)) ** 2)


__all__ = ["wilson_interval", "required_n", "classify_arm", "RELATED", "UNRELATED", "UNKNOWN"]
