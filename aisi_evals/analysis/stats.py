"""Statistics for comparing eval results.

Implements the recommendations of Miller, "Adding Error Bars to Evals"
(arXiv 2411.00640): CLT standard errors, clustering by question when a
question is answered more than once, paired differences when two models
answered the same questions, and multiple-comparison correction when many
models are compared. Pure numpy; no dependency on Inspect.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import numpy as np

_BOOT_CHUNK = 2_000


def _z(alpha: float) -> float:
    return NormalDist().inv_cdf(1.0 - alpha / 2.0)


def _as_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("values must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("values must be finite")
    return arr


def is_binary(values: Sequence[float] | np.ndarray) -> bool:
    """True when every value is exactly 0 or 1."""
    arr = np.asarray(values, dtype=float).ravel()
    return arr.size > 0 and bool(np.all((arr == 0.0) | (arr == 1.0)))


def clt_se(values: Sequence[float] | np.ndarray) -> float:
    """Central limit theorem standard error of the mean (sample std, ddof=1)."""
    arr = _as_array(values)
    if arr.size < 2:
        return 0.0
    return float(np.std(arr, ddof=1) / math.sqrt(arr.size))


def wilson_interval(p: float, n: float, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a proportion `p` observed over `n` trials."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must lie in [0, 1]")
    z = _z(alpha)
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    lo = 0.0 if p == 0.0 else max(center - half, 0.0)
    hi = 1.0 if p == 1.0 else min(center + half, 1.0)
    return lo, hi


def mean_ci(
    values: Sequence[float] | np.ndarray, alpha: float = 0.05
) -> tuple[float, float, float, float]:
    """Mean with a two-sided (1 - alpha) confidence interval.

    Returns (mean, lo, hi, se). The interval is Wilson when every value is
    0 or 1, otherwise mean +/- z * se. `se` is always the CLT standard error.
    """
    arr = _as_array(values)
    mean = float(arr.mean())
    se = clt_se(arr)
    if is_binary(arr):
        lo, hi = wilson_interval(mean, arr.size, alpha)
    else:
        z = _z(alpha)
        lo, hi = mean - z * se, mean + z * se
    return mean, lo, hi, se


def _partition(
    values: np.ndarray, clusters: Sequence[Any] | np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Per-cluster sums of `values` and per-cluster sizes, in first-seen order."""
    if clusters is None:
        return values.copy(), np.ones(values.size)
    keys = np.asarray(list(clusters), dtype=object).ravel()
    if keys.size != values.size:
        raise ValueError("clusters must have the same length as values")
    index: dict[Any, int] = {}
    codes = np.empty(values.size, dtype=int)
    for i, key in enumerate(keys):
        if key is None or (isinstance(key, float) and math.isnan(key)):
            raise ValueError(f"cluster id at position {i} is missing")
        codes[i] = index.setdefault(key, len(index))
    sums = np.bincount(codes, weights=values, minlength=len(index))
    sizes = np.bincount(codes, minlength=len(index)).astype(float)
    return sums, sizes


def clustered_se(
    values: Sequence[float] | np.ndarray, clusters: Sequence[Any] | np.ndarray
) -> float:
    """Clustered standard error of the mean (Appendix A of arXiv 2411.00640).

    Uses the finite-cluster correction C / (C - 1), matching
    `inspect_ai.scorer.stderr(cluster=...)`. With one observation per
    cluster this equals the CLT standard error. Returns 0 for fewer than
    two clusters.
    """
    arr = _as_array(values)
    mean = arr.mean()
    sums, sizes = _partition(arr - mean, clusters)
    count = sums.size
    if count < 2:
        return 0.0
    variance = float(np.sum(sums**2)) * count / (count - 1)
    return math.sqrt(variance) / arr.size


@dataclass(frozen=True)
class PairedResult:
    """Paired comparison of two score vectors over the same questions.

    `mean_diff` is mean(a - b). `ci_lo`/`ci_hi` bound a percentile bootstrap
    over the paired differences; `p_value` is a two-sided sign-flip
    permutation test. When clusters are given, both resample or flip whole
    clusters, and `se_paired` is the clustered standard error.
    """

    mean_diff: float
    se_paired: float
    ci_lo: float
    ci_hi: float
    p_value: float
    n: int
    wins: int
    losses: int
    ties: int
    n_clusters: int = 0
    name_a: str = "a"
    name_b: str = "b"

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05


def paired_difference(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
    clusters: Sequence[Any] | np.ndarray | None = None,
) -> PairedResult:
    """Paired difference a - b with bootstrap interval and sign-flip p-value."""
    arr_a = _as_array(a)
    arr_b = _as_array(b)
    if arr_a.size != arr_b.size:
        raise ValueError(f"a and b must have the same length, got {arr_a.size} and {arr_b.size}")
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    diff = arr_a - arr_b
    n = diff.size
    mean_diff = float(diff.mean())
    sums, sizes = _partition(diff, clusters)
    count = sums.size
    se = clustered_se(diff, clusters) if clusters is not None else clt_se(diff)

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    exceed = 0
    observed = abs(mean_diff)
    for start in range(0, n_boot, _BOOT_CHUNK):
        stop = min(start + _BOOT_CHUNK, n_boot)
        rows = stop - start
        picks = rng.integers(0, count, size=(rows, count))
        boot_means[start:stop] = sums[picks].sum(axis=1) / sizes[picks].sum(axis=1)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(rows, count))
        flipped = np.abs((signs * sums).sum(axis=1)) / n
        exceed += int(np.count_nonzero(flipped >= observed - 1e-12))
    lo = float(np.quantile(boot_means, alpha / 2.0))
    hi = float(np.quantile(boot_means, 1.0 - alpha / 2.0))
    p_value = (exceed + 1) / (n_boot + 1)

    return PairedResult(
        mean_diff=mean_diff,
        se_paired=se,
        ci_lo=lo,
        ci_hi=hi,
        p_value=float(p_value),
        n=int(n),
        wins=int(np.count_nonzero(diff > 0)),
        losses=int(np.count_nonzero(diff < 0)),
        ties=int(np.count_nonzero(diff == 0)),
        n_clusters=int(count),
    )


def holm_correction(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, returned in the input order."""
    p = np.asarray(list(p_values), dtype=float)
    if p.size == 0:
        return []
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p-values must lie in [0, 1]")
    m = p.size
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return [float(x) for x in adjusted]


def _check_design(p: float, alpha: float, power: float) -> None:
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie strictly between 0 and 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    if not 0.0 < power < 1.0:
        raise ValueError("power must lie strictly between 0 and 1")


def min_detectable_difference(
    n: int, p: float = 0.5, alpha: float = 0.05, power: float = 0.8
) -> float:
    """Smallest accuracy gap detectable between two models with `n` questions each.

    Two-sided test at level `alpha` with the given power, treating the two
    models as independent binomials near accuracy `p` (the paper's unpaired
    bound; a paired test on shared questions needs fewer samples).
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    _check_design(p, alpha, power)
    z = _z(alpha) + NormalDist().inv_cdf(power)
    return float(z * math.sqrt(2.0 * p * (1.0 - p) / n))


def required_n(diff: float, p: float = 0.5, alpha: float = 0.05, power: float = 0.8) -> int:
    """Questions per model needed to detect an accuracy gap of `diff`.

    Inverse of `min_detectable_difference` under the same unpaired model.
    """
    if diff <= 0.0:
        raise ValueError("diff must be positive")
    _check_design(p, alpha, power)
    z = _z(alpha) + NormalDist().inv_cdf(power)
    return int(math.ceil(2.0 * p * (1.0 - p) * (z / diff) ** 2))
