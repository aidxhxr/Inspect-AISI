"""Calibration metrics for scorers that attach a confidence to each score.

Each `Score` is expected to carry `metadata["confidence"]` in [0, 1] and a
value of CORRECT, INCORRECT or NOANSWER. NOANSWER marks an abstention: the
model declined to answer, so it is excluded from calibration statistics and
counted separately by `abstention_rate`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from inspect_ai.scorer import Metric, SampleScore, metric

from aisi_evals.metrics.grades import is_abstention, is_correct


@dataclass(frozen=True)
class Attempted:
    confidence: np.ndarray
    correct: np.ndarray


def attempted(scores: list[SampleScore]) -> Attempted:
    """Confidence and correctness arrays over scores where the model answered."""
    conf: list[float] = []
    corr: list[float] = []
    for s in scores:
        if is_abstention(s):
            continue
        meta = s.score.metadata or {}
        if meta.get("confidence") is None:
            continue
        conf.append(float(meta["confidence"]))
        corr.append(1.0 if is_correct(s) else 0.0)
    return Attempted(np.asarray(conf, dtype=float), np.asarray(corr, dtype=float))


def brier_score(confidence: np.ndarray, correct: np.ndarray) -> float:
    if confidence.size == 0:
        return float("nan")
    return float(np.mean((confidence - correct) ** 2))


def expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10
) -> float:
    """Equal-width-bin ECE: sum over bins of |accuracy - mean confidence| weighted by bin mass."""
    if confidence.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, n_bins - 1)
    total = confidence.size
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        ece += mask.sum() / total * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def risk_coverage(confidence: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Risk (error rate) at each coverage level when answers are kept in confidence order."""
    order = np.argsort(-confidence, kind="stable")
    errors = 1.0 - correct[order]
    n = errors.size
    coverage = np.arange(1, n + 1) / n
    risk = np.cumsum(errors) / np.arange(1, n + 1)
    return coverage, risk


def area_under_risk_coverage(confidence: np.ndarray, correct: np.ndarray) -> float:
    if confidence.size == 0:
        return float("nan")
    coverage, risk = risk_coverage(confidence, correct)
    return float(np.trapezoid(risk, coverage)) if confidence.size > 1 else float(risk[0])


def accuracy_at_coverage(confidence: np.ndarray, correct: np.ndarray, coverage: float) -> float:
    if confidence.size == 0:
        return float("nan")
    k = max(1, int(np.ceil(coverage * confidence.size)))
    order = np.argsort(-confidence, kind="stable")
    return float(correct[order][:k].mean())


@metric
def brier() -> Metric:
    """Mean squared error between stated confidence and correctness over attempted answers."""

    def compute(scores: list[SampleScore]) -> float:
        a = attempted(scores)
        return brier_score(a.confidence, a.correct)

    return compute


@metric
def ece(n_bins: int = 10) -> Metric:
    """Expected calibration error with equal-width confidence bins."""

    def compute(scores: list[SampleScore]) -> float:
        a = attempted(scores)
        return expected_calibration_error(a.confidence, a.correct, n_bins)

    return compute


@metric
def aurc() -> Metric:
    """Area under the risk-coverage curve; lower means confidence ranks errors well."""

    def compute(scores: list[SampleScore]) -> float:
        a = attempted(scores)
        return area_under_risk_coverage(a.confidence, a.correct)

    return compute


@metric
def selective_accuracy(coverage: float = 0.5) -> Metric:
    """Accuracy on the most confident `coverage` fraction of attempted answers."""

    def compute(scores: list[SampleScore]) -> float:
        a = attempted(scores)
        return accuracy_at_coverage(a.confidence, a.correct, coverage)

    return compute


@metric
def mean_confidence() -> Metric:
    def compute(scores: list[SampleScore]) -> float:
        a = attempted(scores)
        return float(a.confidence.mean()) if a.confidence.size else float("nan")

    return compute


@metric
def abstention_rate() -> Metric:
    """Fraction of samples where the model declined to answer."""

    def compute(scores: list[SampleScore]) -> float:
        if not scores:
            return float("nan")
        return sum(1 for s in scores if is_abstention(s)) / len(scores)

    return compute
