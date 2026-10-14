"""Metrics from the SimpleQA paper (Wei et al., 2024) for three-way graded answers."""

from __future__ import annotations

from inspect_ai.scorer import Metric, SampleScore, metric

from aisi_evals.metrics.grades import is_abstention, is_correct


def _counts(scores: list[SampleScore]) -> tuple[int, int, int]:
    not_attempted = sum(1 for s in scores if is_abstention(s))
    correct = sum(1 for s in scores if not is_abstention(s) and is_correct(s))
    incorrect = len(scores) - correct - not_attempted
    return correct, incorrect, not_attempted


@metric
def correct_given_attempted() -> Metric:
    """Precision of attempted answers: correct / (correct + incorrect)."""

    def compute(scores: list[SampleScore]) -> float:
        c, i, _ = _counts(scores)
        return c / (c + i) if (c + i) else float("nan")

    return compute


@metric
def attempted_rate() -> Metric:
    def compute(scores: list[SampleScore]) -> float:
        c, i, n = _counts(scores)
        total = c + i + n
        return (c + i) / total if total else float("nan")

    return compute


@metric
def simpleqa_f1() -> Metric:
    """Harmonic mean of overall accuracy and correct-given-attempted."""

    def compute(scores: list[SampleScore]) -> float:
        c, i, n = _counts(scores)
        total = c + i + n
        if not total or not (c + i):
            return float("nan")
        overall = c / total
        precision = c / (c + i)
        if overall + precision == 0:
            return 0.0
        return 2 * overall * precision / (overall + precision)

    return compute
