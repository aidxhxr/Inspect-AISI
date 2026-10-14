"""Read correctness and abstention from a score after epoch reduction.

Inspect converts score values to floats before metrics run, so CORRECT,
INCORRECT and NOANSWER are no longer distinguishable by value alone. Scorers
in this package record the grade in `Score.metadata["grade"]`, which survives
reduction, and these helpers read it when present.
"""

from __future__ import annotations

from inspect_ai.scorer import NOANSWER, SampleScore, value_to_float

_to_float = value_to_float()


def grade(score: SampleScore) -> str | None:
    meta = score.score.metadata or {}
    value = meta.get("grade")
    return str(value) if value is not None else None


def is_abstention(score: SampleScore) -> bool:
    return grade(score) == "not_attempted" or score.score.value == NOANSWER


def is_correct(score: SampleScore) -> bool:
    g = grade(score)
    if g is not None:
        return g == "correct"
    return _to_float(score.score.value) == 1.0
