"""Consistency of answers across permuted presentations of the same question."""

from __future__ import annotations

from collections import defaultdict

from inspect_ai.scorer import Metric, SampleScore, metric

from aisi_evals.metrics.grades import is_correct


@metric
def permutation_consistency(group_key: str = "original_id", answer_key: str = "chosen") -> Metric:
    """Fraction of questions whose chosen option is identical under every permutation.

    Groups scores by `metadata[group_key]` and reads the chosen option text
    from `Score.metadata[answer_key]`. Questions seen only once are skipped.
    """

    def compute(scores: list[SampleScore]) -> float:
        groups: dict[object, list[str | None]] = defaultdict(list)
        for s in scores:
            meta = s.sample_metadata or {}
            if group_key not in meta:
                continue
            groups[meta[group_key]].append((s.score.metadata or {}).get(answer_key))
        multi = [v for v in groups.values() if len(v) > 1]
        if not multi:
            return float("nan")
        return sum(1 for v in multi if len(set(v)) == 1) / len(multi)

    return compute


@metric
def position_bias(position_key: str = "target_position") -> Metric:
    """Spread of accuracy across the position of the correct option (max minus min)."""

    def compute(scores: list[SampleScore]) -> float:
        by_pos: dict[object, list[float]] = defaultdict(list)
        for s in scores:
            meta = s.sample_metadata or {}
            if position_key not in meta:
                continue
            by_pos[meta[position_key]].append(1.0 if is_correct(s) else 0.0)
        if len(by_pos) < 2:
            return float("nan")
        means = [sum(v) / len(v) for v in by_pos.values()]
        return max(means) - min(means)

    return compute
