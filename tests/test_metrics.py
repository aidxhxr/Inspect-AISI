import math

import numpy as np
import pytest
from inspect_ai.scorer import CORRECT, INCORRECT, NOANSWER, SampleScore, Score

from aisi_evals.metrics import (
    abstention_rate,
    aurc,
    brier,
    correct_given_attempted,
    ece,
    permutation_consistency,
    position_bias,
    selective_accuracy,
    simpleqa_f1,
)
from aisi_evals.metrics.calibration import (
    accuracy_at_coverage,
    area_under_risk_coverage,
    brier_score,
    expected_calibration_error,
)

GRADES = {CORRECT: "correct", INCORRECT: "incorrect", NOANSWER: "not_attempted"}


def ss(value, confidence=None, sample_metadata=None, chosen=None, sample_id=None):
    meta = {"grade": GRADES[value]}
    if confidence is not None:
        meta["confidence"] = confidence
    if chosen is not None:
        meta["chosen"] = chosen
    return SampleScore(
        score=Score(value=value, metadata=meta),
        sample_id=sample_id,
        sample_metadata=sample_metadata,
    )


def test_brier_known_values():
    conf = np.array([1.0, 0.0, 0.5, 0.5])
    corr = np.array([1.0, 0.0, 1.0, 0.0])
    assert brier_score(conf, corr) == pytest.approx(0.125)


def test_brier_metric_excludes_abstentions():
    scores = [ss(CORRECT, 0.9), ss(INCORRECT, 0.9), ss(NOANSWER, 0.1)]
    assert brier()(scores) == pytest.approx(((0.1) ** 2 + (0.9) ** 2) / 2)


def test_ece_perfectly_calibrated_bins_is_zero():
    conf = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.2, 0.2, 0.2, 0.2, 0.2])
    corr = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 1], dtype=float)
    assert expected_calibration_error(conf, corr, n_bins=10) == pytest.approx(0.0)


def test_ece_overconfident():
    conf = np.full(4, 0.95)
    corr = np.array([1.0, 0.0, 0.0, 0.0])
    assert expected_calibration_error(conf, corr, n_bins=10) == pytest.approx(0.7)


def test_risk_coverage_ranking():
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    good = np.array([1.0, 1.0, 0.0, 0.0])
    bad = np.array([0.0, 0.0, 1.0, 1.0])
    assert area_under_risk_coverage(conf, good) < area_under_risk_coverage(conf, bad)
    assert accuracy_at_coverage(conf, good, 0.5) == 1.0
    assert accuracy_at_coverage(conf, bad, 0.5) == 0.0


def test_selective_accuracy_metric_uses_top_half():
    scores = [ss(CORRECT, 0.9), ss(CORRECT, 0.8), ss(INCORRECT, 0.3), ss(INCORRECT, 0.2)]
    assert selective_accuracy(0.5)(scores) == 1.0
    assert aurc()(scores) < 0.3


def test_empty_calibration_is_nan():
    assert math.isnan(brier()([]))
    assert math.isnan(ece()([ss(NOANSWER)]))


def test_simpleqa_counts():
    scores = [ss(CORRECT)] * 6 + [ss(INCORRECT)] * 2 + [ss(NOANSWER)] * 2
    assert correct_given_attempted()(scores) == pytest.approx(0.75)
    assert abstention_rate()(scores) == pytest.approx(0.2)
    overall, precision = 0.6, 0.75
    assert simpleqa_f1()(scores) == pytest.approx(2 * overall * precision / (overall + precision))


def test_simpleqa_f1_no_attempts_is_nan():
    assert math.isnan(simpleqa_f1()([ss(NOANSWER), ss(NOANSWER)]))


def test_permutation_consistency():
    scores = [
        ss(CORRECT, sample_metadata={"original_id": 1}, chosen="x"),
        ss(CORRECT, sample_metadata={"original_id": 1}, chosen="x"),
        ss(INCORRECT, sample_metadata={"original_id": 2}, chosen="y"),
        ss(CORRECT, sample_metadata={"original_id": 2}, chosen="z"),
        ss(CORRECT, sample_metadata={"original_id": 3}, chosen="w"),
    ]
    assert permutation_consistency()(scores) == pytest.approx(0.5)


def test_position_bias_spread():
    scores = [
        ss(CORRECT, sample_metadata={"target_position": 0}),
        ss(CORRECT, sample_metadata={"target_position": 0}),
        ss(INCORRECT, sample_metadata={"target_position": 3}),
        ss(CORRECT, sample_metadata={"target_position": 3}),
    ]
    assert position_bias()(scores) == pytest.approx(0.5)
    assert math.isnan(position_bias()(scores[:2]))
