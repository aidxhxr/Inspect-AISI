import math

import numpy as np
import pandas as pd
import pytest
from inspect_ai import Task, eval, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import SampleScore, Score, match, stderr
from inspect_ai.solver import generate

from aisi_evals.analysis import (
    PairedResult,
    clt_se,
    clustered_se,
    compare_models,
    compare_to_best,
    holm_correction,
    is_binary,
    leaderboard,
    load_scores,
    mean_ci,
    min_detectable_difference,
    paired_difference,
    render_markdown,
    required_n,
    wilson_interval,
)


def test_is_binary() -> None:
    assert is_binary([0, 1, 1, 0])
    assert not is_binary([0, 0.5, 1])
    assert not is_binary([])


def test_wilson_interval_seven_of_ten() -> None:
    lo, hi = wilson_interval(0.7, 10)
    assert lo == pytest.approx(0.3968, abs=1e-3)
    assert hi == pytest.approx(0.8922, abs=1e-3)


def test_wilson_interval_extremes_stay_in_unit_range() -> None:
    lo, hi = wilson_interval(1.0, 5)
    assert lo > 0.5
    assert hi == 1.0
    lo, hi = wilson_interval(0.0, 5)
    assert lo == 0.0
    assert hi < 0.5


def test_mean_ci_binary_uses_wilson() -> None:
    values = [1] * 7 + [0] * 3
    mean, lo, hi, se = mean_ci(values)
    assert mean == pytest.approx(0.7)
    assert (lo, hi) == pytest.approx(wilson_interval(0.7, 10))
    assert se == pytest.approx(clt_se(values))
    assert se == pytest.approx(np.std(values, ddof=1) / math.sqrt(10))


def test_mean_ci_continuous_uses_normal() -> None:
    values = [0.1, 0.4, 0.35, 0.8, 0.6]
    mean, lo, hi, se = mean_ci(values)
    assert mean == pytest.approx(np.mean(values))
    assert hi - mean == pytest.approx(1.959964 * se, rel=1e-5)
    assert mean - lo == pytest.approx(1.959964 * se, rel=1e-5)


def test_mean_ci_rejects_empty_and_nonfinite() -> None:
    with pytest.raises(ValueError):
        mean_ci([])
    with pytest.raises(ValueError):
        mean_ci([1.0, float("nan")])


def test_clustered_se_matches_inspect_implementation() -> None:
    rng = np.random.default_rng(123)
    for trial in range(5):
        n_clusters = 12 + trial
        values = rng.random(n_clusters * 4)
        clusters = rng.integers(0, n_clusters, size=values.size).tolist()
        scores = [
            SampleScore(score=Score(value=float(v)), sample_id=i, sample_metadata={"q": c})
            for i, (v, c) in enumerate(zip(values, clusters, strict=True))
        ]
        expected = stderr(cluster="q")(scores)
        assert clustered_se(values, clusters) == pytest.approx(expected, rel=1e-12)


def test_clustered_se_singleton_clusters_equals_clt_se() -> None:
    values = [0.2, 0.9, 0.4, 0.7, 0.1]
    assert clustered_se(values, list(range(5))) == pytest.approx(clt_se(values))
    assert clustered_se(values, [0] * 5) == 0.0


def test_clustered_se_rejects_missing_cluster_ids() -> None:
    with pytest.raises(ValueError):
        clustered_se([1.0, 0.0], ["a", None])
    with pytest.raises(ValueError):
        clustered_se([1.0, 0.0], ["a"])


def test_paired_difference_identical_inputs() -> None:
    a = [1, 0, 1, 1, 0, 1, 0, 0]
    result = paired_difference(a, a, n_boot=2000)
    assert isinstance(result, PairedResult)
    assert result.mean_diff == 0.0
    assert result.se_paired == 0.0
    assert result.ci_lo == 0.0 and result.ci_hi == 0.0
    assert result.p_value == pytest.approx(1.0)
    assert (result.wins, result.losses, result.ties) == (0, 0, 8)
    assert result.n == 8


def test_paired_difference_clearly_separated() -> None:
    a = [1] * 30
    b = [0] * 30
    result = paired_difference(a, b, n_boot=5000, seed=1)
    assert result.mean_diff == 1.0
    assert result.p_value < 0.001
    assert result.wins == 30 and result.losses == 0
    assert result.ci_lo == 1.0 and result.ci_hi == 1.0


def test_paired_difference_known_values() -> None:
    a = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    b = np.array([0.5, 0.7, 0.6, 0.6, 0.1, 0.5])
    result = paired_difference(a, b, n_boot=4000, seed=3)
    diff = a - b
    assert result.mean_diff == pytest.approx(diff.mean())
    assert result.se_paired == pytest.approx(np.std(diff, ddof=1) / math.sqrt(6))
    assert result.ci_lo <= result.mean_diff <= result.ci_hi
    assert (result.wins, result.losses, result.ties) == (4, 1, 1)
    assert 0.0 < result.p_value <= 1.0


def test_paired_difference_is_deterministic_for_seed() -> None:
    rng = np.random.default_rng(7)
    a = rng.random(40)
    b = rng.random(40)
    first = paired_difference(a, b, n_boot=500, seed=11)
    second = paired_difference(a, b, n_boot=500, seed=11)
    assert first == second


def test_paired_difference_with_clusters_uses_clustered_se() -> None:
    a = [1, 1, 0, 0, 1, 0]
    b = [0, 0, 0, 0, 0, 0]
    clusters = ["q1", "q1", "q2", "q2", "q3", "q3"]
    result = paired_difference(a, b, n_boot=1000, clusters=clusters)
    diff = np.array(a, dtype=float) - np.array(b, dtype=float)
    assert result.se_paired == pytest.approx(clustered_se(diff, clusters))
    assert result.n_clusters == 3
    assert result.n == 6


def test_paired_difference_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        paired_difference([1, 0], [1])


def test_holm_correction_known_vector() -> None:
    assert holm_correction([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_correction([0.5]) == [0.5]
    assert holm_correction([]) == []
    adjusted = holm_correction([0.6, 0.7, 0.9])
    assert all(p <= 1.0 for p in adjusted)
    assert holm_correction([0.04, 0.04]) == pytest.approx([0.08, 0.08])


def test_required_n_and_mdd_are_monotone_and_consistent() -> None:
    diffs = [0.2, 0.1, 0.05, 0.02]
    ns = [required_n(d) for d in diffs]
    assert ns == sorted(ns)
    assert required_n(0.05, power=0.9) > required_n(0.05, power=0.8)
    assert required_n(0.05, alpha=0.01) > required_n(0.05, alpha=0.05)
    assert required_n(0.05, p=0.5) > required_n(0.05, p=0.9)
    for d in diffs:
        n = required_n(d)
        assert min_detectable_difference(n) <= d
        assert min_detectable_difference(n - 1) > d
    mdds = [min_detectable_difference(n) for n in (50, 100, 1000)]
    assert mdds == sorted(mdds, reverse=True)
    assert required_n(0.1) == 393


def test_design_helpers_reject_bad_inputs() -> None:
    with pytest.raises(ValueError):
        required_n(0.0)
    with pytest.raises(ValueError):
        min_detectable_difference(0)
    with pytest.raises(ValueError):
        required_n(0.1, p=1.0)


def _scores_frame() -> pd.DataFrame:
    rows = []
    for model, values in {"m1": [1, 1, 1, 0], "m2": [1, 0, 0, 0], "m3": [1, 1, 0, 0]}.items():
        for i, v in enumerate(values):
            rows.append(
                {
                    "task": "t",
                    "model": model,
                    "eval_id": model,
                    "sample_id": f"s{i}",
                    "epoch": 1,
                    "scorer": "match",
                    "value": float(v),
                    "cluster": f"s{i}",
                }
            )
    return pd.DataFrame(rows)


def test_compare_models_on_frame() -> None:
    df = _scores_frame()
    result = compare_models(df, "t", "match", baseline="m2", candidate="m1", n_boot=500)
    assert result.mean_diff == pytest.approx(0.5)
    assert result.name_a == "m1" and result.name_b == "m2"
    assert (result.wins, result.losses, result.ties) == (2, 0, 2)


def test_compare_models_errors() -> None:
    df = _scores_frame()
    with pytest.raises(ValueError, match="no scores"):
        compare_models(df, "other", "match", "m1", "m2")
    with pytest.raises(ValueError, match="has no scores"):
        compare_models(df, "t", "match", "m1", "missing")
    partial = df[~((df["model"] == "m2") & (df["sample_id"] == "s3"))]
    with pytest.raises(ValueError, match="did not run the same samples"):
        compare_models(partial, "t", "match", "m1", "m2")
    doubled = pd.concat([df, df[df["model"] == "m1"]])
    with pytest.raises(ValueError, match="more than one score"):
        compare_models(doubled, "t", "match", "m1", "m2")


def test_leaderboard_and_markdown_on_frame() -> None:
    df = _scores_frame()
    table = leaderboard(df, "t", "match", n_boot=500)
    assert table["model"].tolist() == ["m1", "m3", "m2"]
    assert table["mean"].tolist() == pytest.approx([0.75, 0.5, 0.25])
    assert math.isnan(table.loc[0, "p_holm"])
    assert table.loc[0, "diff_vs_best"] == 0.0
    assert table.loc[2, "diff_vs_best"] == pytest.approx(-0.5)
    assert table.loc[1, "p_holm"] >= table.loc[1, "p_value"]
    assert (table["ci_lo"] <= table["mean"]).all() and (table["mean"] <= table["ci_hi"]).all()
    comparisons = compare_to_best(df, "t", "match", n_boot=500)
    assert set(comparisons) == {"m2", "m3"}
    text = render_markdown(table, comparisons)
    lines = text.splitlines()
    assert lines[0].startswith("| Model |")
    assert "| m1 | 0.750 |" in text
    assert "| m2 | 0.250 |" in text
    assert "m2 (m2 - m1)" in text
    assert render_markdown(table).count("\n") == 4


SAMPLES = MemoryDataset(
    [Sample(input=f"question {i}", target=str(i), id=f"q{i}") for i in range(6)]
)


@task
def _paired_task() -> Task:
    return Task(dataset=SAMPLES, solver=generate(), scorer=match())


def _answers(correct: int):
    def respond(messages, tools, tool_choice, config) -> ModelOutput:
        index = int(messages[-1].text.split()[-1])
        return ModelOutput.from_content("mockllm", str(index) if index < correct else "wrong")

    return respond


def test_integration_two_mockllm_models(log_dir: str) -> None:
    for model, correct in (("mockllm/model-a", 5), ("mockllm/model-b", 2)):
        log = eval(
            _paired_task(),
            model=model,
            model_args={"custom_outputs": _answers(correct)},
            log_dir=log_dir,
            display="none",
            epochs=2,
        )[0]
        assert log.status == "success"

    df = load_scores(log_dir)
    assert list(df.columns) == [
        "task",
        "model",
        "eval_id",
        "sample_id",
        "epoch",
        "scorer",
        "value",
        "cluster",
    ]
    assert len(df) == 24
    assert set(df["model"]) == {"mockllm/model-a", "mockllm/model-b"}
    assert set(df["scorer"]) == {"match"}
    assert set(df["epoch"]) == {1, 2}
    assert set(df["sample_id"]) == {f"q{i}" for i in range(6)}
    assert (df["cluster"] == df["sample_id"]).all()
    assert set(df["value"]) == {0.0, 1.0}
    task_name = df["task"].iloc[0]

    result = compare_models(
        df,
        task_name,
        "match",
        baseline="mockllm/model-b",
        candidate="mockllm/model-a",
        n_boot=2000,
    )
    assert result.mean_diff == pytest.approx(0.5)
    assert result.n == 12
    assert result.n_clusters == 6
    assert (result.wins, result.losses, result.ties) == (6, 0, 6)
    # only three questions separate the models, so the cluster-level sign-flip
    # test has 2^3 patterns and the two-sided p-value is 2/8
    assert result.p_value == pytest.approx(0.25, abs=0.03)

    table = leaderboard(df, task_name, "match", n_boot=2000)
    assert table["model"].tolist() == ["mockllm/model-a", "mockllm/model-b"]
    assert table["mean"].tolist() == pytest.approx([5 / 6, 2 / 6])
    assert table.loc[1, "diff_vs_best"] == pytest.approx(-0.5)
    assert table["n"].tolist() == [12, 12]
    assert table["n_clusters"].tolist() == [6, 6]
    text = render_markdown(table)
    assert "mockllm/model-a" in text and "mockllm/model-b" in text
