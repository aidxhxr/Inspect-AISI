"""Model comparisons built from Inspect eval logs.

`load_scores` flattens a log directory into one row per (task, model,
sample, epoch, scorer). `compare_models` and `leaderboard` then apply the
paired, clustered statistics from `aisi_evals.analysis.stats`.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from statistics import NormalDist

import numpy as np
import pandas as pd
from inspect_ai.analysis import evals_df, samples_df, score_to_float

from aisi_evals.analysis.stats import (
    PairedResult,
    clustered_se,
    holm_correction,
    is_binary,
    mean_ci,
    paired_difference,
    wilson_interval,
)

SCORE_COLUMNS = ["task", "model", "eval_id", "sample_id", "epoch", "scorer", "value", "cluster"]
_SCORE_PREFIX = "score_"


def load_scores(log_dir: str | Sequence[str]) -> pd.DataFrame:
    """Load every scored sample under `log_dir` as a long dataframe.

    Columns: task, model, eval_id, sample_id, epoch, scorer, value, cluster.
    `sample_id` is the dataset sample id (shared across epochs) and `cluster`
    equals it, so repeated answers to one question cluster together. Score
    values are converted with Inspect's `value_to_float`; rows whose scorer
    did not run are dropped.
    """
    samples = samples_df(log_dir)
    evals = evals_df(log_dir)
    if samples.empty or evals.empty:
        return pd.DataFrame(columns=SCORE_COLUMNS)
    score_cols = [c for c in samples.columns if c.startswith(_SCORE_PREFIX)]
    if not score_cols:
        return pd.DataFrame(columns=SCORE_COLUMNS)
    samples = score_to_float(score_cols)(samples)
    merged = samples.merge(
        evals[["eval_id", "task_name", "model"]], on="eval_id", how="left", validate="m:1"
    )
    long = merged.melt(
        id_vars=["task_name", "model", "eval_id", "id", "epoch"],
        value_vars=score_cols,
        var_name="scorer",
        value_name="value",
    )
    long = long.rename(columns={"task_name": "task", "id": "sample_id"})
    long["scorer"] = long["scorer"].str[len(_SCORE_PREFIX) :]
    long["value"] = pd.to_numeric(long["value"], errors="coerce").astype(float)
    long = long.dropna(subset=["value"]).reset_index(drop=True)
    long["task"] = long["task"].astype(str)
    long["model"] = long["model"].astype(str)
    long["epoch"] = long["epoch"].astype(int)
    long["sample_id"] = long["sample_id"].astype(object)
    long["cluster"] = long["sample_id"]
    return long[SCORE_COLUMNS]


def _select(df: pd.DataFrame, task: str, scorer: str) -> pd.DataFrame:
    missing = [c for c in SCORE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"scores dataframe is missing columns {missing}")
    subset = df[(df["task"] == task) & (df["scorer"] == scorer)]
    if subset.empty:
        tasks = sorted(df["task"].astype(str).unique())
        scorers = sorted(df["scorer"].astype(str).unique())
        raise ValueError(
            f"no scores for task={task!r} scorer={scorer!r}; tasks={tasks}, scorers={scorers}"
        )
    return subset


def _model_series(subset: pd.DataFrame, model: str) -> pd.Series:
    rows = subset[subset["model"] == model]
    if rows.empty:
        models = sorted(subset["model"].unique())
        raise ValueError(f"model {model!r} has no scores; models present: {models}")
    keyed = rows.set_index(["sample_id", "epoch"])["value"]
    if keyed.index.has_duplicates:
        dupes = keyed.index[keyed.index.duplicated()].unique().tolist()[:5]
        raise ValueError(
            f"model {model!r} has more than one score per (sample_id, epoch), e.g. {dupes}; "
            "it was probably evaluated more than once in this log directory"
        )
    return keyed.sort_index()


def compare_models(
    df: pd.DataFrame,
    task: str,
    scorer: str,
    baseline: str,
    candidate: str,
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> PairedResult:
    """Paired comparison candidate - baseline, aligned on (sample_id, epoch).

    Raises ValueError when the two models did not answer the same
    (sample_id, epoch) pairs. Differences are clustered by sample_id.
    """
    subset = _select(df, task, scorer)
    base = _model_series(subset, baseline)
    cand = _model_series(subset, candidate)
    if not base.index.equals(cand.index):
        only_base = base.index.difference(cand.index).tolist()[:5]
        only_cand = cand.index.difference(base.index).tolist()[:5]
        raise ValueError(
            f"{baseline!r} and {candidate!r} did not run the same samples on task {task!r}: "
            f"{len(base)} vs {len(cand)} rows; only in baseline {only_base}; "
            f"only in candidate {only_cand}"
        )
    clusters = [key[0] for key in cand.index]
    result = paired_difference(
        cand.to_numpy(), base.to_numpy(), n_boot=n_boot, seed=seed, alpha=alpha, clusters=clusters
    )
    return dataclasses.replace(result, name_a=candidate, name_b=baseline)


def _effective_n(values: np.ndarray, se: float) -> float:
    """Korn-Graubard effective sample size for a clustered proportion."""
    n = float(values.size)
    p = float(values.mean())
    numerator = p * (1.0 - p)
    if numerator > 0.0 and se > 0.0:
        return min(numerator / (se * se), n)
    return n


def _summary_row(model: str, rows: pd.DataFrame, alpha: float) -> dict[str, object]:
    values = rows["value"].to_numpy(dtype=float)
    clusters = rows["cluster"].tolist()
    se = clustered_se(values, clusters)
    mean, lo, hi, _ = mean_ci(values, alpha)
    if is_binary(values):
        lo, hi = wilson_interval(mean, _effective_n(values, se), alpha)
    else:
        half = NormalDist().inv_cdf(1.0 - alpha / 2.0) * se
        lo, hi = mean - half, mean + half
    return {
        "model": model,
        "mean": mean,
        "se": se,
        "ci_lo": lo,
        "ci_hi": hi,
        "n": int(values.size),
        "n_clusters": int(len(set(clusters))),
    }


def compare_to_best(
    df: pd.DataFrame,
    task: str,
    scorer: str,
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, PairedResult]:
    """Paired comparison of every model against the highest-mean model.

    Keys are model names; each result is model minus best. The best model
    itself is not included.
    """
    subset = _select(df, task, scorer)
    means = subset.groupby("model")["value"].mean().sort_values(ascending=False)
    best = str(means.index[0])
    return {
        str(model): compare_models(
            df,
            task,
            scorer,
            baseline=best,
            candidate=str(model),
            n_boot=n_boot,
            seed=seed,
            alpha=alpha,
        )
        for model in means.index[1:]
    }


def leaderboard(
    df: pd.DataFrame,
    task: str,
    scorer: str,
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Per-model summary for one task and scorer, best model first.

    Columns: model, mean, se (clustered by sample_id), ci_lo, ci_hi (Wilson
    with clustered effective n for binary scores, normal otherwise), n,
    n_clusters, diff_vs_best, diff_ci_lo, diff_ci_hi, p_value, p_holm,
    wins, losses, ties. The best row has diff 0 and NaN p-values.
    """
    subset = _select(df, task, scorer)
    rows = [
        _summary_row(str(model), group, alpha)
        for model, group in subset.groupby("model", sort=False)
    ]
    table = pd.DataFrame(rows).sort_values(["mean", "model"], ascending=[False, True])
    table = table.reset_index(drop=True)
    comparisons = compare_to_best(df, task, scorer, n_boot=n_boot, seed=seed, alpha=alpha)
    others = [m for m in table["model"] if m in comparisons]
    adjusted = holm_correction([comparisons[m].p_value for m in others])
    holm = dict(zip(others, adjusted, strict=True))
    cols: dict[str, list[float]] = {
        "diff_vs_best": [],
        "diff_ci_lo": [],
        "diff_ci_hi": [],
        "p_value": [],
        "p_holm": [],
        "wins": [],
        "losses": [],
        "ties": [],
    }
    for model in table["model"]:
        result = comparisons.get(model)
        if result is None:
            for key in cols:
                cols[key].append(0.0 if key.startswith("diff") else float("nan"))
            continue
        cols["diff_vs_best"].append(result.mean_diff)
        cols["diff_ci_lo"].append(result.ci_lo)
        cols["diff_ci_hi"].append(result.ci_hi)
        cols["p_value"].append(result.p_value)
        cols["p_holm"].append(holm[model])
        cols["wins"].append(result.wins)
        cols["losses"].append(result.losses)
        cols["ties"].append(result.ties)
    for key, column in cols.items():
        table[key] = column
    return table


def _fmt(x: object, digits: int = 3) -> str:
    if isinstance(x, (float, np.floating)):
        if math.isnan(x):
            return ""
        return f"{x:.{digits}f}"
    return str(x)


def render_markdown(
    leaderboard_df: pd.DataFrame,
    comparison_results: dict[str, PairedResult] | None = None,
) -> str:
    """Render a leaderboard (and optional pairwise results) as markdown tables."""
    header = ["Model", "Mean", "SE", "95% CI", "n", "Diff vs best", "Diff CI", "p (Holm)", "W/L/T"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for _, row in leaderboard_df.iterrows():
        is_best = isinstance(row.get("p_holm"), float) and math.isnan(row["p_holm"])
        wlt = "" if is_best else f"{int(row['wins'])}/{int(row['losses'])}/{int(row['ties'])}"
        cells = [
            str(row["model"]),
            _fmt(row["mean"]),
            _fmt(row["se"]),
            f"[{_fmt(row['ci_lo'])}, {_fmt(row['ci_hi'])}]",
            str(int(row["n"])),
            "" if is_best else _fmt(row["diff_vs_best"], 3),
            "" if is_best else f"[{_fmt(row['diff_ci_lo'])}, {_fmt(row['diff_ci_hi'])}]",
            "" if is_best else _fmt(row["p_holm"], 4),
            wlt,
        ]
        lines.append("| " + " | ".join(cells) + " |")
    if comparison_results:
        lines.append("")
        pair_header = ["Comparison", "Mean diff", "SE", "95% CI", "p", "n", "W/L/T"]
        lines.append("| " + " | ".join(pair_header) + " |")
        lines.append("|" + "---|" * len(pair_header))
        for key, result in comparison_results.items():
            label = key if key else f"{result.name_a} - {result.name_b}"
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"{label} ({result.name_a} - {result.name_b})",
                        _fmt(result.mean_diff),
                        _fmt(result.se_paired),
                        f"[{_fmt(result.ci_lo)}, {_fmt(result.ci_hi)}]",
                        _fmt(result.p_value, 4),
                        str(result.n),
                        f"{result.wins}/{result.losses}/{result.ties}",
                    ]
                )
                + " |"
            )
    return "\n".join(lines)
