"""Statistical comparison of Inspect eval results."""

from aisi_evals.analysis.compare import (
    SCORE_COLUMNS,
    compare_models,
    compare_to_best,
    leaderboard,
    load_scores,
    render_markdown,
)
from aisi_evals.analysis.stats import (
    PairedResult,
    clt_se,
    clustered_se,
    holm_correction,
    is_binary,
    mean_ci,
    min_detectable_difference,
    paired_difference,
    required_n,
    wilson_interval,
)

__all__ = [
    "PairedResult",
    "SCORE_COLUMNS",
    "clt_se",
    "clustered_se",
    "compare_models",
    "compare_to_best",
    "holm_correction",
    "is_binary",
    "leaderboard",
    "load_scores",
    "mean_ci",
    "min_detectable_difference",
    "paired_difference",
    "render_markdown",
    "required_n",
    "wilson_interval",
]
