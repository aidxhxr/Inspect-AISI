"""SimpleQA-Verified with confidence elicitation and calibration metrics."""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, FieldSpec, hf_dataset
from inspect_ai.model import GenerateConfig

from aisi_evals.scorers import calibrated_qa
from aisi_evals.solvers import answer_with_confidence

DATASET = "codelion/SimpleQA-Verified"


def simpleqa_dataset(limit: int | None = None, shuffle: bool = True, seed: int = 0) -> Dataset:
    return hf_dataset(
        DATASET,
        split="eval",
        sample_fields=FieldSpec(input="problem", target="answer", metadata=["topic"]),
        shuffle=shuffle,
        seed=seed,
        limit=limit,
    )


@task
def simpleqa_calibrated(
    limit: int | None = None,
    seed: int = 0,
    dataset: Dataset | None = None,
) -> Task:
    """Short-form factuality with stated confidence.

    Reports accuracy, SimpleQA F1, abstention rate, Brier score, ECE, AURC and
    selective accuracy at 50% coverage. Grading uses the `grader` model role.
    """
    return Task(
        dataset=dataset or simpleqa_dataset(limit=limit, seed=seed),
        solver=answer_with_confidence(),
        scorer=calibrated_qa(),
        config=GenerateConfig(temperature=0.0),
    )
