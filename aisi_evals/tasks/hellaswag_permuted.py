"""HellaSwag under option permutation: accuracy, consistency and position bias."""

from __future__ import annotations

from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, Sample, hf_dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import multiple_choice

from aisi_evals.scorers import permuted_choice
from aisi_evals.tasks.permute import expand_permutations

DATASET = "Rowan/hellaswag"


def record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        input=record["ctx"],
        choices=record["endings"],
        target=chr(ord("A") + int(record["label"])),
        id=record["ind"],
        metadata={"activity": record["activity_label"]},
    )


def hellaswag_dataset(limit: int | None = None, shuffle: bool = True, seed: int = 0) -> Dataset:
    return hf_dataset(
        DATASET,
        split="validation",
        sample_fields=record_to_sample,
        shuffle=shuffle,
        seed=seed,
        limit=limit,
        trust=True,
    )


@task
def hellaswag_permuted(
    limit: int | None = None,
    n_permutations: int = 4,
    seed: int = 0,
    dataset: Dataset | None = None,
) -> Task:
    """Each question is asked `n_permutations` times with shuffled options.

    Standard errors are clustered by question. `permutation_consistency` is the
    fraction of questions answered identically under every ordering and
    `position_bias` is the accuracy gap between the best and worst position of
    the correct option.
    """
    base = dataset or hellaswag_dataset(limit=limit, seed=seed)
    return Task(
        dataset=expand_permutations(base, n_permutations=n_permutations, seed=seed),
        solver=multiple_choice(),
        scorer=permuted_choice(),
        config=GenerateConfig(temperature=0.0),
    )
