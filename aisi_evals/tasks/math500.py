"""MATH-500 with a normalize, sympy, grader cascade."""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, FieldSpec, hf_dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate, prompt_template

from aisi_evals.scorers import math_equivalence

DATASET = "HuggingFaceH4/MATH-500"

PROMPT_TEMPLATE = """Solve the problem. Reason step by step, then end with a final line of the
form "ANSWER: <answer>" containing only the final answer.

{prompt}
"""


def math500_dataset(limit: int | None = None, shuffle: bool = True, seed: int = 0) -> Dataset:
    return hf_dataset(
        DATASET,
        split="test",
        sample_fields=FieldSpec(
            input="problem", target="answer", id="unique_id", metadata=["level", "subject"]
        ),
        shuffle=shuffle,
        seed=seed,
        limit=limit,
    )


@task
def math500(limit: int | None = None, seed: int = 0, dataset: Dataset | None = None) -> Task:
    """Competition math scored by exact match, then sympy, then a model grader.

    The `decided_by` breakdown shows what fraction of verdicts relied on the grader.
    """
    return Task(
        dataset=dataset or math500_dataset(limit=limit, seed=seed),
        solver=[prompt_template(PROMPT_TEMPLATE), generate()],
        scorer=math_equivalence(),
        config=GenerateConfig(temperature=0.0),
    )
