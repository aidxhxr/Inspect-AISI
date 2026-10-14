"""Multiple-choice scorer that records which option text the model chose."""

from __future__ import annotations

from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState

from aisi_evals.metrics import permutation_consistency, position_bias


@scorer(
    metrics=[
        accuracy(),
        stderr(cluster="original_id"),
        permutation_consistency(),
        position_bias(),
    ]
)
def permuted_choice() -> Scorer:
    """Score a `multiple_choice` answer and store the chosen option text in metadata.

    Requires samples produced by `aisi_evals.tasks.permute.expand_permutations`,
    which sets `original_id` and `target_position` in sample metadata.
    """

    async def score(state: TaskState, target: Target) -> Score:
        selected = [i for i in range(len(state.choices)) if state.choices[i].correct]
        if len(selected) != 1:
            return Score(
                value=INCORRECT,
                answer=None,
                explanation="No single choice selected.",
                metadata={"chosen": None},
            )
        index = selected[0]
        letter = chr(ord("A") + index)
        return Score(
            value=CORRECT if letter == target.text else INCORRECT,
            answer=letter,
            metadata={"chosen": state.choices[index].value},
        )

    return score
