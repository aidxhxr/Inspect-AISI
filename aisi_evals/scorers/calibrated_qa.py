"""Three-way model-graded QA scorer that keeps the model's stated confidence."""

from __future__ import annotations

import re

from inspect_ai.model import get_model
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    NOANSWER,
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState

from aisi_evals.metrics import (
    abstention_rate,
    attempted_rate,
    aurc,
    brier,
    correct_given_attempted,
    ece,
    mean_confidence,
    selective_accuracy,
    simpleqa_f1,
)
from aisi_evals.solvers.confidence import parse_answer, parse_confidence

GRADER_TEMPLATE = """You are grading a short factual answer against a gold target.

Grade the predicted answer as one of:
A: CORRECT. It contains the gold target's information without contradicting it. Extra
   correct detail, different units or spelling, or a less precise but consistent value
   (within the precision the question implies) still count as CORRECT.
B: INCORRECT. It contradicts the gold target, hedges between several candidates, or
   states a different fact. Any factual contradiction is INCORRECT even if the gold
   target also appears.
C: NOT_ATTEMPTED. It gives no answer, says it does not know, or refuses.

Do not use outside knowledge to decide what is true; judge only against the gold target.

Question: {question}
Gold target: {target}
Predicted answer: {answer}

Reply with the single letter A, B or C.
"""

GRADE_RE = re.compile(r"\b([ABC])\b")
GRADE_VALUES = {"A": CORRECT, "B": INCORRECT, "C": NOANSWER}
GRADE_NAMES = {"A": "correct", "B": "incorrect", "C": "not_attempted"}


@scorer(
    metrics=[
        accuracy(),
        stderr(),
        correct_given_attempted(),
        attempted_rate(),
        abstention_rate(),
        simpleqa_f1(),
        mean_confidence(),
        brier(),
        ece(),
        aurc(),
        selective_accuracy(0.5),
    ]
)
def calibrated_qa(template: str = GRADER_TEMPLATE, grader_role: str = "grader") -> Scorer:
    """Grade correct / incorrect / not attempted and record stated confidence.

    An abstention in the answer scores NOANSWER without calling the grader. A
    missing or malformed confidence is recorded as None and the sample is left
    out of calibration metrics but still counts toward accuracy.
    """

    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion
        answer = parse_answer(completion)
        confidence = parse_confidence(completion)
        if answer is None:
            return Score(
                value=NOANSWER,
                answer=None,
                explanation="No answer given.",
                metadata={"grade": "not_attempted", "confidence": confidence},
            )
        prompt = template.format(question=state.input_text, target=target.text, answer=answer)
        result = await get_model(role=grader_role).generate(prompt)
        match = GRADE_RE.search(result.completion.strip().upper())
        letter = match.group(1) if match else "C"
        return Score(
            value=GRADE_VALUES[letter],
            answer=answer,
            explanation=result.completion,
            metadata={"grade": GRADE_NAMES[letter], "confidence": confidence},
        )

    return score
