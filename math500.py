import re
from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec, hf_dataset
from inspect_ai.solver import generate, prompt_template
from inspect_ai.model import get_model
from inspect_ai.scorer import (
    CORRECT, INCORRECT, AnswerPattern, Score, Target,
    accuracy, scorer, stderr
)
from inspect_ai.solver import TaskState

EQUIVALENCE_TEMPLATE = """
Are these two expressions equivalent? Answer Yes or No.

Expression 1: %(expression1)s
Expression 2: %(expression2)s
"""

@scorer(metrics = [accuracy(), stderr()])
def expression_equivalence():
    async def score(state: TaskState, target: Target):
        # extract the model's answer from its output
        match = re.search(AnswerPattern.LINE, state.output.completion)
        if not match:
            return Score(
                value = INCORRECT,
                explanation = "No answer.",
            )
        # are answer and target equivalent?
        answer = match.group(1)
        prompt = EQUIVALENCE_TEMPLATE % {
            'expression1': target.text,
            'expression2': answer,
        }
        result = await get_model(role="grader").generate(prompt)

        # return score with answer and explanations
        correct = result.completion.strip().lower() == "yes"
        return Score(
            value = CORRECT if correct else INCORRECT,
            answer = answer,
            explanation = state.output.completion,
        )
    return score


PROMPT_TEMPLATE = """
Solve the following problem. The last line of your reply
should read "ANSWER: $ANSWER" (without quotes).

{prompt}
"""

@task
def math500(grader_role: str = "grader"):
    return Task(
        dataset=hf_dataset(
            "HuggingFaceH4/MATH-500",
            split="test",
            sample_fields=FieldSpec(input="problem", target="answer"),
        ),
        solver=[prompt_template(PROMPT_TEMPLATE), generate()],
        scorer=expression_equivalence(),
        )