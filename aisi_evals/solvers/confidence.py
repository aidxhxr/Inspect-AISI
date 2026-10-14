"""Prompting that asks for a final answer and a stated confidence."""

from __future__ import annotations

import re

from inspect_ai.solver import Solver, chain, generate, prompt_template, solver

ANSWER_RE = re.compile(r"^\s*ANSWER:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
CONFIDENCE_RE = re.compile(r"^\s*CONFIDENCE:\s*(\d{1,3})\s*%?\s*$", re.IGNORECASE | re.MULTILINE)
ABSTAIN_TOKENS = ("i don't know", "i do not know", "unknown", "cannot answer", "not sure")

CONFIDENCE_TEMPLATE = """Answer the question below as precisely as you can.

If you do not know the answer, say so instead of guessing.

Finish with exactly two lines:
ANSWER: <your final answer, or "I don't know">
CONFIDENCE: <integer from 0 to 100, your probability that ANSWER is correct>

Question: {prompt}
"""


def parse_answer(completion: str) -> str | None:
    """Final answer text, or None when the model abstained or gave no ANSWER line."""
    match = ANSWER_RE.findall(completion)
    if not match:
        return None
    answer = match[-1].strip().strip("\"'")
    if not answer or answer.lower().strip(".") in ABSTAIN_TOKENS:
        return None
    return answer


def parse_confidence(completion: str) -> float | None:
    """Stated confidence as a probability in [0, 1], or None when absent or malformed."""
    match = CONFIDENCE_RE.findall(completion)
    if not match:
        return None
    value = int(match[-1])
    if value > 100:
        return None
    return value / 100.0


@solver
def answer_with_confidence(template: str = CONFIDENCE_TEMPLATE) -> Solver:
    """Prompt for an answer plus a 0 to 100 confidence, then generate once."""
    return chain(prompt_template(template), generate())
