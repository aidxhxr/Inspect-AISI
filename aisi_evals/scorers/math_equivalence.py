"""Cascade scorer for math answers: normalize, compare symbolically, then ask a grader."""

from __future__ import annotations

import re

import sympy
from inspect_ai.model import get_model
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Metric,
    SampleScore,
    Score,
    Scorer,
    Target,
    accuracy,
    metric,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

ANSWER_RE = re.compile(r"ANSWER:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)

EQUIVALENCE_TEMPLATE = """Decide whether two final answers to the same math problem are equivalent.

Treat answers as equivalent when they denote the same mathematical object, allowing for
different but equal forms (fractions vs decimals when exact, factored vs expanded,
reordered sets or tuples when order is irrelevant, units written differently).
Do not solve the problem; compare only the two answers.

Problem: {problem}
Answer 1: {expected}
Answer 2: {given}

Reply with exactly one word: Yes or No.
"""

_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def normalize(text: str) -> str:
    """Strip formatting that does not change the value of a final answer."""
    s = text.strip()
    s = re.sub(r"\\boxed\{(.*)\}", r"\1", s)
    s = re.sub(r"\\(text|mathrm|textbf)\{([^}]*)\}", r"\2", s)
    s = s.replace("\\left", "").replace("\\right", "").replace("\\!", "").replace("\\,", "")
    s = s.replace("$", "").replace("\\%", "%").replace("\\$", "")
    s = re.sub(r"\s+", "", s)
    s = s.rstrip(".")
    s = re.sub(r"^(x|y|z|n|k|a|b)=", "", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\^\{?\\circ\}?|°", "", s)
    s = re.sub(r"(degrees|units|cm|inches|dollars)$", "", s)
    return s.lower()


def _parse(expr: str) -> sympy.Expr | None:
    if "\\" in expr:
        try:
            return parse_latex(expr)
        except Exception:
            return None
    try:
        return parse_expr(expr.replace("^", "**"), transformations=_TRANSFORMS)
    except Exception:
        return None


def symbolic_equal(a: str, b: str) -> bool | None:
    """True/False when both sides parse and compare; None when sympy cannot decide."""
    ea, eb = _parse(a), _parse(b)
    if ea is None or eb is None:
        return None
    try:
        diff = sympy.simplify(ea - eb)
    except Exception:
        return None
    if diff == 0:
        return True
    try:
        if diff.free_symbols:
            return False
        return bool(abs(complex(sympy.N(diff))) < 1e-9)
    except Exception:
        return None


@metric
def decided_by() -> Metric:
    """Share of verdicts settled at each cascade stage (exact, sympy, model)."""

    def compute(scores: list[SampleScore]) -> dict[str, float]:
        counts = {"exact": 0, "sympy": 0, "model": 0}
        for s in scores:
            stage = (s.score.metadata or {}).get("decided_by")
            if stage in counts:
                counts[stage] += 1
        total = len(scores) or 1
        return {k: v / total for k, v in counts.items()}

    return compute


@scorer(metrics=[accuracy(), stderr(), decided_by()])
def math_equivalence(grader_role: str = "grader") -> Scorer:
    """Exact match after normalization, then sympy, then a model grader as the last resort.

    `Score.metadata["decided_by"]` records which stage decided ("exact", "sympy"
    or "model") so the share of grader-dependent decisions can be reported.
    """

    async def score(state: TaskState, target: Target) -> Score:
        match = ANSWER_RE.findall(state.output.completion)
        if not match:
            return Score(
                value=INCORRECT,
                answer=None,
                explanation="No ANSWER line.",
                metadata={"decided_by": "exact"},
            )
        given = match[-1].strip()
        expected = target.text
        n_given, n_expected = normalize(given), normalize(expected)
        if n_given == n_expected:
            return Score(value=CORRECT, answer=given, metadata={"decided_by": "exact"})
        sym = symbolic_equal(n_given, n_expected)
        if sym is not None:
            return Score(
                value=CORRECT if sym else INCORRECT,
                answer=given,
                metadata={"decided_by": "sympy"},
            )
        prompt = EQUIVALENCE_TEMPLATE.format(
            problem=state.input_text, expected=expected, given=given
        )
        result = await get_model(role=grader_role).generate(prompt)
        verdict = result.completion.strip().lower().startswith("yes")
        return Score(
            value=CORRECT if verdict else INCORRECT,
            answer=given,
            explanation=result.completion,
            metadata={"decided_by": "model"},
        )

    return score
