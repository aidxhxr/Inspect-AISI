import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.solver import generate

from aisi_evals.scorers import calibrated_qa, math_equivalence, normalize, symbolic_equal
from aisi_evals.solvers import answer_with_confidence, parse_answer, parse_confidence


def outputs(*texts):
    return [ModelOutput.from_content("mockllm", t) for t in texts]


def keyed(mapping):
    """Mock outputs chosen by which key appears in the last prompt (scheduler order varies)."""

    def respond(input, tools, tool_choice, config):
        prompt = input[-1].text
        for key, text in mapping.items():
            if key in prompt:
                return ModelOutput.from_content("mockllm", text)
        raise AssertionError(f"no mock output for prompt: {prompt[:80]!r}")

    return respond


def run(task, model_outputs, grader_outputs, log_dir):
    grader = get_model("mockllm/model", custom_outputs=keyed(grader_outputs))
    return eval(
        task,
        model="mockllm/model",
        model_args={"custom_outputs": keyed(model_outputs)},
        model_roles={"grader": grader},
        log_dir=log_dir,
        display="none",
    )[0]


@pytest.mark.parametrize(
    "text,answer,conf",
    [
        ("Reasoning...\nANSWER: Paris\nCONFIDENCE: 85%", "Paris", 0.85),
        ("ANSWER: 42\nCONFIDENCE: 100", "42", 1.0),
        ("ANSWER: I don't know\nCONFIDENCE: 10%", None, 0.10),
        ("no structured answer at all", None, None),
        ("ANSWER: 7\nCONFIDENCE: 140%", "7", None),
        ("ANSWER: first\nANSWER: final\nCONFIDENCE: 50%", "final", 0.5),
    ],
)
def test_parse_answer_and_confidence(text, answer, conf):
    assert parse_answer(text) == answer
    assert parse_confidence(text) == conf


def test_calibrated_qa_end_to_end(log_dir):
    ds = MemoryDataset(
        [
            Sample(input="Capital of France?", target="Paris", id="q1"),
            Sample(input="Capital of Peru?", target="Lima", id="q2"),
            Sample(input="Capital of Chad?", target="N'Djamena", id="q3"),
            Sample(input="Capital of Nauru?", target="Yaren", id="q4"),
        ]
    )
    task = Task(dataset=ds, solver=answer_with_confidence(), scorer=calibrated_qa())
    log = run(
        task,
        {
            "France": "ANSWER: Paris\nCONFIDENCE: 95%",
            "Peru": "ANSWER: Bogota\nCONFIDENCE: 60%",
            "Chad": "ANSWER: I don't know\nCONFIDENCE: 5%",
            "Nauru": "ANSWER: Yaren\nCONFIDENCE: 50%",
        },
        {"Paris": "A", "Bogota": "B", "Yaren": "A"},
        log_dir,
    )
    assert log.status == "success"
    m = {k: v.value for k, v in log.results.scores[0].metrics.items()}
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["abstention_rate"] == pytest.approx(0.25)
    assert m["correct_given_attempted"] == pytest.approx(2 / 3)
    assert m["brier"] == pytest.approx((0.05**2 + 0.6**2 + 0.5**2) / 3)
    assert m["attempted_rate"] == pytest.approx(0.75)
    values = {s.id: s.scores["calibrated_qa"].value for s in log.samples}
    assert values == {"q1": "C", "q2": "I", "q3": "N", "q4": "C"}
    assert log.samples[2].scores["calibrated_qa"].metadata["grade"] == "not_attempted"


def test_calibrated_qa_malformed_grade_counts_as_not_attempted(log_dir):
    ds = MemoryDataset([Sample(input="q", target="t", id="q1")])
    task = Task(dataset=ds, solver=answer_with_confidence(), scorer=calibrated_qa())
    log = run(task, {"q": "ANSWER: x\nCONFIDENCE: 10%"}, {"x": "no letter here"}, log_dir)
    assert log.samples[0].scores["calibrated_qa"].value == "N"


@pytest.mark.parametrize(
    "a,b",
    [
        ("\\boxed{12}", "12"),
        ("$x = 5$", "5"),
        ("12.", "12"),
        ("\\text{(A)}", "(a)"),
        ("90^\\circ", "90"),
    ],
)
def test_normalize_pairs(a, b):
    assert normalize(a) == normalize(b)


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ("1/2", "0.5", True),
        ("2*x+2", "2(x+1)", True),
        ("3", "4", False),
        ("x^2-1", "(x-1)(x+1)", True),
        ("x^2", "x^3", False),
        ("sqrt(8)", "2*sqrt(2)", True),
    ],
)
def test_symbolic_equal(a, b, expected):
    assert symbolic_equal(a, b) is expected


def test_symbolic_equal_never_confirms_unparseable_input():
    assert symbolic_equal("the pair (1, 2)", "(1,2)") is None
    assert symbolic_equal("twelve apples", "12") is not True


def test_math_equivalence_cascade_records_stage(log_dir):
    ds = MemoryDataset(
        [
            Sample(input="p1", target="\\frac{1}{2}", id="exact"),
            Sample(input="p2", target="0.25", id="sympy"),
            Sample(input="p3", target="(1,2)", id="model"),
            Sample(input="p4", target="9", id="none"),
        ]
    )
    task = Task(dataset=ds, solver=generate(), scorer=math_equivalence())
    log = run(
        task,
        {
            "p1": "ANSWER: \\frac{1}{2}",
            "p2": "ANSWER: 1/4",
            "p3": "ANSWER: the pair (1, 2)",
            "p4": "no answer line",
        },
        {"pair": "Yes"},
        log_dir,
    )
    assert log.status == "success"
    by_id = {s.id: s.scores["math_equivalence"] for s in log.samples}
    assert by_id["exact"].value == "C" and by_id["exact"].metadata["decided_by"] == "exact"
    assert by_id["sympy"].value == "C" and by_id["sympy"].metadata["decided_by"] == "sympy"
    assert by_id["model"].value == "C" and by_id["model"].metadata["decided_by"] == "model"
    assert by_id["none"].value == "I"
    m = {k: v.value for k, v in log.results.scores[0].metrics.items()}
    assert m["accuracy"] == pytest.approx(0.75)
    assert (m["exact"], m["sympy"], m["model"]) == (0.5, 0.25, 0.25)
