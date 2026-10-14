import re

import pytest
from inspect_ai import eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ModelOutput

from aisi_evals.tasks import hellaswag_permuted, math500, simpleqa_calibrated
from aisi_evals.tasks.permute import expand_permutations


def mc_samples():
    return MemoryDataset(
        [
            Sample(input=f"q{i}", choices=["alpha", "beta", "gamma", "delta"], target="B", id=i)
            for i in range(3)
        ],
        name="mc",
    )


def test_expand_permutations_tracks_target_position():
    ds = expand_permutations(mc_samples(), n_permutations=4, seed=1)
    assert len(ds) == 12
    for s in ds:
        pos = s.metadata["target_position"]
        assert s.choices[pos] == "beta"
        assert s.target == chr(ord("A") + pos)
        assert s.metadata["original_id"] in {0, 1, 2}
    firsts = [s for s in ds if s.metadata["permutation"] == 0]
    assert all(s.choices == ["alpha", "beta", "gamma", "delta"] for s in firsts)


def test_expand_permutations_requires_choices():
    with pytest.raises(ValueError):
        expand_permutations(MemoryDataset([Sample(input="x", target="A", id=1)]))


def test_hellaswag_permuted_consistency_and_position_bias(log_dir):
    task = hellaswag_permuted(dataset=mc_samples(), n_permutations=2, seed=3)

    def respond(input, tools, tool_choice, config):
        prompt = input[-1].text
        if "q2" in prompt:
            return ModelOutput.from_content("mockllm", "ANSWER: A")
        letter = re.search(r"([A-D])\) beta", prompt).group(1)
        return ModelOutput.from_content("mockllm", f"ANSWER: {letter}")

    log = eval(
        task,
        model="mockllm/model",
        model_args={"custom_outputs": respond},
        log_dir=log_dir,
        display="none",
    )[0]
    assert log.status == "success"
    m = {k: v.value for k, v in log.results.scores[0].metrics.items()}
    assert m["accuracy"] == pytest.approx(5 / 6)
    assert m["permutation_consistency"] == pytest.approx(2 / 3)
    assert m["stderr"] > 0.0
    chosen = {s.id: s.scores["permuted_choice"].metadata["chosen"] for s in log.samples}
    assert all(v == "beta" for k, v in chosen.items() if not str(k).startswith("2-"))


def test_tasks_accept_dataset_override_without_network():
    qa = simpleqa_calibrated(dataset=MemoryDataset([Sample(input="q", target="a", id=1)]))
    math = math500(dataset=MemoryDataset([Sample(input="q", target="1", id=1)]))
    assert len(qa.dataset) == 1 and len(math.dataset) == 1
    assert qa.scorer is not None and math.scorer is not None
