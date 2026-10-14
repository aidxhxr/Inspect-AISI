import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import exact
from inspect_ai.solver import generate

from aisi_evals.cli import build_parser, main


def test_power_subcommand(capsys):
    assert main(["power", "--n", "400", "--diff", "0.1"]) == 0
    out = capsys.readouterr().out
    assert "minimum detectable difference" in out and "required n per model" in out


def test_run_subcommand_defaults_to_all_tasks():
    args = build_parser().parse_args(["run"])
    assert args.tasks == [] and args.log_dir == "logs/suite"
    args = build_parser().parse_args(["run", "math500", "ctf", "--limit", "5"])
    assert args.tasks == ["math500", "ctf"] and args.limit == 5


def test_run_rejects_unknown_task():
    with pytest.raises(SystemExit) as exc:
        main(["run", "nope", "--models", "mockllm/model"])
    assert "Unknown task" in str(exc.value)


def test_compare_subcommand_on_mock_logs(tmp_path, capsys):
    ds = MemoryDataset([Sample(input=f"q{i}", target="yes", id=i) for i in range(6)])

    def respond_for(correct_ids):
        def respond(input, tools, tool_choice, config):
            qid = int(input[-1].text.strip()[1:])
            return ModelOutput.from_content("mockllm", "yes" if qid in correct_ids else "no")

        return respond

    log_dir = str(tmp_path / "logs")
    for name, ids in (("model-a", {0, 1, 2, 3, 4}), ("model-b", {0, 1})):
        eval(
            Task(dataset=ds, solver=generate(), scorer=exact(), name="toy"),
            model=f"mockllm/{name}",
            model_args={"custom_outputs": respond_for(ids)},
            log_dir=log_dir,
            display="none",
        )
    out_file = tmp_path / "report.md"
    assert main(["compare", log_dir, "--n-boot", "200", "--output", str(out_file)]) == 0
    report = out_file.read_text()
    assert "toy" in report and "mockllm/model-a" in report and "mockllm/model-b" in report
