# CLAUDE.md

## What this folder is

A personal learning sandbox for **Inspect**, the open-source LLM evaluation framework from the UK AI Security Institute (AISI). Each `*.py` file at the top level is a standalone Inspect task written while working through the docs. There is no package, no tests, and no build step. Prefer small, self-contained example scripts over shared abstractions; the goal is understanding the framework, not building a library.

When helping here, explain *why* Inspect works the way it does (what a solver is, why a scorer returns a `Score`, etc.), not just what to type. Prefer the idioms shown in the official docs over clever alternatives.

## Documentation (source of truth)

Always ground answers in the official docs rather than memory. Inspect moves fast and APIs change.

- Index of all pages: https://inspect.aisi.org.uk/llms.txt
- Full guide in one file (~1 MB): https://inspect.aisi.org.uk/llms-guide.txt
- Every page is also available as Markdown by appending `.md` to the HTML URL, e.g. `https://inspect.aisi.org.uk/tasks.html.md`, `https://inspect.aisi.org.uk/reference/inspect_ai.solver.html.md`.

Most useful pages while learning: `tutorial`, `tasks`, `datasets`, `solvers`, `scorers`, `agents`, `react-agent`, `tools`, `sandboxing`, `options`, `log-viewer`, `eval-logs`, `dataframe`.

Other references:
- Examples: https://github.com/UKGovernmentBEIS/inspect_ai/tree/main/examples
- Ready-made evals: https://github.com/UKGovernmentBEIS/inspect_evals

## Environment

- Python 3.13 (system framework install, no venv). `inspect_ai` 0.3.263 is installed and the `inspect` CLI is on PATH.
- Installed provider SDKs: `openai` only. Using `anthropic/...`, `google/...`, etc. requires `pip install anthropic` (or the relevant SDK) first.
- Also installed: `pandas`, `datasets` (needed for `hf_dataset`).
- `.env` in this folder holds `OPENAI_API_KEY`. Inspect reads `.env` automatically. Never print, commit, or copy its contents. Add `INSPECT_EVAL_MODEL=openai/<model>` there to avoid passing `--model` every time.
- Docker Desktop is installed but the daemon is often **not running**. Any task with `sandbox="docker"` needs it started first (`open -a Docker`, then wait for `docker info` to succeed). Inspect requires Docker Engine >= 24.0.6 and Compose >= 2.21.0.
- Logs go to `./logs/` (`.eval` files). Do not edit them by hand.

## Core concepts (Inspect vocabulary)

A **Task** = **dataset** + **solver** + **scorer** (+ optional `sandbox`, limits, config).

- **Sample**: one row of a dataset. Required field `input`; optional `target`, `choices`, `id`, `metadata`, `files` (copied into the sandbox), `setup`, `sandbox`.
- **Dataset**: `json_dataset()`, `csv_dataset()`, `hf_dataset()`, `example_dataset()`, or a plain `list[Sample]`. Map foreign column names with `FieldSpec(input=..., target=...)` or a `record_to_sample()` function.
- **Solver**: an async function that transforms `TaskState` (`system_message()`, `prompt_template()`, `chain_of_thought()`, `use_tools()`, `generate()`, `multiple_choice()`). Solvers compose in a list. An **agent** such as `react()` can be passed directly as the solver.
- **Scorer**: async `score(state, target) -> Score`. Built-ins: `includes()`, `match()`, `exact()`, `choice()`, `model_graded_qa()`, `model_graded_fact()`. Custom scorers use `@scorer(metrics=[accuracy(), stderr()])`.
- **Tool**: `@tool`-decorated function returning an async `execute()` with type hints and an `Args:` docstring (both required). Built-ins that need a sandbox: `bash()`, `python()`, `text_editor()`, `web_browser()`.
- **Sandbox**: `sandbox="docker"` runs tool commands in a container. With no `Dockerfile` or `compose.yaml` next to the task, Inspect uses the default `aisiuk/inspect-tool-support` image with `network_mode: none`. A custom `compose.yaml` *replaces* the generated one, so add `network_mode: none` yourself unless networking is needed.
- **Limits**: `message_limit`, `token_limit`, `time_limit`, `working_limit`, `cost_limit` on `Task()` or via CLI. Always set at least a `message_limit` on agent tasks.

## Common commands

```bash
# run a task (task discovered via @task in the file)
inspect eval simpleqa.py --model openai/gpt-5

# develop cheaply: few samples, one sample, or a range
inspect eval simpleqa.py --limit 5
inspect eval simpleqa.py --sample-id 3
inspect eval simpleqa.py --limit 10-20

# pass task arguments (parameters of the @task function)
inspect eval ctf.py -T attempts=1 -T message_limit=20

# repeat samples, pick a grader model, set limits
inspect eval simpleqa.py --epochs 3 --model-role grader=openai/gpt-5-mini
inspect eval ctf.py --message-limit 30 --token-limit 200000

# tolerate errors instead of failing the run
inspect eval ctf.py --no-fail-on-error --retry-on-error=2

# browse logs (starts once, auto-refreshes; default http://127.0.0.1:7575)
inspect view
inspect view --log-dir ./logs --port 6565

# re-score an existing log without re-running the model
inspect score logs/<file>.eval --scorer includes

# run several tasks / models with retry + resume
inspect eval-set simpleqa.py ctf.py --model openai/gpt-5 --log-dir logs/run-1

# debug a failing sample by raising instead of logging
inspect eval ctf.py --debug-errors --limit 1
```

Analysis in Python:

```python
from inspect_ai.analysis import evals_df, samples_df
evals = evals_df("logs")      # one row per run
samples = samples_df("logs")  # one row per sample
```

## Conventions for this folder

- One task per file, named after the eval (`simpleqa.py`, `CTF.py`). Task function name should match the file (lowercase) so `inspect eval file.py` finds it without `@name`.
- Give `@task` functions parameters with defaults (e.g. `attempts=3, message_limit=30`) so they can be tuned with `-T` instead of editing code.
- Keep dataset files next to the task that uses them (`json_dataset("challenges.json")` resolves relative to the task file). Sample `files` paths resolve relative to the dataset file.
- For Docker tasks, put the `Dockerfile` / `compose.yaml` in the same directory as the task file; Inspect discovers them automatically. If several Docker tasks need different images, move each into its own subfolder.
- Use `--limit` while iterating. Full runs on Hugging Face datasets are slow and cost money.
- Do not add new dependencies without saying so; this is a bare system Python.

## Current state / known gaps

- `simpleqa.py`: Hugging Face `codelion/SimpleQA-Verified`, `generate()` solver, `model_graded_qa()` scorer. Has been run once (see `logs/`).
- `CTF.py`: `react()` agent with `bash()` + `todo_write()`, `includes()` scorer, `sandbox="docker"`. It references `challenges.json`, which **does not exist yet**, and it has no `Dockerfile`/`compose.yaml` (so it would use the default tool-support image). It also lacks a `message_limit`. Expect it to fail until the dataset is created and Docker is running.
