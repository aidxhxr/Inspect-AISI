# Inspect-AISI

Evaluation tasks built on [Inspect](https://inspect.aisi.org.uk), the UK AI
Security Institute's evaluation framework, with the statistics needed to say
whether one model is actually better than another.

The suite measures four things that a single accuracy number hides:

| Task | What it measures beyond accuracy |
| --- | --- |
| `simpleqa_calibrated` | Whether stated confidence tracks correctness (Brier, ECE, risk-coverage), and whether the model knows when to abstain (SimpleQA F1) |
| `math500` | How much of the score depends on a model grader, via an exact-then-sympy-then-grader cascade that records which stage decided each sample |
| `hellaswag_permuted` | Whether answers survive reordering the options (permutation consistency, position bias), with standard errors clustered by question |
| `ctf` | Whether an agent with bash and python tools can solve sandboxed challenges, reported per difficulty level |

`aisi-evals compare` then turns the logs into paired comparisons with
bootstrap confidence intervals, sign-flip p-values and Holm correction, and
`aisi-evals power` says how many samples a difference needs before you run.
The reasoning behind each measurement is in [docs/methodology.md](docs/methodology.md).

## Setup

```bash
git clone https://github.com/aidxhxr/Inspect-AISI && cd Inspect-AISI
uv venv && source .venv/bin/activate
make install
cp .env.example .env   # add OPENAI_API_KEY / ANTHROPIC_API_KEY, pick AISI_MODELS and AISI_GRADER
```

Python 3.11 or newer. The `ctf` task needs Docker Engine 24 or newer with Compose v2.

## Running

Everything, every model in `AISI_MODELS`, one log directory:

```bash
aisi-evals run --log-dir logs/suite
```

One task, one model, a small slice, from the Inspect CLI directly:

```bash
inspect eval aisi_evals/tasks/simpleqa_calibrated.py --model openai/gpt-5 --limit 100 \
  --model-role grader=openai/gpt-5
inspect eval aisi_evals/tasks/hellaswag_permuted.py -T n_permutations=4 --limit 200
inspect eval aisi_evals/ctf/task.py -T max_difficulty=3
inspect view
```

`aisi-evals run` wraps `eval_set`, so an interrupted run resumes from the
same log directory and only the missing samples are retried.

## Comparing models

```bash
aisi-evals compare logs/suite --output reports/leaderboard.md
aisi-evals power --n 500            # minimum difference detectable with 500 samples
aisi-evals power --diff 0.03        # samples needed to detect a 3 point gap
```

The comparison aligns models on (sample, epoch), so every model in a log
directory must have answered the same samples; it refuses to compare runs
that did not.

Logs are not committed. Results for a given model set are whatever the log
directory contains after a run.

## Layout

```
aisi_evals/
  tasks/        simpleqa_calibrated, math500, hellaswag_permuted, permutation expansion
  scorers/      calibrated_qa (3-way grade + confidence), math_equivalence cascade, permuted_choice
  solvers/      confidence elicitation prompt and parsers
  metrics/      Brier, ECE, AURC, selective accuracy, SimpleQA F1, consistency, position bias
  analysis/     clustered SE, paired bootstrap, sign-flip test, Holm, power, leaderboard
  ctf/          five Docker challenges, dataset loader, react-agent task
  cli.py        run / compare / power
tests/          offline, no keys: mock-model end-to-end runs for every task and scorer
docs/           methodology
```

## Testing

```bash
make test   # pytest, no network or API keys required
make lint   # ruff
```

The tests run each task end to end against Inspect's mock model provider
with scripted outputs, including a full CTF run with the sandbox replaced by
scripted tool calls, and check the clustered standard error against Inspect's
own implementation.

## References

- Miller, E. (2024). Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations. arXiv:2411.00640.
- Wei, J. et al. (2024). Measuring short-form factuality in large language models (SimpleQA). OpenAI.
- Zellers, R. et al. (2019). HellaSwag: Can a Machine Really Finish Your Sentence? ACL.
- Lightman, H. et al. (2023). Let's Verify Step by Step (MATH-500 subset). arXiv:2305.20050.
- Inspect documentation: https://inspect.aisi.org.uk
