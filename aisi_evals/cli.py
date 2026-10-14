"""Command line entry points: run the suite, compare models, print sample-size guidance."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

TASKS = {
    "simpleqa_calibrated": "aisi_evals/tasks/simpleqa_calibrated.py@simpleqa_calibrated",
    "math500": "aisi_evals/tasks/math500.py@math500",
    "hellaswag_permuted": "aisi_evals/tasks/hellaswag_permuted.py@hellaswag_permuted",
    "ctf": "aisi_evals/ctf/task.py@ctf",
}


def _models(arg: str | None) -> list[str]:
    raw = arg or os.environ.get("AISI_MODELS", "")
    models = [m.strip() for m in raw.split(",") if m.strip()]
    if not models:
        sys.exit("No models given. Pass --models or set AISI_MODELS in .env.")
    return models


def run(args: argparse.Namespace) -> int:
    from inspect_ai import eval_set
    from inspect_ai.model import get_model

    unknown = [name for name in args.tasks if name not in TASKS]
    if unknown:
        sys.exit(f"Unknown task(s): {', '.join(unknown)}. Choose from {', '.join(TASKS)}.")
    root = Path(__file__).resolve().parent.parent
    tasks = [str(root / TASKS[name]) for name in (args.tasks or list(TASKS))]
    task_args: dict[str, object] = {}
    if args.limit is not None:
        task_args["limit"] = args.limit
    grader_name = args.grader or os.environ.get("AISI_GRADER")
    model_roles = {"grader": get_model(grader_name)} if grader_name else None
    success, logs = eval_set(
        tasks,
        model=_models(args.models),
        task_args=task_args,
        model_roles=model_roles,
        log_dir=args.log_dir,
        epochs=args.epochs,
        max_connections=args.max_connections,
        retry_attempts=args.retry_attempts,
        fail_on_error=0.2,
    )
    print(f"{'complete' if success else 'incomplete'}: {len(logs)} logs in {args.log_dir}")
    return 0 if success else 1


def compare(args: argparse.Namespace) -> int:
    from aisi_evals.analysis import compare_to_best, leaderboard, load_scores, render_markdown

    df = load_scores(args.log_dir)
    if df.empty:
        sys.exit(f"No scored samples found under {args.log_dir}.")
    pairs = {(t, s) for t, s in zip(df["task"], df["scorer"], strict=True)}
    if args.task:
        pairs = {(t, s) for t, s in pairs if t == args.task}
    out: list[str] = []
    for task, scorer in sorted(pairs):
        board = leaderboard(df, task, scorer, n_boot=args.n_boot)
        results = compare_to_best(df, task, scorer, n_boot=args.n_boot)
        out.append(f"## {task} / {scorer}\n")
        out.append(render_markdown(board, results))
        out.append("")
    text = "\n".join(out)
    if args.output:
        Path(args.output).write_text(text)
        print(f"wrote {args.output}")
    else:
        print(text)
    return 0


def power(args: argparse.Namespace) -> int:
    from aisi_evals.analysis import min_detectable_difference, required_n

    if args.n is not None:
        mdd = min_detectable_difference(args.n, p=args.p, alpha=args.alpha, power=args.power)
        print(f"n={args.n}: minimum detectable difference {mdd:.3f}")
    if args.diff is not None:
        n = required_n(args.diff, p=args.p, alpha=args.alpha, power=args.power)
        print(f"diff={args.diff}: required n per model {n}")
    if args.n is None and args.diff is None:
        sys.exit("Pass --n and/or --diff.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aisi-evals")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run tasks across models with eval_set")
    p_run.add_argument("tasks", nargs="*", help=f"any of {', '.join(TASKS)}; default: all")
    p_run.add_argument("--models", help="comma separated, default AISI_MODELS")
    p_run.add_argument("--grader", help="grader model, default AISI_GRADER")
    p_run.add_argument("--limit", type=int)
    p_run.add_argument("--epochs", type=int, default=1)
    p_run.add_argument("--log-dir", default="logs/suite")
    p_run.add_argument("--max-connections", type=int, default=8)
    p_run.add_argument("--retry-attempts", type=int, default=3)
    p_run.set_defaults(func=run)

    p_cmp = sub.add_parser("compare", help="paired model comparison from a log directory")
    p_cmp.add_argument("log_dir")
    p_cmp.add_argument("--task")
    p_cmp.add_argument("--n-boot", type=int, default=10_000)
    p_cmp.add_argument("--output", help="write markdown here instead of stdout")
    p_cmp.set_defaults(func=compare)

    p_pow = sub.add_parser("power", help="sample size and minimum detectable difference")
    p_pow.add_argument("--n", type=int)
    p_pow.add_argument("--diff", type=float)
    p_pow.add_argument("--p", type=float, default=0.5, help="baseline accuracy")
    p_pow.add_argument("--alpha", type=float, default=0.05)
    p_pow.add_argument("--power", type=float, default=0.8)
    p_pow.set_defaults(func=power)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
