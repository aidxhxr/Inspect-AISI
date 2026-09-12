from inspect_ai import eval_set

success, logs = eval_set(
    tasks = [security_guide(), hellaswag(), math()],
    model = ["openai/gpt-5", "anthropic/claude-sonnet-4.6"],
    log_dir = "logs/run-1"
)