from inspect_ai import Task, task
from inspect_ai.solver import generate, use_tools
from inspect_ai.tool import bash, python
from inspect_ai.scorer import includes

@task
def ctf():
    return Task(
        dataset = read_dataset(),
        solver = [use_tools([bash(timeout=180), python(timeout=180)]), generate()],
        sandbox = "docker",
        scorer = includes()
    )