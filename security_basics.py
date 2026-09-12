from inspect_ai import Task, task
from inspect_ai.dataset import example_dataset
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import generate, system_message

@task
def security_guide(system="devops.txt", grader="expert.txt", grader_model="openai/gpt-4o"):
    return Task(
        dataset = example_dataset("security_guide"),
        solver = [system_message(system), generate()],
        scorer = model_graded_fact(template=grader, model=grader_model)
    )

