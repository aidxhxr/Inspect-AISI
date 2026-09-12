from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import chain_of_thought, generate

@task
def security_guide():
    return Task(
        dataset = json_dataset("security_guide.json"),
        solver = [chain_of_thought(), generate()],
        scorer = model_graded_fact()            
    )