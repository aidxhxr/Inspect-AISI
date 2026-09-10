from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

def record_to_sample(record):
    return Sample(
        input = record["ctx"],
        choices = record["endings"],
        target = chr(ord("A") + int(record["label"])),
    )

@task
def hella_swag():
    return Task(
        dataset = hf_dataset(
            path = "hellaswag",
            split = "validation",
            sample_fields = record_to_sample,
            ),
        solver = multiple_choice(),
        scorer = choice(),
    )