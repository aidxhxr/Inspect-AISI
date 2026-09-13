from inspect_ai.dataset import json_dataset

dataset = json_dataset("popularity.jsonl", record_to_sample)
dataset = dataset.filter(
    lambda sample: sample.metadatta["category"] == "advanced"
)