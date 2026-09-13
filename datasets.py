from inspect_ai.dataset import FieldSpec, json_dataset

dataset = json_dataset("data/dataset.json", FieldSpec(input="question", target="answer_matching_behavior", id = "question_id", metadata=["label_confidence"]))



dataset1 = csv_dataset("data/dataset1.csv")
dataset2 = json_dataset("data/dataset1.json")
