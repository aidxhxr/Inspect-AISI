from inspect_ai.dataset import FieldSpec, hf_dataset

dataset = hf_dataset("openai_humaneval", split = "test", sample_fields = FieldSpec(id="task_id", input="prompt", target="canonical_solution", metadata=["test", "entry_point"]))
