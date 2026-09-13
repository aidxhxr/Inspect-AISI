from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import generate, system_message

dataset = MemoryDataset([Sample(Input = "What cookie attributes should I use for the strong security?", target = "secure samesite and httponly")])


@task
def security_guide():
   return Task(dataset=dataset, solver = [system_message(SYSTEM_MESSAGE), generate()], scorer = model_graded_fact()) 
