from inspect_ai import Task, task
from inspect_ai.dataset import example_dataset
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import chain_of_thought, generate, system_message

SYSTEM_MESSAGE = """
You will read a short story about people entering and leaving rooms
while objects are moved, followed by a single question.

Answer from the point of view the question asks about:
- "Where is X?" or "Where was X at the beginning?" asks about the
  object's real location at that moment.
- "Where will <person> look for X?" or "Where does <person> think X is?"
  asks about that person's belief. A person only knows about moves that
  happened while they were in the same room. If they left before an
  object was moved, they still believe it is where they last saw it.

Track, step by step, who is in the room during each move. Then finish
with one line of the form:

ANSWER: <location>

Give only the location name on that line.
"""


@task
def theory_of_mind(cot: bool = False):
    return Task(
        dataset=example_dataset("theory_of_mind"),
        solver=[
            system_message(SYSTEM_MESSAGE),
            *([chain_of_thought()] if cot else []),
            generate(),
        ],
        scorer=model_graded_fact(),
    )
