"""Expand multiple-choice samples into several permutations of their options."""

from __future__ import annotations

import random

from inspect_ai.dataset import Dataset, MemoryDataset, Sample


def _letter_index(target: str | list[str]) -> int:
    text = target if isinstance(target, str) else target[0]
    text = text.strip().upper()
    if len(text) == 1 and text.isalpha():
        return ord(text) - ord("A")
    return int(text)


def expand_permutations(dataset: Dataset, n_permutations: int = 4, seed: int = 0) -> MemoryDataset:
    """Return `n_permutations` copies of each sample with shuffled options.

    The first copy keeps the original order. Each copy records `original_id`,
    `permutation` and `target_position` in metadata so clustered standard
    errors and consistency metrics can group by question.
    """
    rng = random.Random(seed)
    out: list[Sample] = []
    for sample in dataset:
        if not sample.choices:
            raise ValueError(f"sample {sample.id!r} has no choices")
        n = len(sample.choices)
        correct = _letter_index(sample.target)
        for p in range(n_permutations):
            order = list(range(n))
            if p > 0:
                rng.shuffle(order)
            choices = [sample.choices[i] for i in order]
            position = order.index(correct)
            meta = dict(sample.metadata or {})
            meta.update(
                {
                    "original_id": sample.id,
                    "permutation": p,
                    "target_position": position,
                    "choice_order": order,
                }
            )
            out.append(
                Sample(
                    input=sample.input,
                    choices=choices,
                    target=chr(ord("A") + position),
                    id=f"{sample.id}-p{p}",
                    metadata=meta,
                )
            )
    return MemoryDataset(out, name=f"{dataset.name or 'dataset'}-permuted")
