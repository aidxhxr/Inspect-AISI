"""Load the bundled capture-the-flag challenges as an Inspect dataset."""

import json
from pathlib import Path
from typing import Any

from inspect_ai.dataset import MemoryDataset, Sample

CHALLENGES_ROOT = Path(__file__).resolve().parent / "challenges"
REQUIRED_KEYS = ("id", "name", "difficulty", "category", "prompt", "flag", "tags")


def read_challenge(directory: Path) -> dict[str, Any]:
    """Read and validate the challenge.json for one challenge directory."""
    spec_path = directory / "challenge.json"
    spec = json.loads(spec_path.read_text())
    missing = [key for key in REQUIRED_KEYS if key not in spec]
    if missing:
        raise ValueError(f"{spec_path}: missing keys {missing}")
    if not isinstance(spec["difficulty"], int) or not 1 <= spec["difficulty"] <= 5:
        raise ValueError(f"{spec_path}: difficulty must be an int in 1..5")
    if not (directory / "compose.yaml").is_file():
        raise ValueError(f"{directory}: compose.yaml is missing")
    if not (directory / "Dockerfile").is_file():
        raise ValueError(f"{directory}: Dockerfile is missing")
    if spec["flag"] in spec["prompt"]:
        raise ValueError(f"{spec_path}: the flag must not appear in the prompt")
    return spec


def challenge_dirs(root: Path) -> list[Path]:
    """Return challenge directories under root, sorted by difficulty then id."""
    dirs = [d for d in root.iterdir() if d.is_dir() and (d / "challenge.json").is_file()]
    specs = {d: read_challenge(d) for d in dirs}
    return sorted(dirs, key=lambda d: (specs[d]["difficulty"], specs[d]["id"]))


def load_challenges(root: Path | None = None, max_difficulty: int | None = None) -> MemoryDataset:
    """Build a MemoryDataset of CTF samples, one per challenge directory.

    Each sample's input is the challenge prompt, its target is the flag, and its
    sandbox is the docker compose file in the challenge directory.
    """
    root = CHALLENGES_ROOT if root is None else root
    samples: list[Sample] = []
    for directory in challenge_dirs(root):
        spec = read_challenge(directory)
        if max_difficulty is not None and spec["difficulty"] > max_difficulty:
            continue
        compose_path = directory / "compose.yaml"
        samples.append(
            Sample(
                input=spec["prompt"],
                target=spec["flag"],
                id=spec["id"],
                metadata={
                    "name": spec["name"],
                    "difficulty": spec["difficulty"],
                    "category": spec["category"],
                    "tags": list(spec["tags"]),
                },
                sandbox=("docker", str(compose_path)),
            )
        )
    return MemoryDataset(samples=samples, name="ctf", location=str(root))
