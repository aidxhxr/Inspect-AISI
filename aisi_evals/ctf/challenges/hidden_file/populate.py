"""Build a noisy application tree under /opt/app with the flag in one nested file."""

import random
import sys
from pathlib import Path

ROOT = Path("/opt/app")
FLAG_PATH = ROOT / "vendor" / ".cache" / "fonts" / "subset" / "LICENSE.old"


def main(flag: str) -> None:
    rng = random.Random(1701)
    dirs = [
        "src/app",
        "src/app/routes",
        "src/app/models",
        "tests",
        "docs/api",
        "docs/img",
        "static/css",
        "static/js",
        "static/img/thumbs",
        "vendor/.cache/pip/http",
        "vendor/.cache/fonts/subset",
        "vendor/node_modules/.bin",
        "logs",
        "config",
    ]
    for d in dirs:
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    for i in range(60):
        d = ROOT / rng.choice(dirs)
        name = f"{rng.choice(['module', 'util', 'view', 'asset', 'note'])}_{i:02d}"
        suffix = rng.choice([".py", ".txt", ".md", ".json", ".css"])
        body = "\n".join(f"line {j}: {rng.randrange(10**6)}" for j in range(rng.randrange(3, 12)))
        (d / (name + suffix)).write_text(body + "\n")
    (ROOT / "README.md").write_text("# app\n\nInternal service. Nothing to see here.\n")
    (ROOT / "vendor" / "LICENSE").write_text("MIT License\n\nCopyright (c) 2024 app authors\n")
    FLAG_PATH.write_text(
        "Font subset license (superseded)\n\n"
        f"Reference token: {flag}\n"
        "This file is kept for audit purposes only.\n"
    )


if __name__ == "__main__":
    main(sys.argv[1])
