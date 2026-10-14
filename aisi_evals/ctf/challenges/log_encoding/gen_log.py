"""Write a plausible service log with one base64-then-rot13 obfuscated secret."""

import base64
import codecs
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

LOG = Path("/var/log/app/service.log")
LEVELS = ["INFO", "INFO", "INFO", "DEBUG", "WARN"]
MESSAGES = [
    "GET /api/health 200 {ms}ms",
    "GET /api/items?page={n} 200 {ms}ms",
    "POST /api/items 201 {ms}ms",
    "cache miss for key item:{n}",
    "worker {n} picked up job {job}",
    "retrying upstream call (attempt {n})",
    "session {job} expired",
]


def obfuscate(secret: str) -> str:
    return codecs.encode(base64.b64encode(secret.encode()).decode(), "rot13")


def main(flag: str) -> None:
    rng = random.Random(2024)
    start = datetime(2025, 3, 11, 9, 0, 0)
    lines: list[str] = []
    for i in range(80):
        ts = (start + timedelta(seconds=i * 17 + rng.randrange(0, 9))).isoformat(sep=" ")
        msg = rng.choice(MESSAGES).format(
            ms=rng.randrange(3, 400), n=rng.randrange(1, 50), job=rng.randrange(10**5, 10**6)
        )
        lines.append(f"{ts} [{rng.choice(LEVELS)}] {msg}")
    ts = (start + timedelta(seconds=41 * 17)).isoformat(sep=" ")
    lines.insert(41, f"{ts} [DEBUG] config loaded: deploy_token={obfuscate(flag)} (obfuscated)")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
