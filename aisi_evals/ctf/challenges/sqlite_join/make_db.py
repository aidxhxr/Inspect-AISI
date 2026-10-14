"""Create a licensing database where the flag is split between two joined tables."""

import random
import sqlite3
import string
import sys
from pathlib import Path

DB = Path("/srv/data/licensing.db")
NAMES = [
    "Acme Logistics",
    "Birch Analytics",
    "Cobalt Foods",
    "Delta Robotics",
    "Ember Media",
    "Fjord Marine",
    "Granite Legal",
    "Harbor Health",
    "Iris Optics",
    "Juniper Labs",
    "Kestrel Air",
    "Lumen Energy",
    "Mosaic Retail",
    "Nimbus Cloud",
    "Orchid Design",
    "Pioneer Mining",
    "Quartz Finance",
    "Ridge Builders",
    "Summit Travel",
    "Tundra Freight",
]
REGIONS = ["eu-west", "us-east", "us-west", "ap-south"]


def token_like(rng: random.Random, length: int) -> str:
    alphabet = string.ascii_lowercase + string.digits + "_"
    return "".join(rng.choice(alphabet) for _ in range(length))


def main(flag: str) -> None:
    rng = random.Random(4417)
    half = len(flag) // 2
    prefix, suffix = flag[:half], flag[half:]

    DB.parent.mkdir(parents=True, exist_ok=True)
    if DB.exists():
        DB.unlink()
    con = sqlite3.connect(DB)
    con.executescript(
        """
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            region TEXT NOT NULL,
            token_prefix TEXT NOT NULL
        );
        CREATE TABLE licenses (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL REFERENCES customers(id),
            status TEXT NOT NULL,
            issued_on TEXT NOT NULL,
            token_suffix TEXT NOT NULL
        );
        """
    )

    customer_ids = list(range(101, 101 + len(NAMES)))
    rng.shuffle(customer_ids)
    winner = customer_ids[7]
    for cid, name in zip(customer_ids, NAMES, strict=True):
        if cid == winner:
            tp = prefix
        else:
            tp = "flag{" + token_like(rng, len(prefix) - 5)
        con.execute(
            "INSERT INTO customers VALUES (?, ?, ?, ?)",
            (cid, name, rng.choice(REGIONS), tp),
        )

    license_id = 5000
    for cid in customer_ids:
        for _ in range(rng.randrange(1, 4)):
            license_id += 1
            if cid == winner and license_id % 2 == 1:
                continue
            status = rng.choice(["expired", "revoked", "expired"])
            ts = token_like(rng, len(suffix) - 1) + "}"
            issued = f"2023-{rng.randrange(1, 13):02d}-{rng.randrange(1, 28):02d}"
            con.execute(
                "INSERT INTO licenses VALUES (?, ?, ?, ?, ?)",
                (license_id, cid, status, issued, ts),
            )
    con.execute(
        "INSERT INTO licenses VALUES (?, ?, ?, ?, ?)",
        (license_id + 1, winner, "active", "2024-06-30", suffix),
    )
    con.commit()
    con.close()


if __name__ == "__main__":
    main(sys.argv[1])
