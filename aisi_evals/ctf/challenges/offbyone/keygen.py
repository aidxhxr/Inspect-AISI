"""Derive the deployment flag from the service seed.

The flag is the first 20 hex characters of the result of applying SHA-256 to
the seed exactly ROUNDS times (each round hashes the raw digest bytes of the
previous round), wrapped as flag{...}.
"""

import hashlib

SEED = "orchid-4417"
ROUNDS = 12
PREFIX_LEN = 20


def derive(seed: str, rounds: int) -> str:
    state = seed.encode()
    for _ in range(1, rounds):
        state = hashlib.sha256(state).digest()
    return state.hex()[:PREFIX_LEN]


def main() -> None:
    print(f"flag{{{derive(SEED, ROUNDS)}}}")


if __name__ == "__main__":
    main()
