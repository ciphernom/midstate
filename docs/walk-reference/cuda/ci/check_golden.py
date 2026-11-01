#!/usr/bin/env python3
"""Checks that the bench's known answers are midstate's.

    check_golden.py src/core/walk.rs docs/walk-reference/cuda/walk_core.h

walk.rs pins six (mining hash, nonce, walk hash) answers: three for mainnet
and three for the fast-mining test build. walk_core.h carries a copy that the
bench checks the GPU against. Changing the walk changes those answers, so this
fails until the bench is brought up to date as well.
"""
import re
import sys

HEX = r'"([0-9a-f]{64})"'


def answers(path, nonce_pattern):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    triple = HEX + r"\s*,\s*(" + nonce_pattern + r")\s*,\s*" + HEX
    found = set()
    for mining_hash, nonce, want in re.findall(triple, text):
        nonce = nonce.replace("_", "")
        value = 2**64 - 1 if nonce == "u64::MAX" else int(re.sub(r"(u64|ull|ULL)$", "", nonce))
        found.add((mining_hash, value, want))
    return found


def main():
    if len(sys.argv) != 3:
        print(__doc__.strip())
        return 2
    rust_path, bench_path = sys.argv[1:]
    rust = answers(rust_path, r"u64::MAX|[0-9_]+(?:u64)?")
    bench = answers(bench_path, r"[0-9]+(?:ull|ULL)?")
    if len(rust) != 6:
        print(f"expected 6 known answers in {rust_path}, found {len(rust)}")
        return 1
    if rust != bench:
        for t in sorted(rust - bench):
            print(f"in {rust_path} but not {bench_path}: {t}")
        for t in sorted(bench - rust):
            print(f"in {bench_path} but not {rust_path}: {t}")
        return 1
    print(f"{bench_path}'s {len(bench)} known answers match {rust_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
