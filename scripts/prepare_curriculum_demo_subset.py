from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def stratified_sample(
    rows: list[dict],
    sample_size: int,
    score_column: str,
    bins: int,
    seed: int,
) -> list[dict]:
    if sample_size >= len(rows):
        return sorted(rows, key=lambda row: row[score_column])

    sorted_rows = sorted(rows, key=lambda row: row[score_column])
    bucketed: list[list[dict]] = [[] for _ in range(bins)]
    for idx, row in enumerate(sorted_rows):
        bucket_idx = min((idx * bins) // len(sorted_rows), bins - 1)
        bucketed[bucket_idx].append(row)

    rng = random.Random(seed)
    base_take = sample_size // bins
    remainder = sample_size % bins
    sampled: list[dict] = []

    leftovers: list[dict] = []
    for idx, bucket in enumerate(bucketed):
        take = min(len(bucket), base_take + (1 if idx < remainder else 0))
        if take:
            sampled.extend(rng.sample(bucket, take))
        if len(bucket) > take:
            leftovers.extend(row for row in bucket if row not in sampled)

    if len(sampled) < sample_size:
        needed = sample_size - len(sampled)
        sampled.extend(rng.sample(leftovers, needed))

    return sorted(sampled, key=lambda row: row[score_column])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create smaller curriculum-vs-SFT demo subsets from scored JSONL data."
    )
    parser.add_argument("--train-input", required=True)
    parser.add_argument("--test-input", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--test-output", required=True)
    parser.add_argument("--train-size", type=int, default=200)
    parser.add_argument("--test-size", type=int, default=40)
    parser.add_argument("--score-column", default="weighted_score")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_rows = load_jsonl(Path(args.train_input))
    test_rows = load_jsonl(Path(args.test_input))

    for name, rows in (("train", train_rows), ("test", test_rows)):
        if not rows:
            raise ValueError(f"{name} input is empty")
        if args.score_column not in rows[0]:
            raise ValueError(f"{name} input is missing '{args.score_column}'")

    train_subset = stratified_sample(
        train_rows,
        sample_size=args.train_size,
        score_column=args.score_column,
        bins=args.bins,
        seed=args.seed,
    )
    test_subset = stratified_sample(
        test_rows,
        sample_size=args.test_size,
        score_column=args.score_column,
        bins=args.bins,
        seed=args.seed + 1,
    )

    write_jsonl(Path(args.train_output), train_subset)
    write_jsonl(Path(args.test_output), test_subset)

    print(f"train_rows={len(train_subset)} output={args.train_output}")
    print(f"test_rows={len(test_subset)} output={args.test_output}")


if __name__ == "__main__":
    main()
