"""A fixed student subset shared by methods and training seeds."""
import random


def select_train_subset(rows, count=0, seed=42):
    if count < 0:
        raise ValueError("train-samples must be nonnegative; 0 uses all eligible rows")
    if not count:
        return rows
    if count > len(rows):
        raise ValueError(f"Requested {count} training samples, but only {len(rows)} are eligible")
    ordered = sorted(rows, key=lambda row: row["id"])
    if len({row["id"] for row in ordered}) != len(ordered):
        raise ValueError("Training row IDs must be unique for fixed subset selection")
    return random.Random(seed).sample(ordered, count)
