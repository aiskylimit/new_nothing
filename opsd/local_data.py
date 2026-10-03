from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk


def load_local_split(dataset_path: str, split: str = "train") -> Dataset:
    path = Path(dataset_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Local dataset does not exist: {path}")

    dataset = load_from_disk(str(path))
    if isinstance(dataset, DatasetDict):
        if split not in dataset:
            raise KeyError(f"Split {split!r} is not available in {path}: {list(dataset)}")
        dataset = dataset[split]
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected a Dataset at {path}, got {type(dataset).__name__}")
    return dataset
