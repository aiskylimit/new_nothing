"""Prepare existing raw MTEB files and restore prepared tasks without Hub access.

On the offline server: bash prepare_data.sh --eval_only --eval_tasks STSBenchmark SICK-R
The saved Arrow/JSON bundles include classification training splits and retrieval
corpora, queries and qrels. They can be moved independently of Hugging Face caches.
"""

import argparse
from contextlib import contextmanager
from functools import wraps
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import re
import shutil
import socket
import tempfile
from unittest.mock import patch

import numpy as np
import torch

DEFAULT_EVAL_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "mteb"
DEFAULT_RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
FORMAT_VERSION = 1


@contextmanager
def offline_mode():
    """Also block direct HTTP/socket loaders that ignore Hugging Face flags."""
    import datasets.config
    import huggingface_hub.constants

    def blocked(*args, **kwargs):
        raise RuntimeError("Offline processing blocked network access. Download raw data on an online machine "
                           "with bash download_mteb.sh, copy data/raw, then run "
                           "bash prepare_data.sh --eval_only; "
                           "use a local or already cached model/tokenizer.")

    with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                                 "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1",
                                 "MTEB_CACHE": os.environ.get("MTEB_CACHE", str(DEFAULT_EVAL_DATA_DIR.parent / ".cache" / "mteb"))}), \
         patch.object(datasets.config, "HF_DATASETS_OFFLINE", True), \
         patch.object(huggingface_hub.constants, "HF_HUB_OFFLINE", True), \
         patch.object(huggingface_hub.constants, "HF_HUB_DISABLE_TELEMETRY", True), \
         patch.object(socket, "getaddrinfo", blocked), \
         patch.object(socket.socket, "connect", blocked), \
         patch.object(socket.socket, "connect_ex", blocked):
        yield


def offline_evaluation(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with offline_mode():
            return function(*args, **kwargs)
    return wrapped


def _encode_state(value):
    """Preserve task-local RNGs and attributes changed by dataset_transform."""
    if isinstance(value, random.Random):
        return {"__type__": "random", "state": _encode_state(value.getstate())}
    if isinstance(value, np.random.Generator):
        return {"__type__": "generator", "state": _encode_state(value.bit_generator.state)}
    if isinstance(value, np.ndarray):
        return {"__type__": "array", "dtype": str(value.dtype), "data": value.tolist()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return {"__type__": "tuple", "items": [_encode_state(item) for item in value]}
    if isinstance(value, list):
        return [_encode_state(item) for item in value]
    if isinstance(value, dict):
        return {key: _encode_state(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported prepared MTEB task attribute: {type(value).__name__}")


def _decode_state(value):
    if isinstance(value, list):
        return [_decode_state(item) for item in value]
    if not isinstance(value, dict):
        return value
    kind = value.get("__type__")
    if kind == "tuple":
        return tuple(_decode_state(item) for item in value["items"])
    if kind == "array":
        return np.asarray(value["data"], dtype=value["dtype"])
    if kind == "random":
        result = random.Random()
        result.setstate(_decode_state(value["state"]))
        return result
    if kind == "generator":
        state = _decode_state(value["state"])
        result = np.random.Generator(getattr(np.random, state["bit_generator"])())
        result.bit_generator.state = state
        return result
    return {key: _decode_state(item) for key, item in value.items()}


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def _save_data(value, directory, counter):
    from datasets import Dataset, DatasetDict

    number = next(counter)
    if isinstance(value, (Dataset, DatasetDict)):
        name = f"arrow_{number}"
        value.save_to_disk(str(directory / name))
        return {"kind": "arrow", "path": name}
    # Keep large qrel dictionaries in a separate compact JSON file.
    try:
        serialized = json.dumps(value, ensure_ascii=False, allow_nan=False)
    except TypeError:
        if not isinstance(value, dict):
            raise TypeError(f"Unsupported MTEB dataset container: {type(value).__name__}") from None
        return {"kind": "mapping", "items": {
            key: _save_data(item, directory, counter) for key, item in value.items()}}
    name = f"data_{number}.json"
    (directory / name).write_text(serialized + "\n")
    return {"kind": "json", "path": name}


def _load_data(node, directory):
    from datasets import load_from_disk

    if node["kind"] == "mapping":
        return {key: _load_data(item, directory) for key, item in node["items"].items()}
    path = (directory / node["path"]).resolve()
    if not path.is_relative_to(directory.resolve()) or not path.exists():
        raise FileNotFoundError(f"Missing/invalid local MTEB data: {path}. "
                                "Run bash prepare_data.sh --eval_only --force on the copied raw files.")
    if node["kind"] == "arrow":
        return load_from_disk(str(path))
    if node["kind"] == "json":
        return json.loads(path.read_text())
    raise ValueError(f"Unknown local MTEB data kind: {node['kind']}")


def _task_directory(data_dir, name):
    if not re.fullmatch(r"[\w.-]+", name) or name in {".", ".."}:
        raise ValueError(f"Invalid MTEB task name: {name!r}")
    return Path(data_dir).expanduser().resolve() / name


def read_manifest(data_dir, name, splits, languages, seed):
    path = _task_directory(data_dir, name) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Local MTEB data missing for {name}: {path}. "
                                f"Download raw files for {name} with bash download_mteb.sh on an online "
                                "machine, copy data/raw, then run bash prepare_data.sh --eval_only "
                                f"--eval_tasks {name} --eval_data_dir {Path(data_dir).expanduser().resolve()} "
                                "with the same --eval_splits, --eval_languages and --seed.")
    manifest = json.loads(path.read_text())
    expected = {"format_version": FORMAT_VERSION, "task_name": name,
                "eval_splits": sorted(set(splits)), "eval_languages": sorted(set(languages)), "seed": seed}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Local MTEB data for {name} has incompatible {key}: "
                             f"saved={manifest.get(key)!r}, requested={value!r}. "
                             f"Run bash prepare_data.sh --eval_only --eval_tasks {name} --force with matching options.")
    if manifest["mteb_version"] != version("mteb"):
        raise ValueError(f"Local {name} was prepared with MTEB {manifest['mteb_version']}; "
                         f"installed version is {version('mteb')}. Use matching MTEB or prepare again --force.")
    for relative, size in manifest["files"].items():
        data_path = (path.parent / relative).resolve()
        if (not data_path.is_relative_to(path.parent.resolve()) or not data_path.is_file()
                or data_path.stat().st_size != size):
            raise FileNotFoundError(f"Local MTEB data incomplete for {name}: {data_path}. "
                                    f"Run bash prepare_data.sh --eval_only --eval_tasks {name} --force.")
    return manifest


def require_local_tasks(data_dir, names, splits, languages, seed):
    for name in dict.fromkeys(names):
        read_manifest(data_dir, name, splits, languages, seed)


def save_task(task, directory, *, splits, languages, seed):
    """Save an already prepared task, including its post-transform state."""
    from itertools import count

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if not task.data_loaded or task.dataset is None:
        raise ValueError("Load the MTEB task before saving it.")
    state = {key: value for key, value in vars(task).items()
             if key not in {"dataset", "metadata", "data_loaded"}}
    manifest = {"format_version": FORMAT_VERSION, "mteb_version": version("mteb"),
                "task_name": task.metadata.name, "eval_splits": sorted(set(splits)),
                "eval_languages": sorted(set(languages)), "seed": seed,
                "metadata": task.metadata.model_dump(mode="json"), "task_state": _encode_state(state),
                "dataset": _save_data(task.dataset, directory, count()),
                "global_rng": _encode_state({"python": random.getstate(), "numpy": np.random.get_state(),
                                              "torch": torch.get_rng_state().tolist()})}
    manifest["files"] = {str(path.relative_to(directory)): path.stat().st_size
                         for path in directory.rglob("*") if path.is_file() and path.name != "manifest.json"}
    # A manifest marks completed preparation and is written after every data file.
    _write_json(directory / "manifest.json", manifest)
    return manifest


def load_task(data_dir, name, *, splits, languages, seed):
    """Restore prepared data without calling task.load_data/dataset_transform."""
    import mteb

    manifest = read_manifest(data_dir, name, splits, languages, seed)
    task = mteb.get_task(name, eval_splits=list(splits), languages=list(languages))
    task.metadata = type(task.metadata).model_validate(manifest["metadata"])
    task.__dict__.update(_decode_state(manifest["task_state"]))
    task.dataset = _load_data(manifest["dataset"], _task_directory(data_dir, name))
    task.data_loaded = True
    state = _decode_state(manifest["global_rng"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(torch.tensor(state["torch"], dtype=torch.uint8))
    return task


@offline_evaluation
def prepare_tasks(data_dir=DEFAULT_EVAL_DATA_DIR, *, names, raw_data_dir=None,
                  splits=("test",), languages=("eng",), seed=42, force=False):
    if isinstance(names, str) or names is None:
        raise ValueError("Explicitly select task names with --eval_tasks.")
    names = list(dict.fromkeys(names))
    if not names:
        raise ValueError("Explicitly select at least one task with --eval_tasks.")
    for name in names:
        _task_directory(data_dir, name)
    os.environ.setdefault("MTEB_CACHE", str(Path(data_dir).expanduser().resolve().parent / ".cache" / "mteb"))
    import mteb
    from utils.raw_data import raw_file_plan, prepare_task, require_raw_files

    raw_data_dir = raw_data_dir or DEFAULT_RAW_DATA_DIR
    root = Path(data_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        destination = _task_directory(root, name)
        if destination.exists() and not force:
            read_manifest(root, name, splits, languages, seed)
            print(f"{name}: already prepared at {destination}")
            continue
        temporary = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=root))
        task = None
        try:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            task = mteb.get_task(name, eval_splits=list(splits), languages=list(languages))
            if task.metadata.name != name:
                raise ValueError(f"MTEB resolved {name} to {task.metadata.name}; request the exact task name.")
            plan = raw_file_plan(task, splits)
            require_raw_files(raw_data_dir, plan)
            print(f"{name}: preparing local raw data...")
            prepare_task(task, raw_data_dir, plan, temporary / "cache")
            prepared = temporary / "prepared"
            manifest = save_task(task, prepared, splits=splits, languages=languages, seed=seed)
            manifest.update(preparation="offline_raw_files", source_files=plan)
            _write_json(prepared / "manifest.json", manifest)
            if destination.exists():
                shutil.rmtree(destination)
            prepared.replace(destination)
            print(f"{name}: ready at {destination}")
        finally:
            if task is not None:
                task.unload_data()
            if temporary.exists():
                shutil.rmtree(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_data_dir", default=str(DEFAULT_EVAL_DATA_DIR))
    parser.add_argument("--raw_data_dir", default=str(DEFAULT_RAW_DATA_DIR))
    parser.add_argument("--eval_tasks", nargs="+", required=True,
                        help="Prepare only these selected evaluation tasks from local raw files")
    parser.add_argument("--eval_splits", nargs="+", default=["test"])
    parser.add_argument("--eval_languages", nargs="+", default=["eng"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Replace prepared tasks, e.g. after changing options")
    args = parser.parse_args(argv)
    prepare_tasks(args.eval_data_dir, raw_data_dir=args.raw_data_dir, names=args.eval_tasks, splits=args.eval_splits,
                  languages=args.eval_languages, seed=args.seed, force=args.force)


if __name__ == "__main__":
    main()
