"""Resolve downloaded encoders without falling back to the Hugging Face Hub."""

import os
from pathlib import Path


MODEL_REPOS = {
    "bert": "google-bert/bert-base-uncased",
    "tinybert": "huawei-noah/TinyBERT_General_6L_768D",
    "bgem3": "BAAI/bge-m3",
    "qwen": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
}


def downloaded_model_path(key):
    """Use project models/ by default, or MRL_MODELS_DIR when set."""
    root = Path(os.environ.get("MRL_MODELS_DIR") or Path(__file__).resolve().parent.parent / "models")
    return str((root.expanduser() / MODEL_REPOS[key].split("/")[-1]).resolve())


MODEL_CONFIGS = {key: {"model_name": downloaded_model_path(key)} for key in MODEL_REPOS}


def resolve_local_model(source):
    """Accept an existing directory, a supported alias, or a downloaded repo ID."""
    path = Path(source).expanduser()
    if not path.is_dir():
        key = source if source in MODEL_REPOS else next(
            (key for key, repo in MODEL_REPOS.items() if source == repo), None)
        if key is not None:
            path = Path(downloaded_model_path(key))
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(
            f"Local model directory not found: {path}. Download the model into MRL-text/models/ "
            "using download.txt, set MRL_MODELS_DIR, or pass --model_name /path/to/model. "
            "Models are loaded offline; no Hub download will be attempted.")
    return str(path)
