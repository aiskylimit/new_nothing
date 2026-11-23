import argparse
from copy import deepcopy
from importlib.metadata import version
import json
import math
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from utils.model_paths import resolve_local_model
from utils.mteb_tasks import DEFAULT_EVAL_TASKS
from utils.mteb_results import format_score, write_result_summaries
from utils.mteb_data import DEFAULT_EVAL_DATA_DIR, load_task, offline_evaluation, require_local_tasks


def add_eval_args(parser, *, training=False):
    if training:
        parser.add_argument("--run_eval", type=int, choices=[0, 1], default=1,
                            help="Run selected MTEB tasks after each epoch; 0 disables evaluation")
    parser.add_argument("--eval_tasks", nargs="+", default=list(DEFAULT_EVAL_TASKS),
                        help="Override the selected task preset from utils/mteb_tasks.py")
    parser.add_argument("--eval_data_dir", default=str(DEFAULT_EVAL_DATA_DIR),
                        help="Prepared local MTEB data; run bash prepare_data.sh --eval_only on copied raw files")
    parser.add_argument("--eval_splits", nargs="+", default=["test"])
    parser.add_argument("--eval_languages", nargs="+", default=["eng"])
    parser.add_argument(*(["--eval_batch_size"] if training else ["--eval_batch_size", "--batch_size"]),
                        dest="eval_batch_size", type=int, default=64)
    parser.add_argument("--eval_max_length", type=int, default=256)


def validate_eval_args(args):
    if args.eval_batch_size < 1 or args.eval_max_length < 1:
        raise ValueError("eval_batch_size and eval_max_length must be positive.")
    if getattr(args, "run_eval", True) and not args.eval_tasks:
        raise ValueError("Specify --eval_tasks to select MTEB benchmarks, or use --run_eval 0 for training only.")
    if getattr(args, "run_eval", True):
        require_local_tasks(args.eval_data_dir, args.eval_tasks, args.eval_splits, args.eval_languages, args.seed)


def evaluation_dims(full_dim, dims=None):
    dims = list(dims) if dims is not None else [d for d in (16, 32, 64, 128, 256, 512) if d < full_dim] + [full_dim]
    if not dims or any(isinstance(d, bool) or not isinstance(d, int) or not 1 <= d <= full_dim for d in dims):
        raise ValueError(f"Evaluation dimensions must be integers in [1, {full_dim}].")
    return sorted(set(dims))


class MatryoshkaEncoder:
    def __init__(self, model, tokenizer, dim, *, pooling="cls", max_length=256, batch_size=64):
        self.model, self.tokenizer, self.dim = model, tokenizer, dim
        self.pooling, self.max_length, self.batch_size = pooling, max_length, batch_size
        self._embedding_cache = {}
        evaluation_dims(model.config.hidden_size, [dim])
        if pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be cls or mean.")
        self.mteb_model_meta = None  # Assigned by evaluate_model; keeps the adapter importable without MTEB.

    def set_dim(self, dim):
        evaluation_dims(self.model.config.hidden_size, [dim])
        self.dim = dim

    def clear_cache(self):
        self._embedding_cache.clear()

    @torch.inference_mode()
    def encode(self, inputs, *, task_metadata=None, hf_split=None, hf_subset=None, prompt_type=None, **kwargs):
        chunks = []
        device = next(self.model.parameters()).device
        # Keep the unprompted training protocol for both queries and passages.
        for batch in inputs:
            texts = list(batch["text"])
            if not texts:
                continue
            missing = list(dict.fromkeys(text for text in texts if text not in self._embedding_cache))
            for start in range(0, len(missing), self.batch_size):
                unseen = missing[start:start + self.batch_size]
                features = self.tokenizer(unseen, padding=True,
                                          truncation=True, max_length=self.max_length, return_tensors="pt")
                features = {key: value.to(device) for key, value in features.items()}
                hidden = self.model(**features, return_dict=True).last_hidden_state.float()
                if self.pooling == "cls":
                    embeddings = hidden[:, 0]
                else:
                    mask = features["attention_mask"].unsqueeze(-1).float()
                    embeddings = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                # Cache unnormalized full coordinates; every dimension uses its own prefix norm.
                for text, embedding in zip(unseen, embeddings.cpu().numpy()):
                    self._embedding_cache[text] = embedding.copy()
            # Restore duplicates and the exact input order expected by the task evaluator.
            prefix = np.stack([self._embedding_cache[text][:self.dim] for text in texts])
            chunks.append(F.normalize(torch.from_numpy(prefix), dim=-1).numpy())
        return np.concatenate(chunks) if chunks else np.empty((0, self.dim), dtype=np.float32)

    @staticmethod
    def similarity(embeddings1, embeddings2):
        first = F.normalize(torch.as_tensor(embeddings1).float(), dim=-1)
        second = F.normalize(torch.as_tensor(embeddings2).float(), dim=-1)
        return first @ second.T

    @staticmethod
    def similarity_pairwise(embeddings1, embeddings2):
        first = F.normalize(torch.as_tensor(embeddings1).float(), dim=-1)
        second = F.normalize(torch.as_tensor(embeddings2).float(), dim=-1)
        return (first * second).sum(-1)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def write_eval_progress(directory, results):
    """Persist completed scores before starting the next dimension or task."""
    dims, epoch = results["matryoshka_dims"], results["epoch"]
    rows = ["| Task | Metric | " + " | ".join(str(dim) for dim in dims) + " |",
            "| --- | --- | " + " | ".join("---:" for _ in dims) + " |"]
    for name in results["tasks"]:
        rows.append("| " + name + " | " + results["metrics"][name] + " | " + " | ".join(
            format_score(results["scores"].get(f"dim_{dim}", {}).get(name)) for dim in dims) + " |")
    label = f"epoch {epoch}" if epoch is not None else "standalone evaluation"
    progress = (f"Completed tasks: {len(results['completed_tasks'])}/{len(results['requested_tasks'])}; "
                f"status: {results['status']}. N/A means missing or non-finite.")
    summary = f"MTEB main_score by embedding dimension ({label})\n{progress}\n\n" + "\n".join(rows) + "\n"
    suffixes = [""] + ([f"_epoch_{epoch}"] if epoch is not None else [])
    for suffix in suffixes:
        write_json(directory / f"result{suffix}.json", results)
        path = directory / f"summary{suffix}.txt"
        temporary = path.with_suffix(".txt.tmp")
        temporary.write_text(summary)
        temporary.replace(path)
    write_result_summaries(directory)
    return summary


@offline_evaluation
def evaluate_model(model, tokenizer, dims, output_dir, *, eval_tasks=None, epoch=None,
                   eval_splits=None, eval_languages=None, batch_size=64, max_length=256,
                   pooling=None, seed=42, eval_data_dir=None):
    eval_data_dir = eval_data_dir or DEFAULT_EVAL_DATA_DIR
    if eval_tasks is None:
        eval_tasks = DEFAULT_EVAL_TASKS
    if not eval_tasks:
        raise ValueError("Provide explicit MTEB task names in --eval_tasks.")
    if batch_size < 1 or max_length < 1:
        raise ValueError("Evaluation batch_size and max_length must be positive.")
    dims = evaluation_dims(model.config.hidden_size, dims)
    pooling = pooling or getattr(model, "embedding_pooling", "cls")
    names = list(dict.fromkeys(eval_tasks))
    splits, languages = list(eval_splits or ["test"]), list(eval_languages or ["eng"])
    require_local_tasks(eval_data_dir, names, splits, languages, seed)
    directory = Path(output_dir).expanduser().resolve()
    raw_dir = directory / ("mteb" if epoch is None else f"mteb_epoch_{epoch}")
    results = {"backend": "mteb", "mteb_version": version("mteb"), "epoch": epoch,
               "model_name": getattr(model.config, "_name_or_path", ""), "output_dir": str(directory),
               "tasks": [], "requested_tasks": names, "completed_tasks": [], "status": "in_progress",
               "eval_splits": splits, "offline": True,
               "eval_data_dir": str(Path(eval_data_dir).expanduser().resolve()),
               "eval_languages": languages, "matryoshka_dims": dims, "pooling": pooling,
               "max_length": max_length, "seed": seed, "metrics": {}, "scores": {}, "raw_results": {},
               "dataset_load_seconds": {}}
    was_training = model.training
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        model.eval()
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            import mteb
            from mteb.models.model_meta import ModelMeta

            raw_by_dim = {}
            for name in names:
                adapter = MatryoshkaEncoder(model, tokenizer, dims[0], pooling=pooling,
                                            max_length=max_length, batch_size=batch_size)
                prepared_task = task = None
                try:
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    start_load = perf_counter()
                    prepared_task = load_task(eval_data_dir, name, splits=splits, languages=languages, seed=seed)
                    task_name = prepared_task.metadata.name
                    results["dataset_load_seconds"][task_name] = perf_counter() - start_load
                    print(f"{task_name}: dataset ready in "
                          f"{results['dataset_load_seconds'][task_name]:.2f}s; reusing across {len(dims)} dims")
                    results["tasks"].append(task_name)
                    results["metrics"][task_name] = prepared_task.metadata.main_score
                    # Transforms can consume RNG state. Every dim starts from the same post-load state.
                    loaded_python_state, loaded_numpy_state = random.getstate(), np.random.get_state()
                    for dim in dims:
                        random.setstate(loaded_python_state)
                        np.random.set_state(loaded_numpy_state)
                        # Copy task-local RNGs/configuration, but share the prepared dataset in RAM.
                        task = deepcopy(prepared_task, {id(prepared_task.dataset): prepared_task.dataset})
                        adapter.set_dim(dim)
                        adapter.mteb_model_meta = ModelMeta.create_empty(overwrites={
                            "name": f"local/{directory.name}", "revision": f"epoch-{epoch}-dim-{dim}",
                            "embed_dim": dim, "max_tokens": max_length,
                            "modalities": ["text"], "similarity_fn_name": "cosine"})
                        # Recompute scores for each dimension/epoch; reuse data and full embeddings.
                        # None selects serial dataset loading; num_proc=0 is invalid for datasets.
                        with torch.random.fork_rng(devices=devices):
                            model_result = mteb.evaluate(adapter, tasks=[task], cache=None, overwrite_strategy="always",
                                                         co2_tracker=False, raise_error=True, num_proc=None,
                                                         encode_kwargs={"batch_size": batch_size})
                        key = f"dim_{dim}"
                        results["scores"].setdefault(key, {}).update({
                            result.task_name: result.get_score() for result in model_result.task_results})
                        raw = model_result.model_dump(mode="json")
                        if key not in raw_by_dim:
                            raw_by_dim[key] = raw
                        else:
                            raw_by_dim[key]["task_results"].extend(raw["task_results"])
                        raw_path = raw_dir / key / "results.json"
                        # Save each finished task so a later failure does not discard its raw scores.
                        write_json(raw_path, raw_by_dim[key])
                        results["raw_results"][key] = str(raw_path.relative_to(directory))
                        write_eval_progress(directory, results)
                        print(f"{task_name} | dim={dim} | {results['metrics'][task_name]}="
                              f"{format_score(results['scores'][key].get(task_name))}", flush=True)
                    # Save the completed task and cumulative table before loading the next task.
                    results["completed_tasks"].append(task_name)
                    if len(results["completed_tasks"]) == len(names):
                        results["status"] = "complete"
                    summary = write_eval_progress(directory, results)
                    print(f"Task complete: {task_name} "
                          f"({len(results['completed_tasks'])}/{len(names)}). "
                          f"Summary updated: {directory / 'summary.txt'}", flush=True)
                    print(summary, flush=True)
                finally:
                    adapter.clear_cache()
                    if task is not None:
                        task.unload_data()
                    if prepared_task is not None:
                        prepared_task.unload_data()

    finally:
        model.train(was_training)
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    print(f"MTEB results: {directory / 'result.json'}")
    print(f"All epochs: {directory / 'summary_all_epochs.txt'} | CSV: {directory / 'results.csv'}")
    return json_safe(results)


@offline_evaluation
def evaluate(checkpoint_dir=None, *, method="mrl", model_name=None, model_key=None,
             matryoshka_dims=None, device=None, output_dir=None, pooling=None,
             eval_tasks=None, eval_splits=None, eval_languages=None,
             eval_batch_size=64, eval_max_length=256, seed=42, eval_data_dir=None):
    eval_data_dir = eval_data_dir or DEFAULT_EVAL_DATA_DIR
    if eval_tasks is None:
        eval_tasks = DEFAULT_EVAL_TASKS
    if not eval_tasks:
        raise ValueError("Provide --eval_tasks with the selected MTEB task names.")
    require_local_tasks(eval_data_dir, eval_tasks, eval_splits or ["test"], eval_languages or ["eng"], seed)
    checkpoint = Path(checkpoint_dir).expanduser().resolve() if checkpoint_dir else None
    config_path = checkpoint / "training_config.json" if checkpoint else None
    configuration = json.loads(config_path.read_text()) if config_path and config_path.is_file() else {}
    if checkpoint and (checkpoint / "flag_config.json").is_file():
        from FlagMRL.model import FlagTextModel

        model = FlagTextModel.from_pretrained(checkpoint, local_files_only=True)
        source = str(checkpoint)
        pooling = pooling or model.embedding_pooling
        if matryoshka_dims is None:
            matryoshka_dims = model.projector.dims
    else:
        if not checkpoint and not model_name:
            if not model_key:
                raise ValueError("Provide a checkpoint, model_name, or model_key.")
            from MRL.config import MODEL_CONFIGS

            model_name = MODEL_CONFIGS[model_key]["model_name"]
        source = resolve_local_model(str(checkpoint) if checkpoint else model_name)
        model = AutoModel.from_pretrained(source, local_files_only=True, trust_remote_code=True)
        pooling = pooling or configuration.get("pooling", "mean" if method in {"ese", "original"} else "cls")
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs a pad token or EOS token for batching.")
        tokenizer.pad_token = tokenizer.eos_token
    dims = evaluation_dims(model.config.hidden_size, matryoshka_dims if matryoshka_dims is not None
                           else configuration.get("matryoshka_dims"))
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval()
    model.embedding_pooling = pooling
    output_dir = output_dir or checkpoint or Path("results") / "original" / (model_key or Path(source).name)
    return evaluate_model(model, tokenizer, dims, output_dir, eval_tasks=eval_tasks,
                          eval_splits=eval_splits, eval_languages=eval_languages,
                          batch_size=eval_batch_size, max_length=eval_max_length, pooling=pooling, seed=seed,
                          eval_data_dir=eval_data_dir)


def main(method="mrl", argv=None):
    from MRL.config import MODEL_CONFIGS

    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint_dir")
    source.add_argument("--model_name", help="Downloaded encoder directory or supported repo ID (offline only)")
    source.add_argument("--model_key", choices=list(MODEL_CONFIGS))
    parser.add_argument("--matryoshka_dims", nargs="+", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--pooling", choices=["cls", "mean"], default=None)
    parser.add_argument("--seed", type=int, default=42)
    add_eval_args(parser)
    args = parser.parse_args(argv)
    validate_eval_args(args)
    return evaluate(method=method, **vars(args))
