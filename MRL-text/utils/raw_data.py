"""Plan and prepare existing raw files. This module never downloads anything."""

import csv
from pathlib import Path


SOURCE_CATALOG = Path(__file__).with_name("mteb_sources.tsv")


def raw_file_plan(task, splits):
    """Use the same pinned file catalog as the wget-only shell script."""
    name = task.metadata.name
    source = task.metadata.dataset
    prefix = f"https://huggingface.co/datasets/{source['path']}/resolve/{source['revision']}/"
    required = list(dict.fromkeys(([getattr(task, "train_split", "train")]
                                  if task.metadata.type == "Classification" else []) + list(splits)))
    with SOURCE_CATALOG.open() as stream:
        records = list(csv.reader((line for line in stream if not line.startswith("#")), delimiter="\t"))
    plan = []
    for record in records:
        task_name, language, subset, split, role, path, url = record
        if task_name != name or subset not in task.hf_subsets or split not in required:
            continue
        if not url.startswith(prefix):
            raise ValueError(f"The raw-file catalog revision does not match installed MTEB for {name}.")
        plan.append({"subset": subset, "split": split, "role": role, "path": path, "url": url,
                     "raw_path": f"mteb/{name}/{source['revision']}/{path}"})
    roles = {"corpus", "queries", "qrels"} if task.metadata.type == "Retrieval" else {"dataset"}
    if not task.hf_subsets:
        raise ValueError(f"No subsets match the requested languages for {name}.")
    for subset in task.hf_subsets:
        for split in required:
            found = {item["role"] for item in plan if item["subset"] == subset and item["split"] == split}
            if found != roles:
                raise ValueError(f"No complete raw-file recipe for {name}/{subset}/{split} in {SOURCE_CATALOG}.")
    return plan


def require_raw_files(raw_dir, plan):
    root = Path(raw_dir).expanduser().resolve()
    for item in plan:
        path = root / item["raw_path"]
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Raw data missing: {path}. Run download_mteb.sh on an online machine "
                                    "and copy the raw data directory to this server before preparing.")


def prepare_task(task, raw_dir, plan, cache_dir):
    """Read local GZIP/JSONL/Parquet unchanged, then apply native MTEB transforms."""
    from datasets import Dataset, DatasetDict, Value

    require_raw_files(raw_dir, plan)
    root = Path(raw_dir).expanduser().resolve()
    loaded = {}
    for item in plan:
        path = root / item["raw_path"]
        if item["path"] not in loaded:
            if path.suffix == ".parquet":
                dataset = Dataset.from_parquet(str(path), cache_dir=str(cache_dir))
            else:
                # The local JSON reader handles .gz; preserve the original raw file.
                dataset = Dataset.from_json(str(path), cache_dir=str(cache_dir))
            loaded[item["path"]] = dataset
    if task.metadata.type == "Retrieval":
        dataset = {}
        for subset, split in dict.fromkeys((item["subset"], item["split"]) for item in plan):
            parts = {item["role"]: loaded[item["path"]] for item in plan
                     if item["subset"] == subset and item["split"] == split}
            corpus, queries = parts["corpus"], parts["queries"]
            if "_id" in corpus.column_names:
                corpus = corpus.cast_column("_id", Value("string")).rename_column("_id", "id")
            if "_id" in queries.column_names:
                queries = queries.cast_column("_id", Value("string")).rename_column("_id", "id")
            qrels = {}
            for row in parts["qrels"]:
                qrels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = int(row["score"])
            queries = queries.select([i for i, query_id in enumerate(queries["id"]) if query_id in qrels])
            dataset.setdefault(subset, {})[split] = {
                "corpus": corpus, "queries": queries, "relevant_docs": qrels, "top_ranked": None}
        task.dataset = dataset
    elif task.metadata.is_multilingual:
        task.dataset = {}
        for item in plan:
            dataset = loaded[item["path"]]
            if "lang" in dataset.column_names:
                dataset = dataset.remove_columns("lang")
            task.dataset.setdefault(item["subset"], DatasetDict())[item["split"]] = dataset
    else:
        task.dataset = DatasetDict({item["split"]: loaded[item["path"]] for item in plan})
    task.dataset_transform(num_proc=None)
    task.data_loaded = True
