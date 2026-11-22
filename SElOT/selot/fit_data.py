"""Per-model fitting sets (steps v and vi of the data pipeline; steps i-iv are scripts/build_fit_sets.py, which also
fixes the train/val split by behaviour group, the same for every model).

(v)  Behaviour labelling. The unsteered model answers every pooled prompt (stage `label`, scripts/label_behaviour.py):
       refused    a vanilla harmful prompt enters D_r when the model refuses: the response starts with a refusal
                  prefix AND the guard calls it safe (a harmless answer that is not a refusal is dropped)
       jailbreak  a jailbreak prompt enters D_j when the guard calls the response unsafe
       benign     a benign prompt enters D_b when the response is not a refusal (refusal-prefix matcher; a safety
                  guard cannot tell an answer from a refusal)
     fit_data.behaviour: llamaguard (the guard is Llama Guard), keyword (guard = refusal prefix) or
     none (ablation: every pooled prompt is kept).
(vi) Balancing and split, by fit_data.workflow:
     2  (default) the pools carry the fixed `split` (train / val, behaviour groups shared by the three pools). D_r and
        D_j are capped per topic within each split: train at topic_cap - round(topic_cap / val_folds), val at
        round(topic_cap / val_folds) (150 -> 129 + 21), so val stays about 1 / val_folds.
     1  the pipeline of the first Colab run: the pool split is ignored; D_r and D_j are capped per topic on the whole
        set, then 1 / val_folds of each set is held out per model (stratified by topic, benign by source), whole
        `group_v1` groups at a time (near-duplicates within one pool), fixed seed. The vanilla and the jailbreak
        version of one behaviour can land on different sides.
     Within a topic the curated sources share the cap equally, WildGuardMix fills the rest. The train part of D_r and
     D_j gets inverse-topic-frequency weights (mean 1, clipped at max_weight).

Outputs per model, workflow and behaviour mode:
<work_dir>/fitsets/<model.tag>/wf<workflow>/<behaviour>/{benign,refused,jailbreak}.jsonl.
The extract stage reads the train rows and their weights (fit_records).
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from selot.data import load_records

FIT_SETS = ("benign", "refused", "jailbreak")
FILLER_PREFIX = "wgm_"          # WildGuardMix only fills what the curated sources cannot

DEFAULTS = {
    "behaviour": "llamaguard",
    "workflow": 2,              # 1: first Colab run (per-pool groups, per-model split) | 2: fixed behaviour-group split
    "engine": "hf",             # hf | vllm: engine of the label stage's responses (Llama Guard always runs on vLLM)
    "vllm_gpu_mem": 0.85,
    "guard": {"path": None, "dtype": "bfloat16"},
    "max_new_tokens": {"harmful": 256, "benign": 64},   # a refusal shows in the first tokens
    "batch_size": 32,
    "topic_cap": 150,
    "val_folds": 7,             # must match build_fit_sets.py --val-folds (the split itself is fixed there)
    "max_weight": 5.0,
    "seed": 0,
}


def data_cfg(cfg: dict) -> dict:
    out = json.loads(json.dumps(DEFAULTS))
    for k, v in (cfg.get("fit_data") or {}).items():
        out[k] = {**out[k], **v} if isinstance(out.get(k), dict) and isinstance(v, dict) else v
    if out["engine"] not in ("hf", "vllm"):
        raise ValueError(f"fit_data.engine must be hf or vllm, not {out['engine']!r}")
    if out["workflow"] not in (1, 2):
        raise ValueError(f"fit_data.workflow must be 1 or 2, not {out['workflow']!r}")
    if out["behaviour"] not in ("llamaguard", "keyword", "none"):
        raise ValueError(f"fit_data.behaviour must be llamaguard, keyword or none, not {out['behaviour']!r}")
    return out


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").split("\n") if l.strip()]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.replace(path)


def read_labels(path: Path) -> Dict[str, dict]:
    """id -> label row; the last row of an id wins, so re-judged rows replace old ones."""
    return {str(r["id"]): r for r in read_jsonl(path)}


def pools(cfg: dict) -> Dict[str, List[dict]]:
    """The model-independent candidate pools named in data.fit."""
    out = {}
    for name in FIT_SETS:
        recs = load_records(cfg["data"]["fit"][name])
        for r in recs:
            r["id"] = str(r["id"])
        out[name] = recs
    return out


def unsafe_key(behaviour: str) -> str:
    return f"unsafe_{behaviour}"


def keep(name: str, label: dict, behaviour: str) -> bool:
    if behaviour == "none":
        return True
    if name == "benign":
        return not label["refused"]
    unsafe = label[unsafe_key(behaviour)]
    if name == "refused":
        return (not unsafe) and label["refused"]
    return unsafe


# ---------------------------------------------------------------- balancing
def waterfill(avail: Dict[str, int], target: int) -> Dict[str, int]:
    """Share `target` equally over sources; a source short of its share passes the rest on."""
    take = {s: 0 for s in avail}
    left = target
    while left > 0:
        act = sorted((s for s in avail if avail[s] > take[s]), key=lambda s: (avail[s] - take[s], s))
        if not act:
            break
        share = max(1, left // len(act))
        for s in act:
            g = min(share, avail[s] - take[s], left)
            take[s] += g
            left -= g
            if left == 0:
                break
    return take


def cap_per_topic(recs: List[dict], cap: int, rng: random.Random) -> List[dict]:
    """At most `cap` prompts per topic: curated sources share it equally, WildGuardMix tops up.
    Records without a topic are kept as they are."""
    cells = defaultdict(lambda: defaultdict(list))
    free = []
    for r in recs:
        if r.get("topic"):
            cells[r["topic"]][r.get("source", "-")].append(r)
        else:
            free.append(r)
    out = list(free)
    for topic, by_src in sorted(cells.items()):
        primary = {s: len(v) for s, v in by_src.items() if not s.startswith(FILLER_PREFIX)}
        filler = sorted(s for s in by_src if s.startswith(FILLER_PREFIX))
        take = waterfill(primary, cap)
        left = cap - sum(take.values())
        for s in filler:
            take[s] = min(left, len(by_src[s]))
            left -= take[s]
        for s, k in sorted(take.items()):
            out += rng.sample(by_src[s], k) if k < len(by_src[s]) else by_src[s]
    order = {id(r): i for i, r in enumerate(recs)}
    return sorted(out, key=lambda r: order[id(r)])


def split_val(recs: List[dict], folds: int, seed: int, stratum) -> None:
    """Sets r["split"] = train | val: about 1/folds of every stratum goes to val, whole `group`s at a time."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for r in recs:
        groups[r.get("group") or f"_{r['id']}"].append(r)
    by_stratum = defaultdict(list)
    for g, rows in groups.items():
        by_stratum[Counter(stratum(r) for r in rows).most_common(1)[0][0]].append(g)
    for r in recs:
        r["split"] = "train"
    for s, gs in sorted(by_stratum.items()):
        gs = sorted(gs)
        rng.shuffle(gs)
        want = round(sum(len(groups[g]) for g in gs) / folds)
        got = 0
        for g in gs:
            if got >= want:
                break
            if got + len(groups[g]) > want and got > 0 and got + len(groups[g]) - want > want - got:
                continue                     # this group would overshoot more than it helps; try a smaller one
            for r in groups[g]:
                r["split"] = "val"
            got += len(groups[g])


def topic_weights(recs: List[dict], max_weight: float) -> None:
    """Inverse topic frequency over the train rows, mean 1, clipped at max_weight (then renormalised).
    Val rows and rows without a topic get weight 1."""
    train = [r for r in recs if r["split"] == "train"]
    for r in recs:
        r["weight"] = 1.0
    topical = [r for r in train if r.get("topic")]
    if not topical:
        return
    count = Counter(r["topic"] for r in topical)
    raw = [len(topical) / (len(count) * count[r["topic"]]) for r in topical]
    w = [min(x, max_weight) for x in raw]
    mean = sum(w) / len(w)
    for r, x in zip(topical, w):
        r["weight"] = round(x / mean, 6)


def build_sets(cfg: dict, pool: Dict[str, List[dict]], labels: Dict[str, Dict[str, dict]]):
    """(iv) + (v) for one model. Returns ({set: records with split and weight}, report)."""
    dc = data_cfg(cfg)
    behaviour = dc["behaviour"]
    rng = random.Random(dc["seed"])
    sets, report = {}, {"behaviour": behaviour, "workflow": dc["workflow"],
                        "split": "pool" if dc["workflow"] == 2 else "per_model", "topic_cap": dc["topic_cap"], "val_folds": dc["val_folds"], "sets": {}}
    for name in FIT_SETS:
        recs = [dict(r) for r in pool[name]]
        if behaviour != "none":
            missing = [r["id"] for r in recs if r["id"] not in labels[name]]
            if missing:
                raise SystemExit(f"{len(missing)} {name} prompts have no behaviour label: run the label stage")
        kept = [r for r in recs if keep(name, labels[name].get(r["id"], {}), behaviour)]
        if dc["workflow"] == 2:
            if not all(r.get("split") in ("train", "val") for r in recs):
                raise SystemExit(f"workflow 2: the {name} pool has no split column (rebuild with "
                                 "scripts/build_fit_sets.py)")
            if name == "benign":
                capped = kept
            else:
                cap_val = round(dc["topic_cap"] / dc["val_folds"])
                part = {sp: cap_per_topic([r for r in kept if r["split"] == sp], cap, rng)
                        for sp, cap in (("train", dc["topic_cap"] - cap_val), ("val", cap_val))}
                ids = {id(r) for rs in part.values() for r in rs}
                capped = [r for r in kept if id(r) in ids]
        else:
            if not all(r.get("group_v1") for r in recs):
                raise SystemExit(f"workflow 1: the {name} pool has no group_v1 column (rebuild with "
                                 "scripts/build_fit_sets.py)")
            for r in recs:
                r["group"] = r.pop("group_v1")
                r.pop("split", None)
            capped = cap_per_topic(kept, dc["topic_cap"], rng) if name != "benign" else kept
        if not capped:
            raise SystemExit(f"no {name} prompt left after behaviour labelling")
        if dc["workflow"] == 1:
            strat = (lambda r: r.get("source", "-")) if name == "benign" else (lambda r: r.get("topic") or "-")
            split_val(capped, dc["val_folds"], dc["seed"], strat)
        for r in capped:
            r.pop("group_v1", None)
        if name == "benign":
            for r in capped:
                r["weight"] = 1.0
        else:
            topic_weights(capped, dc["max_weight"])
        sets[name] = capped
        report["sets"][name] = summarize(recs, kept, capped)
        if name == "refused" and behaviour != "none":     # why vanilla prompts were dropped, per source
            why = defaultdict(Counter)
            for r in recs:
                l = labels[name][r["id"]]
                reason = ("kept" if keep(name, l, behaviour) else
                          "unsafe" if l[unsafe_key(behaviour)] else "safe_not_refused")
                why[r.get("source", "-")][reason] += 1
            report["sets"][name]["outcome_by_source"] = {s: dict(c) for s, c in sorted(why.items())}
    return sets, report


def summarize(pool, kept, final) -> dict:
    def tab(rows, key):
        return dict(sorted(Counter(r.get(key) or "-" for r in rows).items()))
    train = [r for r in final if r["split"] == "train"]
    return {"pool": len(pool), "kept": len(kept), "final": len(final), "train": len(train),
            "val": len(final) - len(train),
            "kept_rate_by_source": {s: round(k / n, 3) for s, n in tab(pool, "source").items()
                                    for k in [tab(kept, "source").get(s, 0)]},
            "train_by_topic": tab(train, "topic"), "train_by_source": tab(train, "source"),
            "val_by_topic": tab([r for r in final if r["split"] == "val"], "topic"),
            "weight_range": [min(r["weight"] for r in train), max(r["weight"] for r in train)] if train else None}


def fit_records(cfg: dict, paths: dict, name: str, split: str = "train") -> List[dict]:
    """Rows of the per-model fit set `name` in `split` (written by the label stage)."""
    path = paths["fitsets"] / f"{name}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} does not exist: run the label stage first")
    recs = [r for r in read_jsonl(path) if r.get("split", "train") == split]
    if not recs:
        raise SystemExit(f"no {name} prompt in split {split!r} of {path}")
    return recs
