"""YAML configs with `inherit:` (files merged left to right, then this file) and `--set a.b=value` overrides."""

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    merged = {}
    for parent in cfg.pop("inherit", []) or []:
        merged = deep_merge(merged, _load((path.parent / parent).resolve()))
    return deep_merge(merged, cfg)


def apply_overrides(cfg: dict, overrides) -> dict:
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def load_config(paths, overrides=None) -> dict:
    """paths: one config file or a list, merged left to right; then the --set overrides."""
    cfg = {}
    for path in [paths] if isinstance(paths, (str, Path)) else paths:
        cfg = deep_merge(cfg, _load(Path(path).resolve()))
    return apply_overrides(cfg, overrides)


def resolve_path(p) -> Path:
    """Relative paths in configs are relative to the SElOT project root."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def run_paths(cfg: dict) -> dict:
    """Where every stage reads and writes. work_dir: large files; results_dir: small files."""
    work = resolve_path(cfg["work_dir"])
    results = resolve_path(cfg.get("results_dir") or f"results_{cfg['run_id']}")
    model_tag = cfg["model"]["tag"]
    fd = cfg.get("fit_data") or {}
    behaviour = fd.get("behaviour", "llamaguard")
    wf = f"wf{fd.get('workflow', 2)}"          # fit-data workflow (selot/fit_data.py): 1 = first Colab run, 2 = current
    return {
        "acts": work / "acts" / model_tag / wf / behaviour,
        "behavior": work / "behavior" / model_tag,       # behaviour labels: the same for both workflows
        "fitsets": work / "fitsets" / model_tag / wf / behaviour,
        "map": work / "maps" / cfg["run_id"] / "map.pt",
        "generations": work / "generations" / cfg["run_id"],
        "results": results,
    }


def add_config_args(parser):
    parser.add_argument("--config", required=True, action="append",
                        help="YAML config (see configs/); repeat to merge, e.g. base + model + method")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        help="override a config key, e.g. --set method.beta=1000 (repeatable)")
    return parser
