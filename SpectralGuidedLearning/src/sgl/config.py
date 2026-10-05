"""Experiment configuration: YAML files composed with `defaults`, `${...}` interpolation and
dotted command-line overrides.

    defaults:                       # merged first, in order; paths are relative to this file
      - ../models/r1-qwen-1.5b.yaml
    name: sft-nll
    track: r1-qwen-1.5b-palign
    data_dir: data/${track}          # ${a.b} reads another key of the merged config
    stages:
      train:
        args: {data-path: "${data_dir}/train-vanilla.jsonl"}

Merging is a deep merge of mappings; any other value (lists included) is replaced, so a later
file or an override always wins. Interpolation runs once on the fully merged tree: a string that
is exactly one `${key}` takes the referenced value with its type, otherwise references are
formatted into the string. `${env:NAME}` / `${env:NAME,default}` read the environment.
"""
import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml

_REFERENCE = re.compile(r"\$\{([^${}]+)\}")


class _Loader(yaml.SafeLoader):
    """SafeLoader with YAML 1.2 booleans: only true/false (any case) are bools, so values like
    `think_prefix: off` or `answer: no` stay strings instead of silently becoming False."""


_Loader.yaml_implicit_resolvers = {
    key: [(tag, regexp) for tag, regexp in resolvers if tag != "tag:yaml.org,2002:bool"]
    for key, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:bool", re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF")
)


def parse_yaml(text: str):
    return yaml.load(text, Loader=_Loader)
_MAX_INTERPOLATION_DEPTH = 32


class ConfigError(ValueError):
    pass


def deep_merge(base: dict, override: dict) -> dict:
    """New dict: `override` merged into `base`, recursing only where both sides are mappings."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_yaml_tree(path: str | Path, _stack: tuple[Path, ...] = ()) -> dict:
    """One YAML file with its `defaults` chain merged underneath it (no interpolation yet)."""
    path = Path(path).resolve()
    if path in _stack:
        cycle = " -> ".join(str(p) for p in (*_stack, path))
        raise ConfigError(f"defaults cycle: {cycle}")
    data = parse_yaml(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    defaults = data.pop("defaults", []) or []
    if isinstance(defaults, str):
        defaults = [defaults]
    merged: dict = {}
    for entry in defaults:
        merged = deep_merge(merged, load_yaml_tree(path.parent / entry, (*_stack, path)))
    return deep_merge(merged, data)


def parse_override(text: str) -> tuple[list[str], Any]:
    """`a.b.c=value` -> (["a", "b", "c"], yaml-parsed value). Keys may contain '-'."""
    if "=" not in text:
        raise ConfigError(f"override {text!r} must look like key.path=value")
    key, raw = text.split("=", 1)
    parts = [part for part in key.strip().split(".") if part]
    if not parts:
        raise ConfigError(f"override {text!r} has an empty key")
    return parts, parse_yaml(raw) if raw.strip() else ""


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    config = copy.deepcopy(config)
    for text in overrides:
        parts, value = parse_override(text)
        node = config
        for part in parts[:-1]:
            child = node.get(part)
            if child is None:
                child = node[part] = {}
            if not isinstance(child, dict):
                raise ConfigError(f"override {text!r}: {part!r} is not a mapping")
            node = child
        node[parts[-1]] = value
    return config


def lookup(config: dict, dotted: str) -> Any:
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise ConfigError(f"unknown reference ${{{dotted}}}")
        node = node[part]
    return node


def _resolve_reference(config: dict, key: str) -> Any:
    key = key.strip()
    if key.startswith("env:"):
        name, has_default, default = key[4:].partition(",")
        value = os.environ.get(name.strip())
        if value is None:
            if not has_default:
                raise ConfigError(f"environment variable {name.strip()} is not set")
            return default.strip()
        return value
    return lookup(config, key)


def _interpolate(value: Any, config: dict, depth: int) -> Any:
    if depth > _MAX_INTERPOLATION_DEPTH:
        raise ConfigError(f"interpolation too deep (cycle?) at {value!r}")
    if isinstance(value, dict):
        return {key: _interpolate(item, config, depth) for key, item in value.items()}
    if isinstance(value, list):
        return [_interpolate(item, config, depth) for item in value]
    if not isinstance(value, str) or "${" not in value:
        return value
    whole = _REFERENCE.fullmatch(value)
    if whole:
        return _interpolate(_resolve_reference(config, whole.group(1)), config, depth + 1)

    def substitute(match: re.Match) -> str:
        resolved = _interpolate(_resolve_reference(config, match.group(1)), config, depth + 1)
        if isinstance(resolved, (dict, list)):
            raise ConfigError(f"${{{match.group(1)}}} is a {type(resolved).__name__}, cannot embed it in {value!r}")
        return _format_scalar(resolved)

    return _REFERENCE.sub(substitute, value)


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return "" if value is None else str(value)


def interpolate(config: dict) -> dict:
    return _interpolate(config, config, 0)


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Merged, overridden and interpolated experiment config."""
    tree = load_yaml_tree(path)
    tree = apply_overrides(tree, overrides or [])
    return interpolate(tree)
