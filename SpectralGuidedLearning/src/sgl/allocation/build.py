"""Build the Vanilla-IWC and IWC-Stable weighted datasets from spectral signals.

The two outputs intentionally reuse the same spectral selection as
``train-spectral.jsonl``.  Their only additional field is ``loss_weights``;
``loss_mask`` and ``input_ids`` remain identical to spectral so a result can be
attributed to allocation rather than a different supervised set.
"""

import argparse
import json
import statistics
from pathlib import Path

import pandas as pd
import yaml

from sgl.allocation.iwc import (
    reverse_iwc_step_weights,
    shuffled_iwc_step_weights,
    stable_iwc_step_weights,
    vanilla_iwc_step_weights,
)
from sgl.allocation.region import (
    PREFIX,
    region_budget,
    region_gain_ratio,
    region_stable_step_weights,
    step_regions,
)
from sgl.segment.sentence import record_step_spans
from sgl.selection.energy import build_loss_mask, select_steps_by_energy, selection_stats

# Every non-"iwc" variant is IWC-Stable with a specific (interpolation, entropy-mapping) choice --
# the shared token-mass normalization is what makes them comparable controls (see docs/iwc-baselines.md).
_STABLE_VARIANTS = {
    "iwc-stable": lambda entropies, lengths, temperature, interpolation, clip, epsilon, seed:
        stable_iwc_step_weights(entropies, lengths, temperature, interpolation, clip, epsilon),
    "iwc-stable-lambda0": lambda entropies, lengths, temperature, interpolation, clip, epsilon, seed:
        stable_iwc_step_weights(entropies, lengths, temperature, 0.0, clip, epsilon),
    "iwc-stable-shuffled": lambda entropies, lengths, temperature, interpolation, clip, epsilon, seed:
        shuffled_iwc_step_weights(entropies, lengths, temperature, interpolation, clip, epsilon, seed),
    "iwc-stable-reverse": lambda entropies, lengths, temperature, interpolation, clip, epsilon, seed:
        reverse_iwc_step_weights(entropies, lengths, temperature, interpolation, clip, epsilon),
}

# Region variants (sgl.allocation.region, SARW) need the prefix/continuation region of each step:
# region-iwc = a_k only, region-gain = b_r only, sarw = a_k * b_r, region-cont = the teacher-prefix
# steps dropped from the loss (continuation-only SFT, the b_P -> 0 limit without inflating weights).
_REGION_VARIANTS = ("region-iwc", "region-gain", "sarw", "region-cont")


def build_example(
    record: dict,
    strengths: list[float],
    entropies: list[float],
    threshold: float,
    variant: str,
    temperature: float,
    interpolation: float,
    clip: float,
    epsilon: float,
    seed: int = 42,
    regions: list[str] | None = None,
    region_ratio: float = 1.0,
) -> tuple[dict, dict]:
    """Build one weighted record and its selection/allocation diagnostics."""
    step_spans = record_step_spans(record)
    if len(strengths) != len(step_spans) or len(entropies) != len(step_spans):
        raise ValueError(
            f"record {record['id']}: steps={len(step_spans)}, strengths={len(strengths)}, "
            f"entropies={len(entropies)}"
        )
    selected = select_steps_by_energy(strengths, threshold)
    if variant == "region-cont":
        if regions is None or len(regions) != len(step_spans):
            raise ValueError(f"record {record['id']}: {variant} needs one region per step")
        selected = [index for index in selected if regions[index] != PREFIX]
    selected_entropies = [entropies[index] for index in selected]
    selected_lengths = [step_spans[index][1] - step_spans[index][0] for index in selected]
    if variant == "iwc":
        selected_weights = vanilla_iwc_step_weights(selected_entropies, selected_lengths, temperature)
    elif variant in _STABLE_VARIANTS:
        # per-record seed offset so "shuffled" draws an independent permutation per sample
        # instead of repeating one fixed pattern relative to step position.
        selected_weights = _STABLE_VARIANTS[variant](
            selected_entropies, selected_lengths, temperature, interpolation, clip, epsilon,
            seed + record["id"],
        )
    elif variant == "region-cont":
        selected_weights = [1.0] * len(selected)
    elif variant in _REGION_VARIANTS:
        if regions is None or len(regions) != len(step_spans):
            raise ValueError(f"record {record['id']}: {variant} needs one region per step")
        selected_regions = [regions[index] for index in selected]
        within = (
            region_stable_step_weights(
                selected_entropies, selected_lengths, selected_regions,
                temperature, interpolation, clip, epsilon,
            )
            if variant != "region-gain" else [1.0] * len(selected)
        )
        budget = (
            region_budget(selected_lengths, selected_regions, region_ratio)
            if variant != "region-iwc" else [1.0] * len(selected)
        )
        selected_weights = [a * b for a, b in zip(within, budget)]
    else:
        raise ValueError(f"unknown IWC variant: {variant}")

    loss_mask = build_loss_mask(len(record["input_ids"]), step_spans, selected)
    loss_weights = [0.0] * len(loss_mask)
    for step_index, weight in zip(selected, selected_weights):
        start, end = step_spans[step_index]
        loss_weights[start:end] = [weight] * (end - start)

    # The existing spectral arm always supervises the stop token.  Its coefficient
    # is fixed at one, which preserves exact spectral behaviour at lambda=0.
    for position in range(record["response_token_span"][1], len(loss_mask)):
        loss_mask[position] = 1
        loss_weights[position] = 1.0

    selected_token_mass = sum(selected_lengths)
    weighted_token_mass = sum(
        length * weight for length, weight in zip(selected_lengths, selected_weights)
    )
    stats = selection_stats(step_spans, selected)
    stats.update(
        {
            "selected_token_mass": selected_token_mass,
            "weighted_token_mass": weighted_token_mass,
            "weight_mass_ratio": weighted_token_mass / max(selected_token_mass, 1),
            "weight_min": min(selected_weights, default=1.0),
            "weight_max": max(selected_weights, default=1.0),
        }
    )
    return {
        "id": record["id"],
        "input_ids": record["input_ids"],
        "loss_mask": loss_mask,
        "loss_weights": loss_weights,
    }, stats


def emit_dataset(
    records: list[dict],
    signals: dict[int, tuple[list[float], list[float]]],
    output_path: Path,
    threshold: float,
    variant: str,
    temperature: float,
    interpolation: float,
    clip: float,
    epsilon: float,
    seed: int = 42,
    regions: dict[int, list[str]] | None = None,
    region_ratio: float = 1.0,
) -> list[dict]:
    """Write one IWC arm and return per-record diagnostics."""
    all_stats = []
    with output_path.open("w") as handle:
        for record in records:
            strengths, entropies = signals[record["id"]]
            example, stats = build_example(
                record, strengths, entropies, threshold, variant, temperature,
                interpolation, clip, epsilon, seed,
                regions[record["id"]] if regions is not None else None, region_ratio,
            )
            handle.write(json.dumps(example) + "\n")
            all_stats.append(stats)
    return all_stats


def summarize(stats: list[dict]) -> dict:
    """Corpus diagnostics needed to check the IWC-Stable invariance claim."""
    return {
        "samples": len(stats),
        "step_drop_mean": statistics.mean(item["step_drop"] for item in stats),
        "token_drop_mean": statistics.mean(item["token_drop"] for item in stats),
        "weight_mass_ratio_mean": statistics.mean(item["weight_mass_ratio"] for item in stats),
        "weight_mass_ratio_min": min(item["weight_mass_ratio"] for item in stats),
        "weight_mass_ratio_max": max(item["weight_mass_ratio"] for item in stats),
        "weight_min": min(item["weight_min"] for item in stats),
        "weight_max": max(item["weight_max"] for item in stats),
    }


def record_regions(records: list[dict], tokenizer_name: str) -> dict[int, list[str]]:
    """Prefix/continuation/junction region of every step, from the <End_of_Prefix> marker."""
    from transformers import AutoTokenizer

    from sgl.transforms.provenance import prefix_token_count

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    regions = {}
    for record in records:
        start, end = record["response_token_span"]
        n_prefix = prefix_token_count(tokenizer, record["response"], record["input_ids"][start:end])
        if n_prefix is None:
            raise ValueError(f"record {record['id']}: no <End_of_Prefix> marker; region variants "
                             "are only defined on P-ALIGN hybrid targets")
        regions[record["id"]] = step_regions(record_step_spans(record), start + n_prefix)
    return regions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="optional yaml base; CLI flags override it")
    parser.add_argument("--data-path", help="segmented JSONL from sgl.data.prepare")
    parser.add_argument("--strengths", help="spectral-strengths.parquet from sgl.signals.capture")
    parser.add_argument("--energy-threshold-p", type=float, help="the shared spectral gate threshold")
    parser.add_argument("--temperature", type=float, help="entropy temperature tau")
    parser.add_argument("--interpolation", type=float, help="IWC-Stable lambda in [0, 1]")
    parser.add_argument("--clip", type=float, help="IWC-Stable standardized-entropy clip bound")
    parser.add_argument("--epsilon", type=float, help="standardization stabilizer")
    parser.add_argument("--seed", type=int, help="base seed for the shuffled-entropy control")
    parser.add_argument(
        "--variants",
        help="comma-separated subset of iwc,iwc-stable,iwc-stable-lambda0,iwc-stable-shuffled,"
        "iwc-stable-reverse,region-iwc,region-gain,sarw (default: iwc,iwc-stable)",
    )
    parser.add_argument(
        "--tokenizer",
        help="region variants: tokenizer of the stored input_ids, used to locate <End_of_Prefix>",
    )
    parser.add_argument(
        "--region-signal",
        help="region-gain/sarw: parquet whose step_entropies hold the frozen student's per-step "
        "answer gain (signal-answer-gain.parquet), for the prefix/continuation budget",
    )
    parser.add_argument(
        "--region-gamma", type=float,
        help="region-gain/sarw: b_P / b_C = (G_P / G_C) ** gamma (default 1.0)",
    )
    parser.add_argument(
        "--region-tail-fraction", type=float,
        help="region-gain/sarw: leave the last ceil(f * n_steps) steps of each sample out of the "
        "G_P / G_C estimate (they stay weighted; default 0.0)",
    )
    parser.add_argument(
        "--check-mass", action="store_true",
        help="fail unless every iwc-stable* variant keeps the selected token mass "
        "(mean weighted/selected mass prints as 1.000000)",
    )
    parser.add_argument(
        "--output-name",
        help="with a single variant: write train-<output-name>.jsonl and "
        "<output-name>-selection-stats.json instead of train-<variant>.jsonl and "
        "iwc-selection-stats.json, so several arms can share one data dir",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = yaml.safe_load(Path(args.config).read_text()) if args.config else {}
    overrides = {
        "data_path": args.data_path,
        "strengths": args.strengths,
        "energy_threshold_p": args.energy_threshold_p,
        "temperature": args.temperature,
        "interpolation": args.interpolation,
        "clip": args.clip,
        "epsilon": args.epsilon,
        "seed": args.seed,
        "variants": args.variants,
        "tokenizer": args.tokenizer,
        "region_signal": args.region_signal,
        "region_gamma": args.region_gamma,
        "region_tail_fraction": args.region_tail_fraction,
    }
    config.update({key: value for key, value in overrides.items() if value is not None})
    config.setdefault("energy_threshold_p", 0.95)
    config.setdefault("temperature", 1.0)
    config.setdefault("interpolation", 1.0)
    config.setdefault("clip", 2.0)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("seed", 42)
    config.setdefault("variants", "iwc,iwc-stable")
    required = [key for key in ("data_path", "strengths") if key not in config]
    if required:
        parser.error(f"missing required settings: {required}")

    with open(config["data_path"]) as handle:
        records = [json.loads(line) for line in handle]
    frame = pd.read_parquet(config["strengths"])
    if "step_entropies" not in frame.columns:
        raise ValueError(
            f"{config['strengths']} has no step_entropies; rerun sgl.signals.capture after the IWC update"
        )
    signals = {
        int(row.id): (list(row.step_strengths), list(row.step_entropies))
        for row in frame.itertuples()
    }
    missing = [record["id"] for record in records if record["id"] not in signals]
    if missing:
        raise ValueError(f"spectral signals missing for {len(missing)} records (first: {missing[0]})")

    variants = config["variants"].split(",")
    if args.output_name and len(variants) != 1:
        parser.error("--output-name needs exactly one variant")
    regions, region_ratio = None, 1.0
    if any(variant in _REGION_VARIANTS for variant in variants):
        if "tokenizer" not in config:
            parser.error("region variants need --tokenizer")
        regions = record_regions(records, config["tokenizer"])
        if any(variant in ("region-gain", "sarw") for variant in variants):
            # defaults set here only, so the stats json of the older arms stays byte-identical
            config.setdefault("region_gamma", 1.0)
            config.setdefault("region_tail_fraction", 0.0)
            if "region_signal" not in config:
                parser.error("region-gain/sarw need --region-signal")
            gains = {int(row.id): list(row.step_entropies)
                     for row in pd.read_parquet(config["region_signal"]).itertuples()}
            gain_prefix, gain_cont, region_ratio = region_gain_ratio(
                [gains[record["id"]] for record in records],
                [[end - start for start, end in record_step_spans(record)] for record in records],
                [regions[record["id"]] for record in records],
                config["region_gamma"],
                config["region_tail_fraction"],
            )
            config.update({"region_gain_prefix": gain_prefix, "region_gain_continuation": gain_cont,
                           "region_ratio": region_ratio})
            print(f"answer gain per token: prefix {gain_prefix:.6g}, continuation {gain_cont:.6g} "
                  f"-> b_P/b_C = {region_ratio:.4f} (gamma {config['region_gamma']}, "
                  f"tail fraction {config['region_tail_fraction']})")
    data_dir = Path(config["data_path"]).parent
    summaries = {}
    for variant in variants:
        output_path = data_dir / f"train-{args.output_name or variant}.jsonl"
        stats = emit_dataset(
            records, signals, output_path, config["energy_threshold_p"], variant,
            config["temperature"], config["interpolation"], config["clip"], config["epsilon"],
            config["seed"], regions, region_ratio,
        )
        summaries[variant] = summarize(stats)
        print(
            f"{variant}: {len(stats)} records -> {output_path}; "
            f"mean weighted/selected mass={summaries[variant]['weight_mass_ratio_mean']:.6f}"
        )
        mass = f"{summaries[variant]['weight_mass_ratio_mean']:.6f}"
        mass_preserving = variant.startswith("iwc-stable") or variant in _REGION_VARIANTS
        if args.check_mass and mass_preserving and mass != "1.000000":
            raise SystemExit(f"{variant}: token mass not preserved (mean weighted/selected mass={mass})")

    summary_path = data_dir / (
        f"{args.output_name}-selection-stats.json" if args.output_name else "iwc-selection-stats.json"
    )
    summary_path.write_text(json.dumps({"config": config, "variants": summaries}, indent=2))
    print(f"diagnostics -> {summary_path}")


if __name__ == "__main__":
    main()
