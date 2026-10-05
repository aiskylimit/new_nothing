"""Pick the important segments of every trace from its IG scores (Segment-Selective SFT).

Port of SegmentSelectiveSFT/Attribution/get_important_segments.py:

    strength_k    = sum|IG| / sqrt(n_tokens)          (length-normalised attribution mass)
    keep          = strongest segments until their normalised cumulative strength >= --cumulative-ratio
    coherence_k   = |sum IG| / sum|IG|                (1 = every token pushes the same way)
    selected      = kept segments with coherence <= --coherence-max, in original order

A trace with no attribution signal at all selects nothing (training then falls back to the
always-learned first / second-to-last / last segments, see sgl.ssft.build).
"""
import argparse
import json
from pathlib import Path

import numpy as np


def to_compact(row) -> list[tuple[int, float, float]]:
    """Compact rows {"segments": [[n, sum|IG|, sum IG], ...]} or full per-token rows [[...], ...]."""
    if isinstance(row, dict):
        return [tuple(item) for item in row["segments"]]
    out = []
    for segment in row:
        n = len(segment)
        out.append((n, float(np.sum(np.abs(segment))) if n else 0.0, float(np.sum(segment)) if n else 0.0))
    return out


def select_segments(compact: list[tuple[int, float, float]], cumulative_ratio: float = 0.7,
                    coherence_max: float = 0.8) -> list[int]:
    strengths = [sum_abs / (n ** 0.5) if n else 0.0 for n, sum_abs, _ in compact]
    ranked = sorted(enumerate(strengths), key=lambda item: -item[1])
    order = [index for index, _ in ranked]
    values = np.array([value for _, value in ranked])
    total = values.sum()
    if total <= 0:
        return []
    cumulative = np.cumsum(values / total)
    for cut in range(len(cumulative)):
        if cumulative[cut] >= cumulative_ratio:
            break
    important = sorted(order[: cut + 1])
    coherence = [abs(sum_signed) / sum_abs if sum_abs > 0 else 1.0 for _, sum_abs, sum_signed in compact]
    return [index for index in important if coherence[index] <= coherence_max]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", required=True, help="rows with segments (sgl.data.s1k)")
    parser.add_argument("--ig", required=True, help="IG scores, row-aligned (sgl.ssft.attribution --output)")
    parser.add_argument("--output", required=True, help="--data-path rows + selected_spans_ids")
    parser.add_argument("--cumulative-ratio", type=float, default=0.7, help="'top70' in upstream file names")
    parser.add_argument("--coherence-max", type=float, default=0.8, help="'cohe80' in upstream file names")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rows = [json.loads(line) for line in open(args.data_path) if line.strip()]
    scores = [to_compact(json.loads(line)) for line in open(args.ig) if line.strip()]
    if len(rows) != len(scores):
        raise SystemExit(f"{len(rows)} rows but {len(scores)} IG rows: files are not aligned")
    ratios = []
    for index, (row, compact) in enumerate(zip(rows, scores)):
        if len(compact) != len(row["segments"]):
            raise SystemExit(f"row {index}: {len(compact)} IG segments for {len(row['segments'])} segments")
        row["selected_spans_ids"] = select_segments(compact, args.cumulative_ratio, args.coherence_max)
        ratios.append(len(row["selected_spans_ids"]) / len(compact) if compact else 0.0)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"selected segment share: mean {float(np.mean(ratios)) if ratios else 0.0:.3f} -> {output}")


if __name__ == "__main__":
    main()
