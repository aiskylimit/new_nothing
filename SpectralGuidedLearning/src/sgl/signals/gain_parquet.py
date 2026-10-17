"""Turn per-step answer gains (sgl.signals.answer_gain) into a signal parquet sgl.allocation.build can read.

sgl.allocation.build reads ``step_strengths`` (spectral gate) and ``step_entropies`` (the IWC
signal) from one parquet. The answer-gain arm feeds the gain through the same IWC-Stable formula
by putting it in ``step_entropies``. ``step_strengths`` only matters if a gate < 1.0 is used; the
no-gate arms (energy threshold p=1.0) ignore it, so when no spectral capture exists it is filled
with ones (which keeps every step at p=1.0) and the gradient capture can be skipped entirely.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="train-segmented.jsonl")
    parser.add_argument("--gains", required=True, help="step_answer_gain.json from sgl.signals.answer_gain")
    parser.add_argument("--output", required=True, help="signal-answer-gain.parquet")
    parser.add_argument(
        "--strengths",
        help="optional spectral-strengths.parquet whose step_strengths are carried over "
        "(default: ones, i.e. no spectral information)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    records = [json.loads(line) for line in open(args.data_path)]
    gains = json.loads(Path(args.gains).read_text())
    strengths = {}
    if args.strengths:
        strengths = {int(r.id): list(r.step_strengths) for r in pd.read_parquet(args.strengths).itertuples()}

    rows, missing = [], 0
    for record in records:
        n_steps = len(record["steps"])
        gain = gains.get(str(record["id"]))
        if gain is None:
            # sample without a \boxed answer: no gain signal -> constant -> uniform (1.0) weights
            gain, missing = [0.0] * n_steps, missing + 1
        if len(gain) != n_steps:
            raise ValueError(f"record {record['id']}: {len(gain)} gains for {n_steps} steps")
        step_strengths = strengths.get(record["id"], [1.0] * n_steps)
        if len(step_strengths) != n_steps:
            raise ValueError(f"record {record['id']}: {len(step_strengths)} strengths for {n_steps} steps")
        rows.append(
            {
                "id": record["id"],
                "k_star": 0,
                "n_response_tokens": record["response_token_span"][1] - record["response_token_span"][0],
                "n_steps": n_steps,
                "step_strengths": [float(x) for x in step_strengths],
                "step_entropies": [float(x) for x in gain],
            }
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.output)
    print(f"wrote {len(rows)} rows -> {args.output} ({missing} without a gain signal)")


if __name__ == "__main__":
    main()
