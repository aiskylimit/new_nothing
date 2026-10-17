"""Boost the loss on the last supervised tokens of each response (where the answer is concluded).

A student that imitates long teacher traces can learn the reasoning style without learning to stop:
on Qwen2.5-7B ~16% of generations end in a degenerate loop (e.g. repeating the "✅" of the traces'
"### ✅ Final Answer" line). The last tokens of a response -- the final answer and the stop token --
are a tiny share of the loss, so this multiplies the weight of the last ``--tail-tokens`` supervised
tokens by ``--boost`` and rescales every supervised weight of the sample so its total mass is
unchanged (the same per-sample invariant as IWC-Stable and the region variants).

Input is any weighted or plain dataset (train-*.jsonl with loss_mask and optional loss_weights);
input_ids and loss_mask are copied unchanged.
"""

import argparse
import json
import statistics


def boost_tail(loss_mask: list[int], loss_weights: list[float] | None, tail_tokens: int, boost: float) -> list[float]:
    """Weights with the last `tail_tokens` supervised positions scaled by `boost`, sample mass kept."""
    if tail_tokens < 0 or boost <= 0:
        raise ValueError("tail_tokens must be >= 0 and boost > 0")
    weights = [float(w) for w in loss_weights] if loss_weights is not None else [float(m) for m in loss_mask]
    supervised = [index for index, keep in enumerate(loss_mask) if keep]
    mass = sum(weights[index] for index in supervised)
    for index in supervised[-tail_tokens:] if tail_tokens else []:
        weights[index] *= boost
    boosted = sum(weights[index] for index in supervised)
    if boosted > 0:
        scale = mass / boosted
        weights = [weight * scale if loss_mask[index] else 0.0 for index, weight in enumerate(weights)]
    return weights


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="input train-*.jsonl (loss_mask, optional loss_weights)")
    parser.add_argument("--output", required=True, help="output train-*.jsonl")
    parser.add_argument("--tail-tokens", type=int, default=128, help="last supervised tokens to boost")
    parser.add_argument("--boost", type=float, default=3.0, help="weight multiplier before renormalization")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    tail_shares, max_weights, mass_ratios = [], [], []
    with open(args.data_path) as source, open(args.output, "w") as sink:
        for line in source:
            record = json.loads(line)
            mask = record["loss_mask"]
            before = record.get("loss_weights")
            weights = boost_tail(mask, before, args.tail_tokens, args.boost)
            supervised = [index for index, keep in enumerate(mask) if keep]
            tail = supervised[-args.tail_tokens:] if args.tail_tokens else []
            total = sum(weights[index] for index in supervised)
            original = sum((before[index] if before is not None else 1.0) for index in supervised)
            tail_shares.append(sum(weights[index] for index in tail) / max(total, 1e-12))
            max_weights.append(max(weights))
            mass_ratios.append(total / max(original, 1e-12))
            sink.write(json.dumps({**record, "loss_weights": weights}) + "\n")
    print(
        f"{len(tail_shares)} records -> {args.output}; tail loss share mean {statistics.mean(tail_shares):.4f}, "
        f"max weight {max(max_weights):.3f}, mean mass ratio {statistics.mean(mass_ratios):.6f}"
    )


if __name__ == "__main__":
    main()
