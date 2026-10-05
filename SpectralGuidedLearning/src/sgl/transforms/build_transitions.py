"""Add the L_trans step fields to a masked SFT dataset.

Reads the segmented records (sgl.data.prepare: response text + response_token_span) for the step
structure and a masked dataset (sgl.selection.build_masks / sgl.allocation.build: input_ids + loss_mask
[+ loss_weights]) for the NLL supervision, and writes the masked records plus
step_id / step_end / num_steps / pair_src (sgl.transforms.step_transitions). The NLL mask is copied
untouched, so `--masked train-vanilla.jsonl` gives SFT + L_trans and `--masked
train-spectral.jsonl` gives SGL + L_trans on exactly the same transition pairs.

The response is re-tokenized with offsets exactly as sgl.data.prepare did (one pass over the whole
response, never piecewise) and checked against the stored input_ids before any boundary is mapped.
"""

import argparse
import json
import statistics
from pathlib import Path

from transformers import AutoTokenizer

from sgl.segment.sentence import encode_with_offsets
from sgl.transforms.step_transitions import DEFAULT_MIN_STEP_TOKENS, build_transition_fields


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def transition_fields_for_record(tokenizer, record: dict, min_step_tokens: int) -> dict:
    start, end = record["response_token_span"]
    response_ids, token_starts = encode_with_offsets(tokenizer, record["response"])
    if response_ids != record["input_ids"][start:end]:
        raise RuntimeError(
            f"record {record['id']}: re-tokenized response differs from stored input_ids "
            "(different tokenizer than sgl.data.prepare?)"
        )
    return build_transition_fields(
        record["response"], token_starts, start, len(record["input_ids"]), min_step_tokens
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmented", required=True, help="sgl.data.prepare output (train-segmented.jsonl)")
    parser.add_argument("--masked", required=True,
                        help="masked dataset to extend (train-vanilla.jsonl, train-spectral.jsonl, ...)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", required=True, help="same tokenizer sgl.data.prepare used")
    parser.add_argument("--min-step-tokens", type=int, default=DEFAULT_MIN_STEP_TOKENS)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    segmented = {record["id"]: record for record in load_jsonl(Path(args.segmented))}
    masked = load_jsonl(Path(args.masked))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    steps_per_sample, pairs_per_sample, step_lengths, no_pairs = [], [], [], 0
    with output.open("w") as handle:
        for record in masked:
            source = segmented[record["id"]]
            if source["input_ids"] != record["input_ids"]:
                raise RuntimeError(f"record {record['id']}: input_ids differ between --segmented and --masked")
            fields = transition_fields_for_record(tokenizer, source, args.min_step_tokens)
            handle.write(json.dumps({**record, **fields}) + "\n")

            steps_per_sample.append(fields["num_steps"])
            pairs_per_sample.append(len(fields["pair_src"]))
            no_pairs += not fields["pair_src"]
            lengths = [0] * fields["num_steps"]
            for step in fields["step_id"]:
                if step >= 0:
                    lengths[step] += 1
            step_lengths.extend(lengths)

    print(f"wrote {len(masked)} samples -> {output}")
    print(
        f"steps/sample : mean={statistics.mean(steps_per_sample):.1f} "
        f"median={statistics.median(steps_per_sample):.0f} max={max(steps_per_sample)}"
    )
    print(
        f"pairs/sample : mean={statistics.mean(pairs_per_sample):.1f} "
        f"total={sum(pairs_per_sample):,} samples_without_pairs={no_pairs}"
    )
    print(
        f"tokens/step  : mean={statistics.mean(step_lengths):.1f} "
        f"median={statistics.median(step_lengths):.0f} min={min(step_lengths)} max={max(step_lengths)}"
    )
    (output.parent / f"{output.stem}-stats.json").write_text(
        json.dumps(
            {
                "samples": len(masked),
                "min_step_tokens": args.min_step_tokens,
                "steps_total": sum(steps_per_sample),
                "pairs_total": sum(pairs_per_sample),
                "samples_without_pairs": no_pairs,
                "steps_per_sample_mean": statistics.mean(steps_per_sample),
                "pairs_per_sample_mean": statistics.mean(pairs_per_sample),
                "tokens_per_step_mean": statistics.mean(step_lengths),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
