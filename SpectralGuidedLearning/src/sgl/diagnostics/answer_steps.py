"""How much of the supervised trace, and of the loss, sits on steps that contain the final answer.

Answer-gain weighting can be read as "only rewarding the steps that restate the answer". This
measures it on the data that is trained on. The answer is the one sgl.signals.answer_gain probes
(last \\boxed{...} of the response), and a step "contains" it under two definitions:

  boxed    the step writes the answer inside \\boxed{} (strict: a final-answer statement)
  mention  the answer string occurs anywhere in the step as a standalone term, boxed included
           (upper bound: a short answer such as "2" also matches unrelated 2s)

Per definition, pooled over response tokens:
  token_share        tokens of those steps / response tokens
  answer_token_share tokens of the answer string itself / response tokens
  weight_share       loss weight on those steps / total loss weight       (needs --weights)
  weight_enrichment  weight_share / token_share; 1.0 = no preference over plain SFT
  gain_share         positive answer gain of those steps / total positive gain (needs --gains)

Writes summary.json, answer-steps.jsonl (every matching step with its text), answer-steps.html
(sampled traces, step by step) and position.png.
"""

import argparse
import html
import json
import random
import re
import statistics
from pathlib import Path

from sgl.eval.graders.answer_scoring import extract_boxed
from sgl.segment.sentence import encode_with_offsets

DEFINITIONS = ("boxed", "mention")
_BOXED_OPEN = re.compile(r"\\boxed\s*\{")


def boxed_spans(text: str, answer: str) -> list[tuple[int, int]]:
    """Character span of the content of every \\boxed{...} holding `answer` (spaces ignored)."""
    spans = []
    for match in _BOXED_OPEN.finditer(text):
        depth, index = 1, match.end()
        while index < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[index], 0)
            index += 1
        content = text[match.end() : index - 1]
        if depth == 0 and content.replace(" ", "") == answer.replace(" ", ""):
            spans.append((match.end(), index - 1))
    return spans


def mention_spans(text: str, answer: str) -> list[tuple[int, int]]:
    """Standalone occurrences of `answer`: not part of a longer word or number (12 is not in 120 or 3.12)."""
    pattern = re.escape(answer)
    if answer[0].isalnum():
        pattern = r"(?<![A-Za-z0-9])(?<!\d[.,])" + pattern
    if answer[-1].isalnum():
        pattern += r"(?![A-Za-z0-9])(?![.,]\d)"
    return [match.span() for match in re.finditer(pattern, text)]


def analyze_record(record: dict, tokenizer, weights: list[float] | None, gains: list[float] | None) -> dict | None:
    """Per-step token counts, answer matches, weights and gains; None without a boxed answer."""
    response = record["response"]
    answer = extract_boxed(response)
    if not answer:
        return None
    start, end = record["response_token_span"]
    ids, token_starts = encode_with_offsets(tokenizer, response)
    if ids != record["input_ids"][start:end]:
        raise ValueError(f"record {record['id']}: --tokenizer does not reproduce the stored response ids")

    bounds = [*token_starts, len(response)]
    boxed = boxed_spans(response, answer)
    matches = {"boxed": boxed, "mention": sorted(set(boxed + mention_spans(response, answer)))}
    tokens_of = {
        name: {i for i in range(len(ids)) for x, y in spans if bounds[i] < y and max(bounds[i + 1], bounds[i] + 1) > x}
        for name, spans in matches.items()
    }

    steps = []
    for index, step in enumerate(record["steps"]):
        first, last = step["token_start"] - start, step["token_end"] - start
        entry = {
            "index": index,
            "text": response[bounds[first] : bounds[last]],
            "tokens": last - first,
            "weight_mass": sum(weights[step["token_start"] : step["token_end"]]) if weights else None,
            "gain": gains[index] if gains else None,
        }
        for name in DEFINITIONS:
            entry[f"{name}_answer_tokens"] = sum(1 for i in tokens_of[name] if first <= i < last)
            entry[name] = entry[f"{name}_answer_tokens"] > 0
        steps.append(entry)
    return {"id": record["id"], "answer": answer, "steps": steps}


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def summarize(samples: list[dict]) -> dict:
    steps = [step for sample in samples for step in sample["steps"]]
    tokens = sum(step["tokens"] for step in steps)
    has_weights = all(step["weight_mass"] is not None for step in steps)
    with_gain = [step for step in steps if step["gain"] is not None]
    result = {"samples": len(samples), "steps": len(steps), "response_tokens": tokens}
    for name in DEFINITIONS:
        hits = [step for step in steps if step[name]]
        hit_tokens = sum(step["tokens"] for step in hits)
        answer_tokens = sum(step[f"{name}_answer_tokens"] for step in hits)
        token_share = _ratio(hit_tokens, tokens)
        entry = {
            "answer_steps": len(hits),
            "step_share": _ratio(len(hits), len(steps)),
            "answer_step_tokens": hit_tokens,
            "answer_tokens": answer_tokens,
            "token_share": token_share,
            "answer_token_share": _ratio(answer_tokens, tokens),
            "answer_tokens_within_answer_steps": _ratio(answer_tokens, hit_tokens),
            "token_share_per_sample_mean": statistics.mean(
                sum(s["tokens"] for s in sample["steps"] if s[name]) / sum(s["tokens"] for s in sample["steps"])
                for sample in samples
            ),
        }
        if has_weights:
            share = _ratio(sum(s["weight_mass"] for s in hits), sum(s["weight_mass"] for s in steps))
            entry["weight_share"] = share
            entry["weight_enrichment"] = _ratio(share, token_share) if share is not None else None
        if with_gain:
            entry["gain_share"] = _ratio(
                sum(max(s["gain"], 0.0) for s in with_gain if s[name]),
                sum(max(s["gain"], 0.0) for s in with_gain),
            )
        result[name] = entry
    return result


def render_html(samples: list[dict]) -> str:
    """Sampled traces: answer-containing steps are outlined, the background shows the mean loss weight."""
    parts = [
        "<!doctype html><meta charset='utf-8'><title>Answer steps</title><style>",
        "body{font:14px/1.45 sans-serif;max-width:70em;margin:auto;background:#fff;color:#111}",
        ".s{white-space:pre-wrap;padding:2px 6px;border-left:4px solid transparent}",
        ".mention{border-left-color:#e69500}.boxed{border-left-color:#c00}small{color:#555}</style>",
        "<p>Red bar: step boxes the answer. Orange bar: step mentions it. Blue shade: loss weight above 1; "
        "grey shade: below 1.</p>",
    ]
    for sample in samples:
        parts.append(f"<h3>sample {sample['id']} — answer: <code>{html.escape(sample['answer'])}</code></h3>")
        for step in sample["steps"]:
            kind = "boxed" if step["boxed"] else "mention" if step["mention"] else ""
            note, style = f"#{step['index']} · {step['tokens']} tok", ""
            if step["weight_mass"] is not None:
                weight = step["weight_mass"] / step["tokens"]
                note += f" · w={weight:.2f}"
                colour = "0,90,200" if weight > 1 else "110,110,110"
                style = f" style='background:rgba({colour},{min(abs(weight - 1) * 0.35, 0.35):.2f})'"
            if step["gain"] is not None:
                note += f" · gain={step['gain']:+.3f}"
            parts.append(f"<div class='s {kind}'{style}><small>{note}</small> {html.escape(step['text'])}</div>")
    return "\n".join(parts)


def plot_position(samples: list[dict], path: Path) -> None:
    """Where in the trace the answer-containing steps sit (share of each position decile's tokens)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    total = [0] * 10
    hits = {name: [0] * 10 for name in DEFINITIONS}
    for sample in samples:
        n = len(sample["steps"])
        for step in sample["steps"]:
            decile = min(10 * step["index"] // n, 9)
            total[decile] += step["tokens"]
            for name in DEFINITIONS:
                hits[name][decile] += step["tokens"] if step[name] else 0
    figure, axis = plt.subplots(figsize=(7, 3.5))
    for offset, name in zip((-0.2, 0.2), DEFINITIONS):
        axis.bar([i + offset for i in range(10)], [h / t if t else 0 for h, t in zip(hits[name], total)],
                 width=0.4, label=name)
    axis.set_xticks(range(10), [f"{i * 10}-{i * 10 + 10}%" for i in range(10)], rotation=45)
    axis.set_xlabel("step position in the trace")
    axis.set_ylabel("share of tokens in answer steps")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="train-segmented.jsonl")
    parser.add_argument("--tokenizer", required=True, help="tokenizer the data was prepared with")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--weights", help="train-<variant>.jsonl with loss_weights (sgl.allocation.build)")
    parser.add_argument("--gains", help="step_answer_gain.json from sgl.signals.answer_gain")
    parser.add_argument("--examples", type=int, default=20, help="traces shown in answer-steps.html")
    parser.add_argument("--seed", type=int, default=42, help="which traces are shown")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    weights = {}
    if args.weights:
        with open(args.weights) as handle:
            for line in handle:
                row = json.loads(line)
                weights[row["id"]] = row["loss_weights"]
    gains = json.loads(Path(args.gains).read_text()) if args.gains else {}

    samples, skipped = [], 0
    with open(args.data_path) as handle:
        for line in handle:
            record = json.loads(line)
            if args.weights and len(weights.get(record["id"], ())) != len(record["input_ids"]):
                raise ValueError(f"record {record['id']}: --weights does not match --data-path")
            sample = analyze_record(record, tokenizer, weights.get(record["id"]), gains.get(str(record["id"])))
            if sample is None:
                skipped += 1
            else:
                samples.append(sample)
    if not samples:
        raise ValueError("no sample has a \\boxed{} answer")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = {**summarize(samples), "skipped_without_boxed": skipped}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    with (output / "answer-steps.jsonl").open("w") as handle:
        for sample in samples:
            for step in sample["steps"]:
                if step["mention"]:
                    handle.write(json.dumps({"id": sample["id"], "answer": sample["answer"], **step}) + "\n")
    shown = random.Random(args.seed).sample(samples, min(args.examples, len(samples)))
    (output / "answer-steps.html").write_text(render_html(sorted(shown, key=lambda s: s["id"])))
    plot_position(samples, output / "position.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
