"""End-to-end regression harness: run every CPU stage of the pipeline on fixed assets and hash
what it writes.

    python tests/regression/pipeline.py            # compare against golden.json
    python tests/regression/pipeline.py --update   # rewrite golden.json (only on purpose!)

golden.json was produced by the pre-refactor code (tag paper-v1), so a refactor that changes any
dataset byte -- token ids, masks, weights, stats -- fails here. Each stage runs as a subprocess
from inside the work dir with relative paths, so paths written into outputs (e.g. the config echo
in iwc-selection-stats.json) are stable. Parquet files are hashed by content, not bytes, because
pyarrow embeds its own version in the file.

Only the command lines in STAGES may change when modules move; the hashes may not.
"""
import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
GOLDEN = HERE / "golden.json"
TOKENIZER = str(ASSETS / "tokenizer")
SSFT = ROOT / "references" / "SegmentSelectiveSFT"
if not SSFT.exists():
    SSFT = ROOT / "SegmentSelectiveSFT"

PY = sys.executable


def sgl(module: str) -> list[str]:
    """Command prefix for one sgl pipeline module."""
    return [PY, "-m", f"sgl.{module}"]


PREP_ARGS = [
    "--dataset-name", str(ASSETS / "palign_sample.jsonl"), "--tokenizer", TOKENIZER,
    "--max-tokens", "100000", "--seed", "42",
]

# (stage name, argv, stdout capture file or None). Run in order, cwd = work dir.
STAGES = [
    ("data_prep", sgl("data.prepare") + PREP_ARGS + [
        "--chat-template", "--no-enable-thinking", "--palign-prompt",
        "--output-path", "data/train-segmented.jsonl",
    ], None),
    ("data_prep_plain", sgl("data.prepare") + PREP_ARGS + [
        "--no-chat-template", "--output-path", "plain/train-segmented.jsonl",
    ], None),
    ("signals", None, None),  # synthetic spectral/entropy/gain signals, see write_signals()
    ("build_masks", sgl("selection.build_masks") + [
        "--data-path", "data/train-segmented.jsonl", "--strengths", "signals/spectral.parquet",
        "--energy-threshold-p", "0.8", "--vanilla",
    ], None),
    ("build_masks_sweep", sgl("selection.build_masks") + [
        "--data-path", "data/train-segmented.jsonl", "--strengths", "signals/spectral.parquet",
        "--sweep", "0.5,0.8,0.95",
    ], None),
    ("build_iwc", sgl("allocation.build") + [
        "--data-path", "data/train-segmented.jsonl", "--strengths", "signals/spectral.parquet",
        "--energy-threshold-p", "0.8", "--interpolation", "0.5",
        "--variants", "iwc,iwc-stable,iwc-stable-lambda0,iwc-stable-shuffled,iwc-stable-reverse",
    ], None),
    ("build_gain_signal", sgl("signals.gain_parquet") + [
        "--data-path", "gain/train-segmented.jsonl", "--gains", "signals/gains.json",
        "--output", "gain/signal-answer-gain.parquet",
    ], None),
    ("build_iwc_gain", sgl("allocation.build") + [
        "--data-path", "gain/train-segmented.jsonl", "--strengths", "gain/signal-answer-gain.parquet",
        "--energy-threshold-p", "1.0", "--interpolation", "0.5", "--variants", "iwc-stable",
    ], None),
    ("build_provenance", sgl("transforms.provenance") + [
        "--data-dir", "data", "--tokenizer", TOKENIZER,
    ], None),
    ("build_provenance_junction", sgl("transforms.provenance") + [
        "--data-dir", "data", "--tokenizer", TOKENIZER, "--junction-tokens", "8",
        "--output-name", "train-provenance-j8.jsonl",
    ], None),
    ("build_trans", sgl("transforms.build_transitions") + [
        "--segmented", "data/train-segmented.jsonl", "--masked", "data/train-spectral.jsonl",
        "--output", "data/train-spectral-trans.jsonl", "--tokenizer", TOKENIZER,
    ], None),
    ("iwc_diagnostics", sgl("diagnostics.iwc") + [
        "--data-path", "data/train-segmented.jsonl", "--strengths", "signals/spectral.parquet",
        "--output-dir", "diagnostics", "--energy-threshold-p", "0.8", "--tokenizer", TOKENIZER,
    ], None),
    ("rescore", sgl("eval.evaluate") + [
        "--rescore", str(ASSETS / "raw_aime24.jsonl"), "--grader", "math_verify",
    ], "eval/rescore-math_verify.json"),
    ("rescore_builtin", sgl("eval.evaluate") + [
        "--rescore", str(ASSETS / "raw_aime24.jsonl"), "--grader", "builtin",
    ], "eval/rescore-builtin.json"),
    ("compare_results", sgl("eval.compare") + ["--results-dir", "results"], None),
    ("ssft_split_paragraph", [PY, str(SSFT / "Attribution" / "segment_split.py"),
        "--input_data_file", "ssft/train.jsonl", "--output_data_file", "ssft/segments-paragraph.jsonl",
        "--tokenizer", "none", "--segment_mode", "paragraph",
    ], None),
    ("ssft_split_cue", [PY, str(SSFT / "Attribution" / "segment_split.py"),
        "--input_data_file", "ssft/train.jsonl", "--output_data_file", "ssft/segments-cue.jsonl",
        "--tokenizer", "none", "--segment_mode", "cue",
    ], None),
    ("ssft_ig", None, None),  # synthetic compact IG per segment, see write_ssft_ig()
    ("ssft_select", [PY, str(SSFT / "Attribution" / "get_important_segments.py"),
        "--input_data_file", "ssft/segments-paragraph.jsonl", "--IG_score_data_file", "ssft/ig_compact.jsonl",
        "--output_data_file", "ssft/selected.jsonl",
    ], None),
    # framework stages added after paper-v1: new outputs only, the ones above must not move
    ("data_prep_paragraph", sgl("data.prepare") + PREP_ARGS + [
        "--chat-template", "--no-enable-thinking", "--palign-prompt", "--segmenter", "paragraph",
        "--output-path", "paragraph/train-segmented.jsonl",
    ], None),
    ("sgl_ssft_select", sgl("ssft.select") + [
        "--data-path", "ssft/segments-paragraph.jsonl", "--ig", "ssft/ig_compact.jsonl",
        "--output", "ssft/selected-sgl.jsonl",
    ], None),
    ("sgl_ssft_build", sgl("ssft.build") + [
        "--data-path", "ssft/selected-sgl.jsonl", "--tokenizer", "tokenizer-chatml",
        "--output", "ssft/train-ssft.jsonl",
        "--no-deepseek", "--think-prefix", "plain", "--prompt-style", "palign",
    ], None),
]

# Pairs of outputs that must be byte-identical: framework port vs the upstream script.
SAME_OUTPUT = [("ssft/selected.jsonl", "ssft/selected-sgl.jsonl")]

# Outputs that are figures or otherwise not byte-stable; everything else in the work dir is hashed.
UNHASHED_SUFFIXES = {".png"}


def write_inputs(work: Path) -> None:
    """Inputs that are copies of repo files rather than stage outputs."""
    results = work / "results"
    for summary in sorted((ROOT / "results").glob("*/summary.json")):
        target = results / summary.parent.name / "summary.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(summary, target)

    # Segment-Selective SFT probes for a plain ChatML generation prompt ("<|im_start|>assistant\n");
    # the asset tokenizer opens <think> after it, so give that stage a plain-ChatML copy.
    chatml = work / "tokenizer-chatml"
    shutil.copytree(TOKENIZER, chatml)
    template = chatml / "chat_template.jinja"
    template.write_text(template.read_text().replace("assistant\n<think>\n", "assistant\n"))

    # SegmentSelectiveSFT reads question / solution / answer rows (its prepare_s1k.py format).
    (work / "ssft").mkdir()
    with (ASSETS / "palign_sample.jsonl").open() as handle, (work / "ssft" / "train.jsonl").open("w") as out:
        for line in handle:
            row = json.loads(line)
            solution = row["output"]
            answer = solution.rsplit("\\boxed{", 1)[-1].split("}", 1)[0]
            out.write(json.dumps({"question": row["input"], "solution": solution, "answer": answer}) + "\n")


def write_signals(work: Path) -> None:
    """Deterministic per-step spectral strengths, entropies and answer gains."""
    import pandas as pd

    records = [json.loads(line) for line in (work / "data" / "train-segmented.jsonl").open()]
    rows, gains = [], {}
    for record in records:
        rng = random.Random(1000 + record["id"])
        n_steps = len(record["steps"])
        rows.append({
            "id": record["id"],
            "k_star": rng.randint(1, 8),
            "n_response_tokens": record["response_token_span"][1] - record["response_token_span"][0],
            "n_steps": n_steps,
            "step_strengths": [round(rng.uniform(0.0, 1.0), 6) for _ in range(n_steps)],
            "step_entropies": [round(rng.uniform(0.0, 3.0), 6) for _ in range(n_steps)],
        })
        if record["id"] % 5 != 4:  # leave some records without a gain signal on purpose
            gains[str(record["id"])] = [round(rng.gauss(0.0, 1.0), 6) for _ in range(n_steps)]
    (work / "signals").mkdir()
    pd.DataFrame(rows).to_parquet(work / "signals" / "spectral.parquet")
    (work / "signals" / "gains.json").write_text(json.dumps(gains))
    (work / "gain").mkdir()
    shutil.copyfile(work / "data" / "train-segmented.jsonl", work / "gain" / "train-segmented.jsonl")


def write_ssft_ig(work: Path) -> None:
    """Deterministic compact IG rows [n_tok, sum|IG|, sum IG] per segment, with edge cases."""
    with (work / "ssft" / "segments-paragraph.jsonl").open() as handle, \
         (work / "ssft" / "ig_compact.jsonl").open("w") as out:
        for index, line in enumerate(handle):
            segments = json.loads(line)["segments"]
            rng = random.Random(2000 + index)
            compact = []
            for segment in segments:
                n_tok = max(1, len(segment) // 4)
                sum_abs = round(rng.uniform(0.0, 1.0), 6)
                compact.append([n_tok, sum_abs, round(sum_abs * rng.uniform(-1.0, 1.0), 6)])
            if index == 0:  # no attribution signal at all -> empty selection
                compact = [[n, 0.0, 0.0] for n, _, _ in compact]
            out.write(json.dumps({"segments": compact}) + "\n")


SYNTHETIC = {"signals": write_signals, "ssft_ig": write_ssft_ig}


def content_hash(path: Path) -> str:
    if path.suffix == ".parquet":
        import pandas as pd

        frame = pd.read_parquet(path)
        payload = frame.to_json(orient="records", double_precision=15).encode()
    else:
        payload = path.read_bytes()
    return hashlib.sha256(payload).hexdigest()


def run_pipeline(work: Path) -> dict[str, str]:
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), TOKENIZERS_PARALLELISM="false",
               HF_HUB_OFFLINE="1", MPLBACKEND="Agg")
    write_inputs(work)
    for name, argv, capture in STAGES:
        if argv is None:
            SYNTHETIC[name](work)
            continue
        result = subprocess.run(argv, cwd=work, env=env, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"stage {name} failed:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}")
        if capture:
            target = work / capture
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(result.stdout)
    return {
        str(path.relative_to(work)): content_hash(path)
        for path in sorted(work.rglob("*"))
        if path.is_file() and path.suffix not in UNHASHED_SUFFIXES
    }


def diff(expected: dict[str, str], actual: dict[str, str]) -> list[str]:
    problems = [f"port differs from upstream: {a} vs {b}" for a, b in SAME_OUTPUT if actual.get(a) != actual.get(b)]
    problems += [f"missing output: {name}" for name in sorted(expected.keys() - actual.keys())]
    problems += [f"unexpected output: {name}" for name in sorted(actual.keys() - expected.keys())]
    problems += [
        f"changed: {name}" for name in sorted(expected.keys() & actual.keys())
        if expected[name] != actual[name]
    ]
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true", help="rewrite golden.json from this run")
    parser.add_argument("--keep", help="run in this directory and keep it, for debugging")
    args = parser.parse_args()

    if args.keep:
        work = Path(args.keep)
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        hashes = run_pipeline(work)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            hashes = run_pipeline(Path(tmp))

    if args.update:
        GOLDEN.write_text(json.dumps(hashes, indent=1, sort_keys=True) + "\n")
        print(f"wrote {len(hashes)} hashes -> {GOLDEN}")
        return
    problems = diff(json.loads(GOLDEN.read_text()), hashes)
    print("\n".join(problems) if problems else f"all {len(hashes)} outputs identical")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
