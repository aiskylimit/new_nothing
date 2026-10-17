"""Build the offline assets the regression pipeline runs on (run once; outputs are committed).

    python tests/regression/build_assets.py

- assets/tokenizer/: a small byte-level BPE trained on the P-ALIGN training outputs, with an
  R1-style chat template (the generation prompt opens `<think>`), so the chat-template and
  P-ALIGN close-thinking paths are exercised without downloading a real model.
- assets/palign_sample.jsonl: the shortest SAMPLE_SIZE rows of P-ALIGN's training set, in its
  original alpaca format (instruction / input / output with <Begin/End_of_Prefix> markers).
- assets/raw_aime24.jsonl (only when PALIGN_RESULT_DIR points at upstream P-ALIGN's
  data/result/): its published AIME24 generations in evaluate.py's raw format, each output cut
  to its tail (the grader only reads the last \\boxed{}), for the --rescore stage.
- assets/palign_labels.jsonl (same condition): P-ALIGN's published AIME24/AIME25 generations with
  the 0/1 labels its own evaluation.py assigned; texts are cut to their last 1500 characters unless
  the label needs more of the text. tests/test_palign.py checks our grader reproduces every label.

The regression test never rebuilds these: it only needs them to be fixed, not reproducible.
"""
import gzip
import json
import os
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(__file__).resolve().parent / "assets"
SOURCE = ROOT / "references" / "P-ALIGN" / "data" / "palign_sft_qwen2.5-7b.json.gz"
SAMPLE_SIZE = 16
VOCAB_SIZE = 4000
EOS = "<|endoftext|>"
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n<think>\n{% endif %}"
)


def main() -> None:
    source = SOURCE if SOURCE.exists() else ROOT / "P-ALIGN" / "data" / SOURCE.name
    rows = json.load(gzip.open(source))

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=[EOS, "<|im_start|>", "<|im_end|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    texts = [f"{row['instruction']}{row['input']}\n{row['output']}" for row in rows]
    tokenizer.train_from_iterator(texts, trainer=trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token=EOS, pad_token=EOS)
    fast.chat_template = CHAT_TEMPLATE
    fast.save_pretrained(ASSETS / "tokenizer")

    sample = sorted(rows, key=lambda row: len(row["output"]))[:SAMPLE_SIZE]
    with (ASSETS / "palign_sample.jsonl").open("w") as handle:
        for row in sample:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"tokenizer ({len(fast)} tokens) + {len(sample)} samples -> {ASSETS}")

    result_dir = os.environ.get("PALIGN_RESULT_DIR")
    if result_dir:
        write_raw_generations(Path(result_dir) / "aime24-output.jsonl", ASSETS / "raw_aime24.jsonl")
        write_labelled_generations(
            [Path(result_dir) / f"{name}-output.jsonl" for name in ("aime24", "aime25")],
            ASSETS / "palign_labels.jsonl",
        )


def write_labelled_generations(sources: list[Path], output: Path, tail_chars: int = 1500) -> None:
    from sgl.eval.graders.palign import grade_math_verify

    with output.open("w") as out:
        for source in sources:
            for line in source.open():
                row = json.loads(line)
                texts = []
                for text, label in zip(row["output"], row["label"]):
                    tail = text[-tail_chars:]
                    texts.append(tail if grade_math_verify([tail], str(row["answer"]))[0] == int(label) else text)
                out.write(json.dumps({
                    "benchmark": source.stem.replace("-output", ""), "answer": str(row["answer"]),
                    "outputs": texts, "labels": [int(label) for label in row["label"]],
                }, ensure_ascii=False) + "\n")


def write_raw_generations(source: Path, output: Path, tail_chars: int = 600) -> None:
    """P-ALIGN output rows -> evaluate.py raw rows ({id, gold, task_type, generations})."""
    with source.open() as handle, output.open("w") as out:
        for index, line in enumerate(handle):
            row = json.loads(line)
            generations = [
                {"text": text[-tail_chars:], "finish_reason": "stop", "n_tokens": 0}
                for text in row["output"]
            ]
            out.write(json.dumps(
                {"id": index, "gold": str(row["answer"]), "task_type": "math", "generations": generations},
                ensure_ascii=False,
            ) + "\n")


if __name__ == "__main__":
    main()
