"""Shipped configs run end to end through the CLI (prepare -> masks -> torchrun training) on CPU,
with a tiny random model standing in for the student. Evaluation needs vLLM + GPU and is not run."""
import json
from pathlib import Path

import pytest

from sgl import cli

ASSETS = Path(__file__).resolve().parent / "assets"


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    import torch
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(str(ASSETS / "tokenizer"))
    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=4096,
                         tie_word_embeddings=False)
    path = tmp_path_factory.mktemp("tiny-student")
    Qwen2ForCausalLM(config).save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


CPU_OVERRIDES = [
    "run.gpus=[]", "stages.train.effective_batch=4", "stages.train.args.deepspeed-config=null",
    "stages.train.args.epochs=1", "stages.train.args.save-strategy=no", "stages.train.args.max-seq-len=4096",
    "stages.train.args.model-dtype=float32",
]


@pytest.mark.regression
@pytest.mark.parametrize("family,config", [
    ("sgl", "r1-qwen-1.5b-palign/sft-nll"),
    ("sgl", "r1-qwen-1.5b-palign/prov-nll-dft"),
    ("palign", "r1-qwen-1.5b"),
])
def test_config_trains_end_to_end(tmp_path, tiny_model, family, config):
    overrides = [f"model.name={tiny_model}", f"dataset={ASSETS / 'palign_sample.jsonl'}", "track=e2e", *CPU_OVERRIDES]
    stages = [s for s in ("prepare", "vanilla", "provenance", "train")
              if s in cli.load_family_config(config, family)["stages"]]
    code = cli.main(["run", config, "--cwd", str(tmp_path), "--stages", ",".join(stages), *overrides], family)
    assert code == 0
    run_name = cli.load_family_config(config, family, overrides)["run_name"]
    summary = json.loads((tmp_path / "checkpoints" / run_name / "run-summary.json").read_text())
    assert summary
    assert (tmp_path / "logs").is_dir()


@pytest.mark.regression
def test_ssft_config_attributes_selects_builds_and_trains(tmp_path, tiny_model):
    """ssft run: IG with the tiny model as attribution model -> select -> build -> train."""
    import shutil

    from sgl.segment.rules import split_segments

    student = tmp_path / "student"
    shutil.copytree(tiny_model, student)
    template = student / "chat_template.jinja"  # plain ChatML, as Qwen2.5-Instruct renders it
    template.write_text(template.read_text().replace("assistant\n<think>\n", "assistant\n"))
    build_dir = tmp_path / "data" / "traces"
    build_dir.mkdir(parents=True)
    with (ASSETS / "palign_sample.jsonl").open() as source, (build_dir / "train.jsonl").open("w") as out:
        for line in list(source)[:4]:
            row = json.loads(line)
            out.write(json.dumps({"question": row["input"], "solution": row["output"], "answer": "1",
                                  "segments": split_segments(row["output"], "paragraph")}) + "\n")
    overrides = [
        f"model.name={student}", f"attribution_model={student}", f"build_dir={build_dir}", "track=e2e",
        "think_prefix=none", "stages.build.args.deepseek=false", "stages.attribution.args.ig-steps=3",
        "stages.attribution.args.device-map=none", *CPU_OVERRIDES,
    ]
    code = cli.main(["run", "r1-qwen-1.5b", "--cwd", str(tmp_path),
                     "--stages", "attribution,select,build,train", *overrides], "ssft")
    assert code == 0
    selected = [json.loads(line) for line in (build_dir / "selected.jsonl").open()]
    assert len(selected) == 4 and all("selected_spans_ids" in row for row in selected)
    assert (tmp_path / "checkpoints" / "ssft-r1-qwen-1.5b" / "run-summary.json").exists()
