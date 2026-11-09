"""Real three-process CPU/Gloo smoke test using a tiny local Qwen model.

Does not validate CUDA/NCCL/B200 performance. Does validate that all workers
finish, synchronize every expert, handle partial batches, and resume correctly.
"""
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cored"))


def tiny_assets(root):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    from core.data_utils import StepDataset
    from train.sft_engine import build_segments

    torch.manual_seed(123)
    model = root / "tiny"
    vocab = {token: i for i, token in enumerate(
        ["[UNK]", "[PAD]", "[EOS]", "<think>", "</think>", "Q", "A", "first", "second", "third", "assistant"])}
    raw = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
    tok.chat_template = "{% for message in messages %}{{ message['content'] + '\\n' }}{% endfor %}{% if add_generation_prompt %}assistant\n{% endif %}"
    tok.save_pretrained(model)
    cfg = Qwen2Config(vocab_size=len(vocab), hidden_size=16, intermediate_size=32,
                      num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                      max_position_embeddings=256, attention_dropout=0.0,
                      pad_token_id=1, eos_token_id=2, bos_token_id=2)
    Qwen2ForCausalLM(cfg).save_pretrained(model)
    source = root / "train.jsonl"
    rows = [{"id": f"s{i}", "question": "Q " * (i + 1),
             "thinking": "first\n\nsecond" + ("\n\nthird" if i % 2 else ""),
             "answer_text": "A", "answer": "A"} for i in range(7)]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    keys, weights = root / "keys", root / "weights"
    keys.mkdir()
    weights.mkdir()
    dataset = StepDataset(str(source), tok, cutoff_len=128)
    for i in range(len(dataset)):
        item = dataset[i]
        np.savez(keys / f"{item.example_id}.npz", key_mask=np.ones(len(build_segments(item)), dtype=bool))
        np.savez(weights / f"{item.example_id}.npz", w=np.linspace(0.5, 1.5, len(item.step_spans), dtype=np.float32))
    return model, source, keys, weights


def launch(root, world, stage, output, model, source, extra):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "cored"), "PYTHONDONTWRITEBYTECODE": "1",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    prefix = [sys.executable]
    if world > 1:
        with socket.socket() as port:
            port.bind(("127.0.0.1", 0))
            free_port = port.getsockname()[1]
        env["GLOO_SOCKET_IFNAME"] = "lo0" if sys.platform == "darwin" else "lo"
        prefix += ["-m", "torch.distributed.run", "--nnodes=1", f"--nproc_per_node={world}",
                   "--master_addr=127.0.0.1", f"--master_port={free_port}"]
    command = prefix + ["-m", "train.b200", "--stage", stage, "--model_name", str(model),
        "--train_file", str(source), "--out_dir", str(output), "--global_batch", "5", "--epochs", "2",
        "--micro_batch_size", "2", "--max_batch_tokens", "256", "--loss_chunk", "4", "--cutoff_len", "128",
        "--lora_r", "2", "--lora_alpha", "2", "--lora_dropout", "0", "--p_drop", "0",
        "--dpp_max_tokens", "4", "--cpu_float32", "--log_every", "1", *map(str, extra)]
    log = root / f"{output.name}-{stage}-{world}.log"
    with log.open("a") as handle:
        result = subprocess.run(command, cwd=ROOT / "cored", env=env,
                                stdout=handle, stderr=subprocess.STDOUT, timeout=120)
    assert result.returncode == 0, log.read_text()[-18000:]


def compare_adapters(left, right, names, atol=2e-5):
    for name in names:
        a = load_file(str(left / name / "adapter_model.safetensors"))
        b = load_file(str(right / name / "adapter_model.safetensors"))
        assert a.keys() == b.keys()
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=atol, rtol=3e-4)


def test_three_rank_council_distillation_and_resume_match_single_rank(tmp_path):
    from proposal.merge import merge_experts, save_merged_adapter

    model, source, keys, weights = tiny_assets(tmp_path)
    one, three, resumed = [tmp_path / name for name in ("one", "three", "resumed")]
    extra = ["--keystep_dir", keys, "--lr", "0.003"]
    launch(tmp_path, 1, "phase1", one, model, source, extra)
    launch(tmp_path, 3, "phase1", three, model, source, extra)
    compare_adapters(one, three, [f"expert_{i}" for i in range(3)])
    assert json.loads((three / "complete.json").read_text())["optimizer_steps"] == 4
    assert len(list(three.glob("timing-rank-*.json"))) == 3
    # Interrupt after an optimizer update, then recover optimizer, LR and per-rank RNG.
    launch(tmp_path, 3, "phase1", resumed, model, source, extra + ["--dry_run", "1"])
    assert not (resumed / "complete.json").exists()
    launch(tmp_path, 3, "phase1", resumed, model, source, extra + ["--resume"])
    compare_adapters(three, resumed, [f"expert_{i}" for i in range(3)], atol=1e-7)
    # Check RNG recovery with actual LoRA/segment dropout, not only deterministic losses.
    stochastic = tmp_path / "stochastic"
    stochastic_resumed = tmp_path / "stochastic-resumed"
    stochastic_options = extra + ["--lora_dropout", "0.05", "--p_drop", "0.1"]
    launch(tmp_path, 3, "phase1", stochastic, model, source, stochastic_options)
    launch(tmp_path, 3, "phase1", stochastic_resumed, model, source,
           stochastic_options + ["--dry_run", "1"])
    launch(tmp_path, 3, "phase1", stochastic_resumed, model, source,
           stochastic_options + ["--resume"])
    compare_adapters(stochastic, stochastic_resumed, [f"expert_{i}" for i in range(3)], atol=1e-7)
    state, rank = merge_experts([str(three / f"expert_{i}") for i in range(3)], "ties",
                                out_rank=2, density=0.2, device="cpu")
    teacher = save_merged_adapter(state, rank, str(three / "expert_0"), str(tmp_path / "merge"))
    dist_one, dist_three = tmp_path / "dist-one", tmp_path / "dist-three"
    options = ["--teacher_adapter", teacher, "--weights_dir", weights, "--lr", "0.003"]
    launch(tmp_path, 1, "distill", dist_one, model, source, options)
    launch(tmp_path, 3, "distill", dist_three, model, source, options)
    compare_adapters(dist_one, dist_three, ["student"])
    records = [json.loads(row) for row in (dist_three / "train_log.jsonl").read_text().splitlines()]
    assert any(row["kl"] > 0 for row in records[1:])


def test_pipeline_preparation_reuses_data_but_rejects_changed_recipe(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    model, _, _, _ = tiny_assets(tmp_path)
    dataset = tmp_path / "snapshot/data"
    dataset.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{"question": "Q", "deepseek_thinking_trajectory": "first\n\nsecond",
                    "deepseek_attempt": "A", "solution": "A"}]), dataset / "train-00000.parquet")
    command = [sys.executable, "-m", "train.pipeline_b200", "--model", str(model),
               "--dataset", str(dataset.parent), "--output_root", str(tmp_path / "pipeline"),
               "--run_id", "test-run", "--gpus", "0,1,2", "--stages", "prepare"]
    env = {**os.environ, "PYTHONPATH": str(ROOT / "cored"), "PYTHONDONTWRITEBYTECODE": "1",
           "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}
    first = subprocess.run(command, cwd=ROOT / "cored", env=env, capture_output=True, text=True, timeout=30)
    assert first.returncode == 0, first.stderr
    output = tmp_path / "pipeline/test-run"
    assert len((output / "train_s1k.jsonl").read_text().splitlines()) == 1
    repeated = subprocess.run(command + ["--micro_batch_size", "1", "--max_batch_tokens", "64"],
                              cwd=ROOT / "cored", env=env, capture_output=True, text=True, timeout=30)
    assert repeated.returncode == 0, repeated.stderr
    assert "reuse prepare" in repeated.stdout
    assert len((output / "timings.jsonl").read_text().splitlines()) == 1
    changed = subprocess.run(command + ["--seed", "43"], cwd=ROOT / "cored", env=env,
                             capture_output=True, text=True, timeout=30)
    assert changed.returncode != 0
    assert "Run configuration differs" in changed.stderr


def test_two_rank_three_experts_and_student_match_single_rank(tmp_path):
    """Real two-process CPU/Gloo coverage for the development topology."""
    from proposal.merge import merge_experts, save_merged_adapter
    model, source, keys, weights = tiny_assets(tmp_path)
    one, two = tmp_path / 'one', tmp_path / 'two'
    extra = ['--keystep_dir', keys, '--lr', '0.003']
    launch(tmp_path, 1, 'phase1', one, model, source, extra)
    launch(tmp_path, 2, 'phase1', two, model, source, extra)
    compare_adapters(one, two, [f'expert_{i}' for i in range(3)])
    assert len(list(two.glob('timing-rank-*.json'))) == 2
    state, rank = merge_experts([str(two / f'expert_{i}') for i in range(3)], 'ties',
                                out_rank=2, density=0.2, device='cpu')
    teacher = save_merged_adapter(state, rank, str(two / 'expert_0'), str(tmp_path/'merged'))
    options = ['--teacher_adapter', teacher, '--weights_dir', weights, '--lr', '0.003']
    dist_one, dist_two = tmp_path/'dist-one', tmp_path/'dist-two'
    launch(tmp_path, 1, 'distill', dist_one, model, source, options)
    launch(tmp_path, 2, 'distill', dist_two, model, source, options)
    compare_adapters(dist_one, dist_two, ['student'])
    assert (dist_two/'student/adapter_model.safetensors').is_file()
