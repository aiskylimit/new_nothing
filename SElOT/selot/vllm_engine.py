"""Run scripts/vllm_worker.py in a subprocess: greedy generation with vLLM for text-only steps.

Used by the label stage (fit_data.engine=vllm: responses of the unsteered model; Llama Guard verdicts) and the judge
stage (Llama Guard, HarmBench and the over-refusal judge). One subprocess per call, so the GPU memory is fully released when
it exits and the next model can load. Inputs are either chat-formatted texts (tokenized in the worker without special
tokens) or token ids built by the caller with the judge's own HF tokenizer, so vLLM sees exactly the tokens the HF path
would.

Data parallel: when CUDA_VISIBLE_DEVICES lists several GPUs, the prompts are split into one contiguous shard per GPU
and one worker runs on each (CUDA_VISIBLE_DEVICES=<that GPU>), all at once; outputs are merged by id. Greedy
decoding makes the result independent of the split. SELOT_VLLM_WORKER replaces the worker script (tests).
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ENGINES = ("hf", "vllm")
WORKER = Path(__file__).resolve().parents[1] / "scripts" / "vllm_worker.py"


def devices() -> List[str]:
    """the GPUs of CUDA_VISIBLE_DEVICES (one worker each), [] when it is not set"""
    v = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    return [d.strip() for d in v.split(",") if d.strip()]


def check_engine(engine: str, key: str) -> str:
    if engine not in ENGINES:
        raise ValueError(f"{key} must be one of {ENGINES}, not {engine!r}")
    return engine


def vllm_generate(model_path: str, ids: Sequence[str], max_tokens: int, dtype: str = "bfloat16",
                  gpu_mem: float = 0.85, texts: Optional[List[str]] = None,
                  token_ids: Optional[List[List[int]]] = None, log=None) -> Dict[str, str]:
    """{id: generated text}; pass texts (chat template applied) or token_ids."""
    if (texts is None) == (token_ids is None):
        raise ValueError("pass exactly one of texts / token_ids")
    items = texts if texts is not None else token_ids
    key = "text" if texts is not None else "ids"
    worker = os.environ.get("SELOT_VLLM_WORKER") or str(WORKER)
    devs = devices()
    n = max(1, min(len(devs), len(items)))
    bounds = [round(i * len(items) / n) for i in range(n + 1)]
    with tempfile.TemporaryDirectory() as d:
        procs = []
        for k in range(n):
            inp, out = Path(d) / f"in{k}.jsonl", Path(d) / f"out{k}.jsonl"
            part = range(bounds[k], bounds[k + 1])
            inp.write_text("".join(json.dumps({"id": str(ids[i]), key: items[i]}, ensure_ascii=False) + "\n"
                                   for i in part), encoding="utf-8")
            cmd = [sys.executable, worker, "--model", model_path, "--inp", str(inp), "--out", str(out),
                   "--max-tokens", str(max_tokens), "--dtype", dtype, "--gpu-mem", str(gpu_mem)]
            env = dict(os.environ)
            if n > 1:
                env["CUDA_VISIBLE_DEVICES"] = devs[k]
            if log:
                log.info("vLLM: %d prompts, %d new tokens, %s%s", len(part), max_tokens, model_path,
                         f" (GPU {devs[k]}, shard {k + 1}/{n})" if n > 1 else "")
            procs.append((subprocess.Popen(cmd, env=env), out))
        codes = [p.wait() for p, _ in procs]
        if any(codes):
            raise subprocess.CalledProcessError(next(c for c in codes if c), worker)
        res = {}
        for _, out in procs:
            for l in out.read_text(encoding="utf-8").split("\n"):
                if l.strip():
                    r = json.loads(l)
                    res[r["id"]] = r["output"]
    return res
