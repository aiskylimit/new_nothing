#!/usr/bin/env python3
"""vLLM eval worker: one model on the GPU given by CUDA_VISIBLE_DEVICES.

--role generate: greedy answers of a policy model for prepared items.
    mode "mc"  -> one token restricted to the option letters (no free-text parsing),
    mode "gen" -> greedy generation up to the item's max_tokens.
--role guard: allenai/wildguard labels of (prompt, response) pairs with the model card's input format
    (harmful request / response refusal / harmful response).
Output is appended per batch, so an interrupted run resumes where it stopped.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LETTERS = "ABCDEFGHIJKLMNOP"
READY_MARKER = "[EVAL_WORKER_READY]"
FINISHED_MARKER = "[EVAL_WORKER_FINISHED]"
T0 = time.time()


def tick(label):
    print(f"[{time.time() - T0:8.1f}s] {label}", flush=True)


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def done_uids(path):
    done = set()
    if Path(path).exists():
        for line in open(path, encoding="utf-8"):
            try:
                done.add(json.loads(line)["uid"])
            except (ValueError, KeyError):
                pass  # a truncated last line from a killed run is regenerated
    return done


def prompt_text(tokenizer, item):
    """Exact text sent to vLLM for a generate item."""
    # raw_prompt: the benchmark's official prompt is already a full chat string (LiveCodeBench).
    return item["prompt"] if item.get("raw_prompt") else tokenizer.apply_chat_template(
        [{"role": "user", "content": item["prompt"]}], tokenize=False, add_generation_prompt=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role", choices=["generate", "guard"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--items", required=True, help="generate: prepared items; guard: generations to judge")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="generate: minimum, raised to fit the longest prompt + its token budget (up to the "
                             "model's context window); guard: used as is")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-mem", type=float, default=0.90)
    parser.add_argument("--guard-max-tokens", type=int, default=32, help="WildGuard answers in three short lines")
    parser.add_argument("--max-num-seqs", type=int, default=512, help="Concurrent sequences inside vLLM")
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--attention-backend", default="FLASH_ATTN",
                        help="vLLM attention backend. FLASH_ATTN ships prebuilt in the vLLM wheel (as P-ALIGN uses on "
                             "this server); vLLM's automatic choice on B200 is FlashInfer, which JIT-compiles kernels "
                             "with nvcc at startup. GPUs below sm_80 (e.g. Kaggle T4) fall back to TRITON_ATTN.")
    args = parser.parse_args()

    items = read_jsonl(args.items)
    done = done_uids(args.out)
    pending = [item for item in items if item["uid"] not in done]
    tick(f"[{args.label}/{args.role}] {len(done)} done, {len(pending)} pending")
    if not pending:
        return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.platforms import current_platform

    backend = args.attention_backend
    capability = current_platform.get_device_capability()  # NVML query, does not initialize CUDA here
    if backend.upper() == "FLASH_ATTN" and capability is not None and (capability.major, capability.minor) < (8, 0):
        tick(f"[{args.label}/{args.role}] FLASH_ATTN needs sm_80+, GPU is sm_{capability.major}{capability.minor}: "
             "using TRITON_ATTN")
        backend = "TRITON_ATTN"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    max_model_len = args.max_model_len
    if args.role == "generate":
        # Size the context so the longest prompt still gets its full budget (1 letter for mc, max_tokens for gen):
        # vLLM rejects a whole batch if one prompt does not fit. Measured over all items, not only the pending
        # ones, so a resumed run uses the same length.
        from transformers import AutoConfig
        window = getattr(AutoConfig.from_pretrained(args.model), "max_position_embeddings", None) or max_model_len
        ids = tokenizer([prompt_text(tokenizer, item) for item in items], add_special_tokens=False)["input_ids"]
        need = max(len(row) + (1 if item["mode"] == "mc" else item["max_tokens"]) for row, item in zip(ids, items))
        max_model_len = min(window, max(max_model_len, -(-need // 256) * 256))
        tick(f"[{args.label}/{args.role}] longest prompt + budget = {need} tokens -> max_model_len {max_model_len} "
             f"(context window {window})")
    # FlashAttention kernels are built/loaded during engine startup (profiling + CUDA graph capture), so a
    # backend problem surfaces here. TRITON_ATTN is the fallback: it is what the FP32 rollout runs on.
    backends = [backend] + (["TRITON_ATTN"] if backend.upper() != "TRITON_ATTN" else [])
    for attempt, backend in enumerate(backends):
        tick(f"[{args.label}/{args.role}] attention backend {backend}")
        try:
            llm = LLM(model=args.model, tokenizer=args.model, dtype=args.dtype, tensor_parallel_size=1,
                      gpu_memory_utilization=args.gpu_mem, max_model_len=max_model_len, seed=0,
                      max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.max_num_batched_tokens,
                      attention_backend=backend)
            break
        except Exception as error:  # noqa: BLE001 - retried with the next backend, re-raised if none is left
            if attempt + 1 == len(backends):
                raise
            tick(f"[{args.label}/{args.role}] {backend} engine failed to start ({type(error).__name__}: "
                 f"{str(error).splitlines()[0][:200]}); retrying with {backends[attempt + 1]}")
    # The orchestrator holds a per-GPU startup lock until this line, so concurrent
    # engines never profile memory at the same time.
    print(READY_MARKER, flush=True)
    letter_ids = []
    if args.role == "generate":
        for letter in LETTERS:
            ids = tokenizer.encode(letter, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(f"letter {letter!r} is not a single token: {ids}")
            letter_ids.append(ids[0])
        # MC items are cheap; long generations batch better together.
        pending.sort(key=lambda item: (item["mode"] != "mc", item.get("max_tokens", 1)))
    else:
        from scripts.eval_lib import wildguard_input

        def encode(text):
            return tokenizer(text, add_special_tokens=False)["input_ids"]

    started, finished = time.time(), 0
    with open(args.out, "a", encoding="utf-8") as handle:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start + args.batch_size]
            prompts, params, keep = [], [], []
            for item in batch:
                if args.role == "generate":
                    text = prompt_text(tokenizer, item)
                    n_prompt = len(tokenizer(text, add_special_tokens=False)["input_ids"])
                    if item["mode"] == "mc":
                        if n_prompt + 1 > max_model_len:
                            # Only beyond the model's context window: recorded as unanswered (scored wrong for
                            # every model, so paired deltas hold) instead of failing the whole batch.
                            handle.write(json.dumps({"uid": item["uid"], "response": "", "choice": None,
                                                     "finish_reason": "prompt_too_long", "n_tokens": 0}) + "\n")
                            continue
                        sampling = SamplingParams(temperature=0.0, max_tokens=1,
                                                  allowed_token_ids=letter_ids[: item["n_options"]])
                    else:
                        budget = min(item["max_tokens"], max_model_len - n_prompt)
                        if budget < 16:
                            handle.write(json.dumps({"uid": item["uid"], "response": "", "choice": None,
                                                     "finish_reason": "prompt_too_long", "n_tokens": 0}) + "\n")
                            continue
                        sampling = SamplingParams(temperature=0.0, max_tokens=budget)
                else:
                    # The format already starts with "<s>", so it is tokenized here without special tokens and
                    # sent as token ids (a text prompt would get a second BOS from vLLM).
                    ids = encode(wildguard_input(item["prompt"], item["response"]))
                    excess = len(ids) + args.guard_max_tokens - max_model_len
                    if excess > 0:
                        # Cut the end of the response, so the instruction and the answer slot stay intact
                        # (8 tokens of slack for re-tokenization at the cut).
                        response = encode(item["response"])
                        response = tokenizer.decode(response[: max(0, len(response) - excess - 8)])
                        ids = encode(wildguard_input(item["prompt"], response))
                    text = {"prompt_token_ids": ids[: max_model_len - args.guard_max_tokens]}
                    sampling = SamplingParams(temperature=0.0, max_tokens=args.guard_max_tokens)
                prompts.append(text)
                params.append(sampling)
                keep.append(item)
            outputs = llm.generate(prompts, sampling_params=params, use_tqdm=False) if prompts else []
            for item, output in zip(keep, outputs, strict=True):
                completion = output.outputs[0]
                text = completion.text.strip()
                if args.role == "generate":
                    record = {"uid": item["uid"], "response": text,
                              "choice": text[:1] if item["mode"] == "mc" else None,
                              "finish_reason": completion.finish_reason, "n_tokens": len(completion.token_ids)}
                else:
                    record = {"uid": item["uid"], "guard_output": text}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            finished += len(batch)
            rate = finished / max(1e-3, time.time() - started)
            tick(f"[{args.label}/{args.role}] {len(done) + finished}/{len(items)} | {rate:.1f} item/s")
    tick(f"[{args.label}/{args.role}] finished -> {args.out}")
    if os.environ.get("GRIT_EVAL_HANDSHAKE") == "1":
        # vLLM frees this engine's GPU memory at exit. If another engine on the same GPU is profiling memory
        # at that moment, its startup fails ("free memory increased"), so exit only once the orchestrator
        # holds the GPU's startup lock.
        print(FINISHED_MARKER, flush=True)
        sys.stdin.readline()


if __name__ == "__main__":
    main()
