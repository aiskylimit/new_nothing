"""vLLM-accelerated rollout generation for training (data-parallel, server
mode) - replaces `tropic.rollout.generate_rollout`'s sequential HF
`model.generate()` calls (the actual bottleneck making training ~25-26 min/
step, ~42h for 100 steps) with N independent `vllm serve` processes, one per
free GPU, each handling a slice of the per-step rollout batch in parallel.

FIDELITY NOTE (see the approved plan, C:\\Users\\LENOVO\\.claude\\plans\\
you-are-a-senior-idempotent-pizza.md): OPSD's REAL colocate mode (verified
live against opsd_trainer.py, github.com/siyan-zhao/OPSD) syncs weights via
`model.merge_adapter()` + vLLM's INTERNAL `engine.model_executor.driver_
worker.model_runner.model.load_weights(...)`, in the SAME process/GPU as
training (~60% extra VRAM per their own launch script,
`--vllm_gpu_memory_utilization 0.6`). That was deliberately NOT replicated
here (real OOM risk on the 40GB GPUs this project already uses, which have
OOM'd on eval alone this session) - instead: vLLM runs as a fully separate
process on a SEPARATE GPU, and weight sync goes through vLLM's own PUBLIC
runtime LoRA-reload HTTP API instead of its private engine internals.

Verified live against the ACTUAL vLLM version installed on the training
cluster (v0.8.3, `pip show vllm` on dionysos) by fetching vllm/entrypoints/
openai/api_server.py + serving_models.py + protocol.py at tag v0.8.3 from
github.com/vllm-project/vllm - not assumed from memory:
  - `POST /v1/load_lora_adapter` {"lora_name": str, "lora_path": str} and
    `POST /v1/unload_lora_adapter` {"lora_name": str} are real, PUBLIC
    endpoints, but only registered when the server is launched with env var
    `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1` (vLLM's own gate - logs a "local
    development only" warning, which is fine for this controlled research
    cluster use).
  - Loading a `lora_name` that's ALREADY loaded is REJECTED with HTTP 400
    ("has already been loaded") - `serving_models.py`'s own validation.
    There is no "just overwrite" call - every refresh must unload-then-load
    the SAME name. The very first refresh has nothing to unload yet, so a
    400 from THAT unload call is expected and swallowed (not from the load
    call - a 400 there is a real error).
  - `POST /v1/completions`'s `prompt` field accepts `list[list[int]]`
    (`protocol.py`'s `CompletionRequest.prompt` type) - a whole shard's
    token-id prompts can be sent as ONE batched request per replica, vLLM's
    own continuous batching handles the parallelism server-side.

KNOWN IMPRECISION (documented, not silently swallowed): `/v1/completions`
returns each completion as DECODED TEXT (`choices[i].text`), not raw token
ids - `generate_rollout_batch_parallel` re-tokenizes that text with the SAME
tokenizer (`add_special_tokens=False`) to recover `generated_ids`/`gen_len`
in the shape the rest of the training loop expects (matching `tropic.
rollout.generate_rollout`'s return contract). Re-tokenizing decoded text does
not ALWAYS reproduce the exact original token sequence at BPE merge
boundaries - a known, generally-accepted imprecision when bridging an
HTTP text-completions API back into token-level training (this is not
unique to this project; every framework that drives vLLM's OpenAI-compatible
server for rollouts faces the same trade-off). Not resolved here because
vLLM's public completions API has no raw-token-id response mode as of
v0.8.3 - flagged for whoever runs this, not hidden.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import PreTrainedTokenizerBase

DEFAULT_LORA_NAME = "tropic_live"


def launch_vllm_replicas(
    model_name: str,
    gpu_ids: list[int],
    base_port: int,
    lora_rank: int,
    startup_timeout: float = 600.0,
    log_dir: str | None = None,
    vllm_executable: str = "vllm",
    gpu_memory_utilization: float | None = None,
    max_model_len: int | None = None,
) -> tuple[list[subprocess.Popen], list[int]]:
    """Launches len(gpu_ids) independent `vllm serve` processes, one per GPU
    (data-parallel replicas, NOT tensor-parallel - the model is 1.7B/4B,
    already fits on one 40GB GPU, so splitting ONE model across GPUs would
    only add cross-GPU communication overhead for no benefit; independent
    replicas instead each handle a slice of the per-step rollout batch).
    Blocks until every replica's `/health` endpoint responds. Caller is
    responsible for terminating the returned processes when done (e.g. in a
    `finally` block) - these are NOT daemonized/auto-cleaned-up.

    `log_dir` (defaults to the CURRENT working directory if not given):
    each replica's stdout+stderr is written to `<log_dir>/vllm_replica_
    gpu<id>.log` - NEVER swallowed to DEVNULL, so a startup crash (wrong CLI
    flag, port already in use, OOM, etc.) is diagnosable instead of showing
    up only as a silent, zero-GPU-memory `<defunct>` process.

    `vllm_executable` (default "vllm", resolved via inherited PATH): pass an
    ABSOLUTE PATH when the TRAINING process runs in a different conda env
    than the one vLLM is actually installed/working in - e.g. this project's
    real setup has vLLM 0.8.3 needing torch==2.6.0's DEFAULT PyPI CUDA-12.4
    wheel (bundled nvidia-cuda-*-cu12==12.4.127 packages), which conflicted
    (undefined symbol in vllm's precompiled _C.abi3.so) with the training
    env's own torch build - so vLLM lives in a SEPARATE env
    (`tropic-vllm`) and must be invoked via its full path, e.g.
    '/research/.../miniconda3/envs/tropic-vllm/bin/vllm', NOT just "vllm"
    (which would resolve to the training env's own, possibly broken, vllm
    on PATH).

    `gpu_memory_utilization` (default None): when given, passed through as
    `--gpu-memory-utilization <value>` to every replica - otherwise vLLM's
    own default (0.9) applies unchanged. Useful on GPUs with much more VRAM
    headroom than this project's original 40GB A100s (e.g. a B200) to widen
    the KV-cache pool and admit more concurrent requests, which matters more
    under an unrestricted (top_k=-1) sampling recipe where a handful of
    generations can run to the full max_new_tokens.

    `max_model_len` (default None): when given, passed through as
    `--max-model-len <value>` - otherwise vLLM auto-detects the model's own
    max_position_embeddings, which for a reasoning ("Think") model can be far
    larger than anything this project ever actually sends it (prompt +
    max_new_tokens is at most ~41000 here, for eval's max_new_tokens=38912).
    vLLM sizes its KV-cache block pool and max concurrent sequence count
    against WHATEVER max_model_len it is using - a needlessly large
    auto-detected value makes it reserve per-sequence KV cache for a context
    length nothing ever reaches, so it can support only a SMALL number of
    concurrent sequences even on a GPU with plenty of free VRAM (observed
    live: ~17GB used out of a B200's 183GB with gpu_memory_utilization=0.9,
    on a single replica running one large eval batch). Capping this to the
    actual maximum needed lets the same memory budget fit many more
    concurrent sequences, directly increasing throughput without adding
    replica GPUs."""
    log_dir_path = log_dir or "."
    procs = []
    ports = [base_port + i for i in range(len(gpu_ids))]
    for gpu_id, port in zip(gpu_ids, ports):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "1"
        cmd = [
            vllm_executable, "serve", model_name,
            "--enable-lora", "--max-lora-rank", str(lora_rank),
            "--port", str(port),
            # NO --disable-log-requests / --disable-uvicorn-access-log here on
            # purpose: that flag was renamed between vLLM versions, and this
            # server's offline install tool has already twice substituted a
            # DIFFERENT vllm version than the one requested in tropic.txt
            # (v0.2.5, then v0.18.0, for requests of v0.28.0 then v0.11.0) -
            # see tropic.txt's own comment for that history. This server can
            # only be debugged by submitting project_commands.sh and waiting
            # (no interactive shell to check `vllm serve --help` first), so a
            # flag whose exact name depends on which version actually lands
            # is worth avoiding entirely rather than re-guessing each time.
            # The flag is a pure logging-verbosity nicety (silences
            # per-request access logs), not required for correctness. The
            # ONLINE A100 cluster's copy of this same file
            # (tropic/vllm_rollout_client.py, no offline_ prefix) still
            # passes "--disable-log-requests" for its real, confirmed v0.8.3
            # install - do not copy that back into this file.
        ]
        if gpu_memory_utilization is not None:
            cmd += ["--gpu-memory-utilization", str(gpu_memory_utilization)]
        if max_model_len is not None:
            cmd += ["--max-model-len", str(max_model_len)]
        log_path = f"{log_dir_path}/vllm_replica_gpu{gpu_id}.log"
        log_file = open(log_path, "w")  # noqa: SIM115 - lifetime matches the subprocess, not closed here
        procs.append(subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT))
    try:
        for port in ports:
            _wait_for_health(port, startup_timeout)
    except Exception:
        dead = [p for p in procs if p.poll() is not None]
        if dead:
            raise RuntimeError(
                f"{len(dead)}/{len(procs)} vLLM replica(s) exited during startup - "
                f"see {log_dir_path}/vllm_replica_gpu<id>.log for the real error"
            ) from None
        raise
    return procs, ports


def shutdown_vllm_replicas(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        p.terminate()
    for p in procs:
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()


def _wait_for_health(port: int, timeout: float) -> None:
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5) as resp:
                if resp.status == 200:
                    return
        except Exception as e:  # noqa: BLE001 - server not up yet, keep polling
            last_error = e
        time.sleep(5)
    raise TimeoutError(f"vLLM server on port {port} did not become healthy within {timeout}s (last error: {last_error})")


def _post_json(port: int, path: str, payload: dict, timeout: float = 60.0) -> tuple[int, str]:
    req = urllib.request.Request(
        f"http://localhost:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def refresh_lora_adapter(ports: list[int], adapter_path: str, lora_name: str = DEFAULT_LORA_NAME) -> None:
    """Hot-swaps the LoRA adapter on ALL replicas to the checkpoint at
    `adapter_path` (must already be written to disk, e.g. via `model.
    save_pretrained(adapter_path)`, BEFORE calling this). Unloads first on
    every call - loading a name that's already loaded is rejected with 400
    (verified against the real v0.8.3 validation, see module docstring); the
    very first refresh has nothing to unload yet, which returns 404
    ("NotFoundError") - verified LIVE against a real running v0.8.3 server
    (the earlier assumption that this was also a 400, inferred from reading
    the source without seeing its actual status_code=... line, was WRONG) -
    that 404 is expected and swallowed, but a 400 from the LOAD call always
    raises."""
    for port in ports:
        status, body = _post_json(port, "/v1/unload_lora_adapter", {"lora_name": lora_name})
        if status not in (200, 404):
            raise RuntimeError(f"unload_lora_adapter on port {port} failed ({status}): {body}")
        status, body = _post_json(
            port, "/v1/load_lora_adapter", {"lora_name": lora_name, "lora_path": str(adapter_path)}
        )
        if status != 200:
            raise RuntimeError(f"load_lora_adapter on port {port} failed ({status}): {body}")


def _generate_on_replica(
    port: int, lora_name: str, prompt_token_ids_batch: list[list[int]], sampling_kwargs: dict,
) -> list[dict]:
    payload = {
        "model": lora_name,
        "prompt": prompt_token_ids_batch,  # list[list[int]] - CompletionRequest.prompt supports this (v0.8.3)
        "max_tokens": sampling_kwargs["max_new_tokens"],
        "temperature": sampling_kwargs["temperature"],
        "top_p": sampling_kwargs["top_p"],
    }
    top_k = sampling_kwargs.get("top_k")
    if top_k is not None and top_k > 0:
        payload["top_k"] = top_k
    status, body = _post_json(port, "/v1/completions", payload, timeout=1800.0)
    if status != 200:
        raise RuntimeError(f"completions on port {port} failed ({status}): {body}")
    data = json.loads(body)
    # OpenAI-compatible response: one choice per prompt, in the same order,
    # when n=1 (the default we rely on - one sample per rollout slot).
    return data["choices"]


def generate_rollout_batch_parallel(
    ports: list[int],
    student_prompt_ids_list: list[torch.Tensor],
    tokenizer: PreTrainedTokenizerBase,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int | None = None,
    lora_name: str = DEFAULT_LORA_NAME,
) -> list[tuple[torch.Tensor, int]]:
    """Data-parallel replacement for `effective_batch_size` sequential calls
    to `tropic.rollout.generate_rollout` - splits `student_prompt_ids_list`
    round-robin across `ports` (one batched `/v1/completions` request per
    replica, covering that replica's whole shard at once), gathers results
    back in ORIGINAL order. Returns the SAME `(generated_ids, gen_len)` per
    item shape `generate_rollout` returns, via re-tokenizing each returned
    completion's text (see module docstring's KNOWN IMPRECISION note)."""
    n = len(student_prompt_ids_list)
    shard_prompts: list[list[list[int]]] = [[] for _ in ports]
    shard_indices: list[list[int]] = [[] for _ in ports]
    for i, ids in enumerate(student_prompt_ids_list):
        shard = i % len(ports)
        shard_prompts[shard].append(ids.tolist())
        shard_indices[shard].append(i)

    sampling_kwargs = dict(max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, top_k=top_k)
    results: list[tuple[torch.Tensor, int] | None] = [None] * n
    with ThreadPoolExecutor(max_workers=len(ports)) as executor:
        future_to_indices = {}
        for port, prompts, indices in zip(ports, shard_prompts, shard_indices):
            if not prompts:
                continue
            future = executor.submit(_generate_on_replica, port, lora_name, prompts, sampling_kwargs)
            future_to_indices[future] = indices
        for future, indices in future_to_indices.items():
            choices = future.result()
            for idx, choice in zip(indices, choices):
                text = choice["text"]
                generated_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)
                if generated_ids.dim() == 0:
                    generated_ids = generated_ids.unsqueeze(0)
                results[idx] = (generated_ids, generated_ids.shape[0])

    assert all(r is not None for r in results), "some rollout slots were never filled by any replica"
    return results  # type: ignore[return-value]


def _generate_eval_on_replica(
    port: int, lora_name: str, prompts_batch: list[str], sampling_kwargs: dict,
) -> list[dict]:
    payload = {
        "model": lora_name,
        # list[str] - standard OpenAI-Completions batch-of-strings form,
        # VERIFIED LIVE against this project's real v0.8.3 server (see
        # generate_eval_batch_parallel's docstring).
        "prompt": prompts_batch,
        "max_tokens": sampling_kwargs["max_tokens"],
        "temperature": sampling_kwargs["temperature"],
        "top_p": sampling_kwargs["top_p"],
        "n": sampling_kwargs["n"],
    }
    top_k = sampling_kwargs.get("top_k")
    if top_k is not None and top_k > 0:
        payload["top_k"] = top_k
    min_p = sampling_kwargs.get("min_p")
    if min_p:
        payload["min_p"] = min_p
    # 604800s (7 days), not the training-rollout client's 1800s: TWO real
    # crashes hit mid-eval on the online cluster (see chat history) - an
    # early checkpoint (step25) exceeding a 3600s (1h) timeout, and later a
    # step10 checkpoint exceeding the 10800s (3h) timeout that replaced it.
    # This offline package's own single-vLLM-replica topology (2 GPUs: 1
    # main + 1 replica, vs the online cluster's 3 replicas) hit that SAME 3h
    # timeout for real on a single aime26/hmmt25 batch (30 problems x 12
    # samples, no data-parallel split across replicas to shorten it). 7 days
    # is a deliberately generous margin, not a measured worst case.
    status, body = _post_json(port, "/v1/completions", payload, timeout=604800.0)
    if status != 200:
        raise RuntimeError(f"eval completions on port {port} failed ({status}): {body}")
    data = json.loads(body)
    return data["choices"]


def generate_eval_batch_parallel(
    ports: list[int],
    prompts: list[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
    n: int,
    top_k: int | None = None,
    min_p: float | None = None,
    lora_name: str = DEFAULT_LORA_NAME,
) -> list[list[str]]:
    """Data-parallel EVAL-time generation (n samples/problem, matching OPSD's
    own eval/evaluate_math.py: k completions per problem via a single
    LoRA-served vLLM engine) - splits `prompts` round-robin across `ports`,
    one batched `/v1/completions` request per replica (n samples per prompt
    within that request), gathers results back in ORIGINAL order. Unlike
    `generate_rollout_batch_parallel` (training), this returns DECODED TEXT
    directly (`list[list[str]]`, outer=per-problem, inner=n samples) - eval
    only needs text for grading (tropic.eval.score_generations), no
    token-level re-tokenization/imprecision concern here.

    VERIFIED LIVE against the real v0.8.3 server (`curl .../v1/completions`
    with `prompt: ["Hello, how are you", "The capital of France is"], n: 2`
    against a running replica on hermes): `prompt` as `list[str]` is accepted
    (not just the `list[list[int]]` form training rollouts use), and each
    prompt's `n` choices come back with `choice["index"] == prompt_position *
    n + sample_position` exactly as assumed (prompt 0 -> index 0,1; prompt 1
    -> index 2,3) - choices are still re-sorted by `index` before grouping,
    as a cheap safety net in case a different vLLM version/config ever
    reorders them."""
    shard_prompts: list[list[str]] = [[] for _ in ports]
    shard_indices: list[list[int]] = [[] for _ in ports]
    for i, prompt in enumerate(prompts):
        shard = i % len(ports)
        shard_prompts[shard].append(prompt)
        shard_indices[shard].append(i)

    sampling_kwargs = dict(max_tokens=max_tokens, temperature=temperature, top_p=top_p, n=n, top_k=top_k, min_p=min_p)
    results: list[list[str] | None] = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=len(ports)) as executor:
        future_to_indices = {}
        for port, shard, indices in zip(ports, shard_prompts, shard_indices):
            if not shard:
                continue
            future = executor.submit(_generate_eval_on_replica, port, lora_name, shard, sampling_kwargs)
            future_to_indices[future] = indices
        for future, indices in future_to_indices.items():
            choices = sorted(future.result(), key=lambda c: c["index"])
            for local_pos, orig_idx in enumerate(indices):
                samples = choices[local_pos * n:(local_pos + 1) * n]
                results[orig_idx] = [c["text"] for c in samples]

    assert all(r is not None for r in results), "some eval prompt slots were never filled by any replica"
    return results  # type: ignore[return-value]
