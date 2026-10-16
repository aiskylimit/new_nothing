"""Persistent vLLM rollout process on a dedicated GPU, without a training framework.

Uses vLLM's public LLM.apply_model API (written for 0.15.1, present in 0.18.0). Weight snapshots are local
safetensors files, loaded once per policy version before any generation.

Optional sleep mode (colocated layout): after each rollout the engine sleeps at
level 2, releasing all of its GPU memory (weights + KV cache) to the actor for
the backward/curvature phases. The next sync wakes the weights, loads the new
policy, then wakes the KV cache; vLLM restores its buffers itself.
The local rollout worker opts into pickle serialization for trusted GRIT
callbacks before importing vLLM or spawning its engine processes.
"""

import atexit
from functools import partial
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import traceback


def load_policy_weights(model, *, path):
    from safetensors import safe_open
    import torch

    device = next(model.parameters()).device
    with torch.no_grad(), safe_open(path, framework="pt", device="cpu") as snapshot:
        # vLLM maps HF names to packed QKV/gate-up weights through its model loader.
        loaded = model.load_weights((name, snapshot.get_tensor(name).to(device)) for name in snapshot.keys())
    if not loaded:
        raise RuntimeError("vLLM did not report any loaded policy weights")
    missing = set(dict(model.named_parameters())) - set(loaded)
    if missing:
        raise RuntimeError(f"vLLM policy snapshot did not load parameters: {sorted(missing)[:10]}")
    return len(loaded)


def _worker(connection, config, gpu):
    # Must precede CUDA initialization, including that inside vLLM.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    # Drop every torchrun launcher variable. vLLM builds its own 1-process torch.distributed group, and
    # an inherited TORCHELASTIC_USE_AGENT_STORE=True makes torch (2.10) open only a TCPStore *client*
    # on every rank, waiting for a torchrun agent store that does not exist: init hangs, then times out.
    launcher = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK", "GROUP_WORLD_SIZE",
                "ROLE_RANK", "ROLE_NAME", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
    for key in [*launcher, *(key for key in os.environ if key.startswith("TORCHELASTIC_"))]:
        os.environ.pop(key, None)
    try:
        from vllm import LLM, SamplingParams
        engine = LLM(**config, tensor_parallel_size=1, enable_prefix_caching=False,
                     logprobs_mode="raw_logprobs", generation_config="vllm")
        connection.send({"ok": True})
        version = None
        sleeping = False
        while True:
            command = connection.recv()
            if command["op"] == "close":
                break
            if command["op"] == "sleep":
                # Level 2 discards weights too; the next sync reloads them anyway.
                engine.sleep(level=2)
                sleeping, version = True, None
                connection.send({"ok": True})
            elif command["op"] == "sync":
                if sleeping:
                    engine.wake_up(tags=["weights"])
                counts = engine.apply_model(partial(load_policy_weights, path=command["path"]))
                if not counts or any(count <= 0 for count in counts):
                    raise RuntimeError("Incomplete vLLM weight synchronization")
                if sleeping:
                    engine.wake_up(tags=["kv_cache"])
                    sleeping = False
                version = command["version"]
                connection.send({"ok": True, "version": version})
            elif command["op"] == "generate":
                if sleeping or version is None or command["version"] != version:
                    raise RuntimeError("Refusing rollout from a stale policy version")
                outputs = engine.generate(
                    [{"prompt_token_ids": ids} for ids in command["prompts"]],
                    SamplingParams(**command["sampling"], logprobs=0), use_tqdm=command.get("progress", True),
                )
                groups = []
                for request in outputs:
                    group = []
                    for output in sorted(request.outputs, key=lambda output: output.index):
                        ids = list(output.token_ids)
                        values = [float(row[token].logprob) for row, token in zip(output.logprobs, ids, strict=True)]
                        group.append({"token_ids": ids, "log_probs": values})
                    groups.append(group)
                connection.send({"ok": True, "version": version, "groups": groups})
            else:
                raise ValueError("Unknown rollout command")
    except BaseException:
        try:
            connection.send({"ok": False, "error": traceback.format_exc()})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class VLLMRollout:
    def __init__(self, *, model, gpu, revision=None, dtype="auto", max_model_len=1024,
                 memory_utilization=0.8, seed=66, timeout=1800, trust_remote_code=False, progress=True,
                 sleep_mode=False):
        self.timeout = timeout
        self.progress = progress
        self.version = None
        self.sleep_mode = sleep_mode
        self.sleeping = False
        self._temporary = tempfile.TemporaryDirectory(prefix="grit-rollout-")
        context = mp.get_context("spawn")
        self._connection, child = context.Pipe()
        self._process = context.Process(target=_worker, args=(child, {
            "model": model, "revision": revision, "dtype": dtype,
            "max_model_len": max_model_len, "gpu_memory_utilization": memory_utilization,
            "seed": seed, "trust_remote_code": trust_remote_code,
            **({"enable_sleep_mode": True} if sleep_mode else {}),
        }, gpu))
        self._process.start()
        child.close()
        atexit.register(self.close)
        try:
            self._receive()
        except BaseException:
            self.close()
            raise

    def _receive(self):
        if not self._connection.poll(self.timeout):
            raise TimeoutError("vLLM rollout process did not respond; training stopped")
        try:
            reply = self._connection.recv()
        except EOFError as error:
            raise RuntimeError("vLLM rollout process exited") from error
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error", "vLLM rollout failed"))
        return reply

    def sync(self, model, version):
        from safetensors.torch import save_file

        if getattr(self, "sleeping", False):
            import torch
            # Return the actor's cached blocks so the waking engine can re-allocate its memory.
            torch.cuda.empty_cache()
        path = Path(self._temporary.name) / "policy.safetensors"
        # Clone tied tensors to avoid shared-storage rejection by safetensors.
        save_file({n: p.detach().cpu().contiguous().clone() for n, p in model.state_dict().items()}, str(path))
        self._connection.send({"op": "sync", "path": str(path), "version": version})
        reply = self._receive()
        if reply.get("version") != version:
            raise RuntimeError("vLLM acknowledged the wrong policy version")
        self.version = version
        self.sleeping = False
        path.unlink()

    def sleep(self):
        """Release vLLM GPU memory until the next sync (no-op without sleep mode)."""
        if not getattr(self, "sleep_mode", False) or self.sleeping:
            return
        self._connection.send({"op": "sleep"})
        self._receive()
        self.version = None
        self.sleeping = True

    def generate(self, prompt_ids, *, version, n=5, max_tokens=128, temperature=1.0, top_p=1.0, seed=66):
        if self.version != version:
            raise RuntimeError("Synchronize policy weights before generating rollouts")
        self._connection.send({"op": "generate", "version": version, "prompts": prompt_ids,
                               "progress": getattr(self, "progress", True),
                               "sampling": dict(n=n, max_tokens=max_tokens, temperature=temperature,
                                                top_p=top_p, seed=seed)})
        reply = self._receive()
        groups = reply.get("groups", [])
        if reply.get("version") != version or len(groups) != len(prompt_ids) or any(len(g) != n for g in groups):
            raise RuntimeError("Rollout version or response count does not match the requested batch")
        return groups

    def close(self):
        process = getattr(self, "_process", None)
        if process is None:
            return
        if process.is_alive():
            try:
                self._connection.send({"op": "close"})
            except (OSError, EOFError):
                pass
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        self._connection.close()
        self._temporary.cleanup()
        self._process = None
        atexit.unregister(self.close)
