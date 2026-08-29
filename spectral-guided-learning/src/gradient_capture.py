"""Phase 3: capture loss gradients, run per-sample SVD, emit step spectral strengths.

--verify cross-checks the analytic gradient against autograd on the first samples before
processing the corpus, since the whole pipeline rests on that identity.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoModelForCausalLM

from gradient_utils import analytic_hidden_gradients, capture_sequence_gradients, shift_for_causal_lm
from segmentation import record_step_spans
from spectral_utils import analyze_gradient_matrix


def sync_if_cuda(device) -> None:
    """Block until the CUDA stream drains, so a perf_counter() delta measures real device
    compute (kernel launches are async — without this the first timing is launch latency)."""
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def load_records(path: str, limit: int | None) -> list[dict]:
    with open(path) as handle:
        records = [json.loads(line) for line in handle]
    return records[:limit] if limit else records


def to_gradient_rows(step_spans: list[tuple[int, int]], response_start: int) -> list[tuple[int, int]]:
    """Convert absolute token spans to row indices of the gradient matrix G."""
    return [(start - response_start, end - response_start) for start, end in step_spans]


def verify_against_autograd(
    model, input_ids: torch.Tensor, span: tuple[int, int], max_positions: int = 512
) -> float:
    """Max absolute deviation between analytic and autograd gradients (float32).

    Only backprop through the small linear+CE head, not the transformer backbone -- doing
    the latter builds a full per-layer graph that OOMs at the paper's 32768 cutoff (Table 3).
    """
    start, end = span
    span = (start, min(end, start + max_positions))
    with torch.no_grad():
        hidden = model.model(input_ids=input_ids).last_hidden_state[0].detach().float()
    rows, targets = shift_for_causal_lm(hidden, input_ids[0], span)

    autograd_rows = rows.clone().requires_grad_(True)
    with torch.enable_grad():
        logits = autograd_rows @ model.get_output_embeddings().weight.float().T
        torch.nn.functional.cross_entropy(logits, targets, reduction="sum").backward()
    reference = autograd_rows.grad

    analytic = analytic_hidden_gradients(rows, targets, model.get_output_embeddings().weight)
    return float((analytic - reference).abs().max().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="optional yaml base; CLI flags below override it")
    parser.add_argument("--model-name", help="HF model id, e.g. Qwen/Qwen3-8B")
    parser.add_argument("--data-path", help="segmented jsonl from data_prep.py (Phase 2 output)")
    parser.add_argument("--output-dir", help="per-sample .npz spectra go here")
    parser.add_argument("--strengths-path", help="merged parquet of per-sample k*/step strengths")
    parser.add_argument("--energy-cutoff", type=float, help="k* cumulative-energy threshold (Eq. 4)")
    parser.add_argument("--chunk-size", type=int, help="positions per chunk (chunk x vocab) when unembedding")
    parser.add_argument("--dtype", help="model forward dtype, e.g. bfloat16")
    parser.add_argument("--device", help="cuda | cpu")
    parser.add_argument("--attn-implementation", help="sdpa | flash_attention_2")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verify", action="store_true", help="cross-check vs autograd first")
    parser.add_argument(
        "--num-shards", type=int, default=1,
        help="split the corpus across N independent GPU processes (capture is embarrassingly "
             "parallel per-sample: each shard writes its own .npz files, none overlap). After "
             "all shards finish, rerun with --num-shards 1 (the default) to merge: every sample "
             "already has an .npz, so that pass hits the resume path for all of them (no GPU "
             "work) and just rebuilds strengths_path from what's on disk.",
    )
    parser.add_argument("--shard-index", type=int, default=0, help="this process's shard, in [0, num-shards)")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text()) if args.config else {}
    overrides = {
        "model_name": args.model_name,
        "data_path": args.data_path,
        "output_dir": args.output_dir,
        "strengths_path": args.strengths_path,
        "energy_cutoff": args.energy_cutoff,
        "chunk_size": args.chunk_size,
        "dtype": args.dtype,
        "device": args.device,
        "attn_implementation": args.attn_implementation,
    }
    config.update({key: value for key, value in overrides.items() if value is not None})
    config.setdefault("energy_cutoff", 0.95)
    config.setdefault("chunk_size", 1024)
    config.setdefault("dtype", "bfloat16")
    config.setdefault("device", "cuda")
    config.setdefault("attn_implementation", "sdpa")

    device = config["device"]
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        dtype=getattr(torch, config["dtype"]),
        attn_implementation=config["attn_implementation"],
    ).to(device).eval()

    records = load_records(config["data_path"], args.limit)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    sharded = args.num_shards > 1
    if sharded:
        records = records[args.shard_index :: args.num_shards]
        print(f"shard {args.shard_index}/{args.num_shards}: {len(records)} records assigned")

    # Cast once for the whole corpus so the (V x d) matrix (V=151936 for Qwen3) isn't re-cast per sequence.
    unembedding = model.get_output_embeddings().weight.float()

    if args.verify and args.shard_index == 0:
        for record in records[:3]:  # paper-fidelity check on 3 samples before the corpus run
            input_ids = torch.tensor([record["input_ids"]], device=device)
            span = tuple(record["response_token_span"])
            deviation = verify_against_autograd(model, input_ids, span)
            print(f"analytic vs autograd max|diff| = {deviation:.3e}")
            assert deviation < 1e-2, "analytic gradient does not match autograd"

    rows, started, computed, resumed = [], time.time(), 0, 0
    grad_times, svd_times = [], []  # per-computed-sample; SVD isolated (paper omits its cost, R3)
    for index, record in enumerate(records):
        response_start, response_end = record["response_token_span"]
        step_spans = record_step_spans(record)
        npz_path = output_dir / f"{record['id']}.npz"

        # Resume: reuse an existing .npz instead of recomputing, so an interrupted run restarts from N, not 0.
        if npz_path.exists():
            cached = np.load(npz_path)
            rows.append(
                {
                    "id": record["id"],
                    "k_star": int(cached["k_star"]),
                    "n_response_tokens": response_end - response_start,
                    "n_steps": len(step_spans),
                    "step_strengths": [float(x) for x in cached["step_strengths"]],
                }
            )
            resumed += 1
            continue

        input_ids = torch.tensor([record["input_ids"]], device=device)
        sync_if_cuda(device)
        grad_start = time.perf_counter()
        gradient_matrix = capture_sequence_gradients(
            model, input_ids, (response_start, response_end),
            chunk_size=config["chunk_size"], unembedding=unembedding,
        )
        sync_if_cuda(device)
        grad_times.append(time.perf_counter() - grad_start)

        result = analyze_gradient_matrix(
            gradient_matrix,
            to_gradient_rows(step_spans, response_start),
            energy_cutoff=config["energy_cutoff"],
        )
        svd_times.append(result.svd_seconds)  # SVD wall time, cuda-synchronized inside
        del gradient_matrix

        # Write the .npz before appending the row: it's the durable progress unit the resume path above keys on.
        np.savez_compressed(
            npz_path,
            k_star=result.k_star,
            singular_values=result.singular_values.numpy(),
            step_strengths=np.asarray(result.step_strengths, dtype=np.float32),
        )
        rows.append(
            {
                "id": record["id"],
                "k_star": result.k_star,
                "n_response_tokens": response_end - response_start,
                "n_steps": len(step_spans),
                "step_strengths": result.step_strengths,
            }
        )
        computed += 1
        if computed % 50 == 0:
            elapsed = time.time() - started
            print(f"{computed} computed (+{resumed} resumed, {index + 1}/{len(records)} seen), "
                  f"{elapsed / computed:.2f}s/sample")

    if resumed:
        print(f"resumed {resumed} samples from existing npz; computed {computed} fresh")

    if sharded:
        print(f"shard {args.shard_index}/{args.num_shards} done ({len(rows)} rows) -- not writing "
              f"{config['strengths_path']} from a partial shard. Once every shard has finished, "
              f"rerun with --num-shards 1 (default) to merge: all .npz already exist, so that "
              f"pass hits the resume path for every record and just rebuilds the full parquet.")
        return

    frame = pd.DataFrame(rows)
    frame.to_parquet(config["strengths_path"])
    print(f"wrote {len(frame)} rows -> {config['strengths_path']} (+ per-sample npz in {output_dir})")
    print(f"runtime: {time.time() - started:.0f}s total, "
          f"{(time.time() - started) / max(computed, 1):.2f}s/computed-sample")
    if svd_times:  # timing summary — the paper never reports the SVD cost (R3)
        total_grad, total_svd = sum(grad_times), sum(svd_times)
        print(
            f"gradient: mean={statistics.mean(grad_times) * 1e3:.0f}ms/sample "
            f"median={statistics.median(grad_times) * 1e3:.0f}ms total={total_grad:.0f}s"
        )
        print(
            f"SVD: mean={statistics.mean(svd_times) * 1e3:.0f}ms/sample "
            f"median={statistics.median(svd_times) * 1e3:.0f}ms max={max(svd_times) * 1e3:.0f}ms "
            f"total={total_svd:.0f}s  ({100 * total_svd / max(total_grad + total_svd, 1e-9):.0f}% of gradient+SVD)"
        )
    print(f"k* : mean={frame.k_star.mean():.1f} median={frame.k_star.median():.0f} max={frame.k_star.max()}")
    ratio = (frame.k_star / frame.n_response_tokens).mean()
    print(f"k*/T mean ratio = {ratio:.4f}  (low-rank premise holds if << 1)")

    strengths = np.concatenate([np.asarray(row) for row in frame.step_strengths])
    quantiles = np.quantile(strengths, [0.1, 0.5, 0.9])
    print(
        f"step strength: mean={strengths.mean():.4f} p10={quantiles[0]:.4f} "
        f"median={quantiles[1]:.4f} p90={quantiles[2]:.4f} max={strengths.max():.4f}"
    )
    print(f"strength spread (p90/p10) = {quantiles[2] / max(quantiles[0], 1e-12):.1f}x  (flat => selection drops little)")


if __name__ == "__main__":
    main()
