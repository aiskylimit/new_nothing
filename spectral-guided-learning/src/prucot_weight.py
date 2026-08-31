"""Phase 3 (Pru-CoT baseline): step-level importance via global optimization on a frozen model.

Implements Pru-CoT's step-wise noise soft-masking + gradient-based optimization (paper Eqs.
1-2, Pru-CoT/cot_weight.py): each reasoning step gets a learnable weight w_t in [0,1] that
interpolates its token embeddings with a fixed filler-token embedding, optimized per-sample to
minimize the loss on the supervised span -- gradients that restore accuracy pull w_t toward 1
for causally load-bearing steps, while steps that contribute nothing stay near init or drop to 0.

See probe_scope() below and segmentation.solution_step_start() for how each record is split
into a scored chain-of-thought and an unscored, always-supervised solution tail (paper Eq. 2:
only the solution is supervised).

--config points at a yaml with the same fields as a fallback/override base; CLI flags always
win when both are given. Unlike gradient_capture.py's single closed-form pass, this backprops
through the full frozen model every epoch (same activation-memory footprint as full
fine-tuning); --num-shards/--shard-index split the corpus the same way gradient_capture.py does.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from gradient_capture import load_records
from segmentation import record_step_spans, solution_step_start

IGNORE_INDEX = -100


def build_replacement_mask(
    seq_len: int, step_spans: list[tuple[int, int]], weights: torch.Tensor, device
) -> torch.Tensor:
    """Per-position soft-mask (1, seq_len, 1): weights[s] over step s's span, 1.0 elsewhere (only
    response steps are prunable, paper Section 3.3). The in-place slice writes below turn this
    leaf tensor into a non-leaf carrying gradient back to `weights` (mirrors the paper's code).
    """
    mask = torch.ones(1, seq_len, 1, device=device, dtype=weights.dtype)
    for index, (start, end) in enumerate(step_spans):
        mask[:, start:end, :] = weights[index]
    return mask


def fuse_embeddings(
    embeddings: torch.Tensor, replacement_mask: torch.Tensor, noise_embedding: torch.Tensor
) -> torch.Tensor:
    """Eq. 1: e~ = w * e + (1 - w) * n."""
    return replacement_mask * embeddings + (1 - replacement_mask) * noise_embedding


def best_epoch_weights(loss_history: list[float], weight_history: list[torch.Tensor]) -> torch.Tensor:
    """The paper retains the weights from whichever epoch reached the lowest loss."""
    return weight_history[int(np.argmin(loss_history))]


def build_labels(seq_len: int, supervised_span: tuple[int, int], input_ids: torch.Tensor) -> torch.Tensor:
    """Supervise only the given span (paper Eq. 2: the solution, when a split is available)."""
    labels = torch.full((1, seq_len), IGNORE_INDEX, dtype=torch.long, device=input_ids.device)
    start, end = supervised_span
    labels[:, start:end] = input_ids[:, start:end]
    return labels


def probe_scope(
    step_spans: list[tuple[int, int]], step_texts: list[str], response_token_span: tuple[int, int]
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    """Split a record into (chain-of-thought step spans to score, span to supervise).

    With a "</think>" marker: CoT = steps before the tag, supervised span = tag step through
    end of response (Eq. 2's solution target). Without one: falls back to the whole response.
    """
    boundary = solution_step_start(step_texts)
    if boundary is None:
        return step_spans, response_token_span
    return step_spans[:boundary], (step_spans[boundary][0], step_spans[-1][1])


def optimize_sample_weights(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    step_spans: list[tuple[int, int]],
    noise_embedding: torch.Tensor,
    epochs: int,
    init_v: float,
    optimizer_name: str,
    lr: float,
) -> list[float]:
    """Run the probing optimization for one sample; returns the best-epoch step weights."""
    device = input_ids.device
    num_steps = len(step_spans)
    if num_steps == 0:
        return []

    with torch.no_grad():
        embeddings = model.get_input_embeddings()(input_ids).detach()

    weights = torch.full((num_steps,), init_v, device=device, dtype=embeddings.dtype, requires_grad=True)
    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW([weights], lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD([weights], lr=lr)
    else:
        raise ValueError(f"unknown optimizer {optimizer_name}")

    loss_history, weight_history = [], []
    for _ in range(epochs):
        optimizer.zero_grad()
        replacement_mask = build_replacement_mask(input_ids.shape[1], step_spans, weights, device)
        fused = fuse_embeddings(embeddings, replacement_mask, noise_embedding)
        outputs = model(inputs_embeds=fused, attention_mask=attention_mask, labels=labels)
        outputs.loss.backward()
        optimizer.step()
        with torch.no_grad():
            weights.clamp_(0, 1)
        loss_history.append(float(outputs.loss.item()))
        weight_history.append(weights.detach().clone().cpu())

    return best_epoch_weights(loss_history, weight_history).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="optional yaml base; CLI flags below override it")
    parser.add_argument("--model-name", help="frozen student model used as the importance probe")
    parser.add_argument("--data-path", help="segmented jsonl from data_prep.py (Phase 2 output)")
    parser.add_argument("--output-dir", help="per-sample .npz weights go here")
    parser.add_argument("--weights-path", help="merged parquet of per-sample step weights")
    parser.add_argument("--epochs", type=int, help="probing epochs per sample (paper: 3)")
    parser.add_argument("--init-v", type=float, help="initial weight value (paper: 0.5)")
    parser.add_argument("--optimizer", choices=["SGD", "AdamW"], help="paper: SGD")
    parser.add_argument("--lr", type=float, help="paper: 10 (SGD on a [0,1]-clamped scalar per step)")
    parser.add_argument("--filler-token", help="noise token substituted for pruned steps (paper: '.')")
    parser.add_argument("--dtype", help="model forward dtype, e.g. bfloat16")
    parser.add_argument("--device", help="cuda | cpu")
    parser.add_argument("--attn-implementation", help="sdpa | flash_attention_2")
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction,
        help="trade compute for activation memory (default true; the probe backprops through "
        "the full frozen model, same footprint as full fine-tuning -- see train_sft.py)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--max-length", type=int, default=None,
        help="drop records longer than this many tokens (paper filtered >8192); unset/None keeps every sample",
    )
    parser.add_argument(
        "--num-shards", type=int, default=1,
        help="split the corpus across N independent GPU processes, same convention as "
        "gradient_capture.py --num-shards (each shard writes its own .npz files). Rerun with "
        "--num-shards 1 (default) after every shard finishes to merge into the parquet.",
    )
    parser.add_argument("--shard-index", type=int, default=0, help="this process's shard, in [0, num-shards)")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text()) if args.config else {}
    overrides = {
        "model_name": args.model_name,
        "data_path": args.data_path,
        "output_dir": args.output_dir,
        "weights_path": args.weights_path,
        "epochs": args.epochs,
        "init_v": args.init_v,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "filler_token": args.filler_token,
        "dtype": args.dtype,
        "device": args.device,
        "attn_implementation": args.attn_implementation,
        "gradient_checkpointing": args.gradient_checkpointing,
    }
    config.update({key: value for key, value in overrides.items() if value is not None})
    config.setdefault("epochs", 3)
    config.setdefault("init_v", 0.5)
    config.setdefault("optimizer", "SGD")
    config.setdefault("lr", 10.0)
    config.setdefault("filler_token", ".")
    config.setdefault("dtype", "bfloat16")
    config.setdefault("device", "cuda")
    config.setdefault("attn_implementation", "sdpa")
    config.setdefault("gradient_checkpointing", True)

    device = config["device"]
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        dtype=getattr(torch, config["dtype"]),
        attn_implementation=config["attn_implementation"],
    ).to(device)
    model.config.use_cache = False
    for param in model.parameters():
        param.requires_grad = False
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # HF only takes the checkpointed path in training mode ("if self.gradient_checkpointing and
        # self.training"). This probe backprops through the full frozen model, so leaving it in
        # eval() silently disables checkpointing and retains every layer's activations -- fine for
        # the 1.7B model but OOMs the 4B one at 32k tokens. Params are frozen and Qwen dropout is 0,
        # so train() is numerically identical here; it only re-enables checkpointing.
        model.train()
    else:
        model.eval()

    filler_id = tokenizer.convert_tokens_to_ids(config["filler_token"])
    with torch.no_grad():
        noise_embedding = model.get_input_embeddings()(torch.tensor([filler_id], device=device)).detach()

    records = load_records(config["data_path"], args.limit)
    if args.max_length is not None:
        kept = [r for r in records if len(r["input_ids"]) <= args.max_length]
        print(f"length filter (<= {args.max_length} tokens): kept {len(kept)}/{len(records)} records")
        records = kept
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    sharded = args.num_shards > 1
    if sharded:
        records = records[args.shard_index :: args.num_shards]
        print(f"shard {args.shard_index}/{args.num_shards}: {len(records)} records assigned")

    rows, started, computed, resumed = [], time.time(), 0, 0
    for index, record in enumerate(records):
        step_spans = record_step_spans(record)
        npz_path = output_dir / f"{record['id']}.npz"

        # Resume: existing .npz is reused (mirrors gradient_capture.py), so a crash restarts from N.
        if npz_path.exists():
            cached = np.load(npz_path)
            rows.append({"id": record["id"], "step_weights": [float(x) for x in cached["step_weights"]]})
            resumed += 1
            continue

        input_ids = torch.tensor([record["input_ids"]], device=device)
        attention_mask = torch.ones_like(input_ids)

        # Decoded rather than re-split from record["response"]: step_token_spans can drop zero-
        # token pieces split_into_step_texts would still emit, so decoding stays 1:1 with step_spans.
        step_texts = [tokenizer.decode(record["input_ids"][start:end]) for start, end in step_spans]
        cot_step_spans, supervised_span = probe_scope(
            step_spans, step_texts, tuple(record["response_token_span"])
        )
        labels = build_labels(input_ids.shape[1], supervised_span, input_ids)

        weights = optimize_sample_weights(
            model, input_ids, attention_mask, labels, cot_step_spans, noise_embedding,
            epochs=config["epochs"], init_v=config["init_v"],
            optimizer_name=config["optimizer"], lr=config["lr"],
        )

        np.savez_compressed(npz_path, step_weights=np.asarray(weights, dtype=np.float32))
        rows.append({"id": record["id"], "step_weights": weights})
        computed += 1
        if computed % 20 == 0:
            elapsed = time.time() - started
            print(f"{computed} computed (+{resumed} resumed, {index + 1}/{len(records)} seen), "
                  f"{elapsed / computed:.2f}s/sample")

    if resumed:
        print(f"resumed {resumed} samples from existing npz; computed {computed} fresh")

    if sharded:
        print(f"shard {args.shard_index}/{args.num_shards} done ({len(rows)} rows) -- not writing "
              f"{config['weights_path']} from a partial shard. Once every shard has finished, "
              f"rerun with --num-shards 1 (default) to merge: all .npz already exist, so that "
              f"pass hits the resume path for every record and just rebuilds the full parquet.")
        return

    frame = pd.DataFrame(rows)
    frame.to_parquet(config["weights_path"])
    print(f"wrote {len(frame)} rows -> {config['weights_path']} (+ per-sample npz in {output_dir})")
    print(f"runtime: {time.time() - started:.0f}s total, "
          f"{(time.time() - started) / max(computed, 1):.2f}s/computed-sample")

    non_empty = [np.asarray(row) for row in frame.step_weights if len(row)]
    if non_empty:
        all_weights = np.concatenate(non_empty)
        quantiles = np.quantile(all_weights, [0.1, 0.5, 0.9])
        print(
            f"step weight: mean={all_weights.mean():.4f} p10={quantiles[0]:.4f} "
            f"median={quantiles[1]:.4f} p90={quantiles[2]:.4f} max={all_weights.max():.4f}"
        )


if __name__ == "__main__":
    main()
