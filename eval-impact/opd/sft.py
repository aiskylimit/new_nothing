"""Prepare a frozen Dolly teacher adapter before the paired OPD experiment."""
import argparse
import json
import math
import random
import time
from pathlib import Path
import torch
from .data import read_jsonl, save_json, digest
from .train import supervised_loss


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--steps", type=int, default=0, help="0 = full epochs; positive = smoke-test cap")
    p.add_argument("--grad-accum", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lora-rank", type=int, default=256)
    p.add_argument("--lora-alpha", type=int, default=8)
    p.add_argument("--lora-dropout", type=float, default=.1)
    p.add_argument("--lora-targets", default="q_proj,v_proj")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--max-prompt-length", type=int, default=256)
    p.add_argument("--save-every", type=int, default=64)
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--resume", default=None)
    args = p.parse_args()
    rows = read_jsonl(args.data)
    per_epoch = len(rows) // args.grad_accum
    steps = args.steps or args.epochs * per_epoch
    if min(per_epoch, steps, args.epochs, args.save_every) < 1 or steps > args.epochs * per_epoch:
        raise ValueError("Invalid SFT epoch/update budget")
    out = Path(args.out)
    config = {**vars(args), "updates": steps, "data_sha256": digest(Path(args.data).read_text()),
              "recipe_note": "Teacher SFT uses the declared Table-I LoRA recipe; exact upstream teacher checkpoint was not released here."}
    if args.resume:
        old = json.loads((out / "config.json").read_text())
        if {k: v for k, v in old.items() if k != "resume"} != {k: v for k, v in config.items() if k != "resume"}:
            raise ValueError("SFT resume config/data mismatch")
    elif out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Use --resume or a fresh output directory: {out}")
    save_json(out / "config.json", config)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype)).to(args.device)
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=args.lora_rank,
        lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=args.lora_targets.split(",")))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: .5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    start = 0
    if args.resume:
        state = torch.load(Path(args.resume) / "training_state.pt", map_location=args.device, weights_only=False)
        model.load_state_dict(state["adapter"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["rng"].cpu())
        if torch.cuda.is_available() and state["cuda_rng"]:
            torch.cuda.set_rng_state_all([r.cpu() for r in state["cuda_rng"]])
        start = state["step"]

    def checkpoint(step):
        folder = out / f"checkpoint-{step}"
        model.save_pretrained(folder)
        tok.save_pretrained(folder)
        torch.save({"step": step, "adapter": {k: v for k, v in model.state_dict().items() if "lora_" in k},
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
            folder / "training_state.pt")
        save_json(out / "latest.json", {"checkpoint": str(folder.resolve()), "step": step})

    begin, step = time.monotonic(), start
    model.train()
    try:
        with (out / "metrics.jsonl").open("a") as log:
            for step in range(start + 1, steps + 1):
                epoch, update = divmod(step - 1, per_epoch)
                order = list(range(len(rows)))
                random.Random(args.seed + epoch).shuffle(order)
                optimizer.zero_grad(set_to_none=True)
                loss_mean = 0.
                for index in order[update * args.grad_accum:(update + 1) * args.grad_accum]:
                    loss = supervised_loss(model, tok, rows[index], args.device, args.max_length, args.max_prompt_length)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite teacher SFT loss")
                    (loss / args.grad_accum).backward()
                    loss_mean += loss.item() / args.grad_accum
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                scheduler.step()
                record = {"step": step, "epoch": epoch, "ce": loss_mean, "lr": scheduler.get_last_lr()[0]}
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
                if step % args.save_every == 0:
                    checkpoint(step)
        checkpoint(step)
        save_json(out / "status.json", {"state": "completed", "step": step, "elapsed_seconds": time.monotonic() - begin})
    except Exception as exc:
        save_json(out / "status.json", {"state": "failed", "step": step, "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
