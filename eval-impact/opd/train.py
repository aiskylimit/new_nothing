"""Single-GPU comparison of direct OPD versus direct OPD + IMPACT.

The student samples fresh responses before each microbatch. Both methods use
the same objective, seed, prompts, optimizer, rollout budget, and checkpoints.
"""
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .alignment import response_alignment_loss
from .losses import require_shared_tokenizer, direct_opd_loss, build_direct_batch
from .bi import collect, render_prompt
from .data import digest, read_jsonl, save_json, write_jsonl



def make_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--student", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    p.add_argument("--teacher", default="Qwen/Qwen2.5-Math-7B-Instruct")
    p.add_argument("--student-revision", default=None)
    p.add_argument("--teacher-revision", default=None)
    p.add_argument("--method", choices=["opd", "opd_impact"], required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--bi-summary", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=256, help="Optimizer updates")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--projector-lr", type=float, default=5e-5)
    p.add_argument("--projector-vocab", type=int, default=4096)
    p.add_argument("--lora-rank", type=int, default=32, help="0 for full training")
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--max-prompt-length", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--rollout-retries", type=int, default=5)
    p.add_argument("--bi-max-length", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--kd-temperature", type=float, default=1.0)
    p.add_argument("--kd-rate", type=float, default=1.0, help="1 = pure OPD; values below 1 explicitly add reference CE")
    p.add_argument("--kd-objective", choices=["forward_kl", "reverse_kl", "js_divergence"], default="reverse_kl")
    p.add_argument("--impact-lambda", type=float, default=0.25)
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--bi-tau", type=float, default=2.0)
    p.add_argument("--selection", choices=["bi", "first", "last", "random"], default="bi")
    p.add_argument("--weighting", choices=["bi", "uniform"], default="bi")
    p.add_argument("--save-every", type=int, default=64)
    p.add_argument("--max-minutes", type=float, default=0, help="0 = no deadline; saves at next update boundary")
    p.add_argument("--resume", default=None, help="Checkpoint directory, including optimizer/RNG state")
    return p


def supervised_loss(student, tokenizer, row, device, max_length):
    prompt_ids = tokenizer.encode(render_prompt(tokenizer, row["prompt"]), add_special_tokens=False)
    response_ids = tokenizer.encode(row.get("response", ""), add_special_tokens=False)
    if not response_ids:
        raise ValueError(f"Missing reference response for CE: {row['id']}; use --kd-rate 1 for prompt-only OPD")
    ids = (prompt_ids + response_ids + [tokenizer.eos_token_id])[:max_length]
    if len(ids) <= len(prompt_ids):
        raise ValueError("No reference response fits context length")
    inputs = torch.tensor([ids[:-1]], device=device)
    labels = torch.tensor([ids[1:]], device=device)
    labels[:, :len(prompt_ids) - 1] = -100
    output = student(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False)
    return torch.nn.functional.cross_entropy(output.logits.float().flatten(0, 1), labels.flatten())


def main():
    args = make_parser().parse_args()
    if min(args.steps, args.grad_accum, args.save_every, args.projector_vocab, args.max_new_tokens,
           args.rollout_retries, args.max_length, args.max_prompt_length, args.top_k) < 1:
        raise ValueError("Step, length, vocabulary and accumulation budgets must be positive")
    if args.temperature <= 0 or not 0 < args.top_p <= 1 or args.bi_tau <= 0 or args.lora_rank < 0:
        raise ValueError("Invalid sampling, BI temperature or LoRA rank")
    if not 0 < args.kd_rate <= 1:
        raise ValueError("kd_rate must be in (0, 1]")
    if args.dtype == "float16":
        raise ValueError("Use bfloat16 on GPU or float32 on CPU; this runner does not use FP16 loss scaling")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError(f"Use a new output directory or --resume: {out}")
    out.mkdir(parents=True, exist_ok=True)
    rows, calibration = read_jsonl(args.data), read_jsonl(args.calibration)
    if not rows or not calibration:
        raise ValueError("Training and calibration sets must be nonempty")
    if {r["prompt_sha256"] for r in rows} & {r["prompt_sha256"] for r in calibration}:
        raise ValueError("Training/calibration overlap")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    student_tok = AutoTokenizer.from_pretrained(args.student, revision=args.student_revision)
    teacher_tok = AutoTokenizer.from_pretrained(args.teacher, revision=args.teacher_revision)
    for tok in (student_tok, teacher_tok):
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "right"
    require_shared_tokenizer(student_tok, teacher_tok)
    eligible, excluded = [], []
    for row in rows:
        lengths = [len(tok.encode(render_prompt(tok, row["prompt"]), add_special_tokens=False))
                   for tok in (student_tok, teacher_tok)]
        if max(lengths) > args.max_prompt_length:
            excluded.append({"id": row["id"], "reason": "prompt_too_long", "lengths": lengths})
        else:
            eligible.append(row)
    if not eligible:
        raise ValueError("No prompts fit --max-prompt-length")
    rows = eligible
    write_jsonl(out / "excluded_samples.jsonl", excluded)
    dtype = getattr(torch, args.dtype)
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, revision=args.teacher_revision,
                                torch_dtype=dtype).to(args.device).eval().requires_grad_(False)
    student = AutoModelForCausalLM.from_pretrained(args.student, revision=args.student_revision,
                                torch_dtype=dtype).to(args.device)
    student.config.use_cache = False
    valid_token_ids = sorted(set(student_tok.get_vocab().values()))
    shared_ids = torch.tensor(valid_token_ids, dtype=torch.long, device=args.device)
    s_vocab = student.get_output_embeddings().weight.shape[0]
    t_vocab = teacher.get_output_embeddings().weight.shape[0]
    if valid_token_ids[-1] >= min(s_vocab, t_vocab):
        raise ValueError("A shared tokenizer ID is missing from one model's output head")
    suppressed_ids = sorted(set(range(s_vocab)) - set(valid_token_ids))
    if args.lora_rank:
        from peft import LoraConfig, get_peft_model
        student = get_peft_model(student, LoraConfig(task_type="CAUSAL_LM", r=args.lora_rank,
                    lora_alpha=args.lora_rank * 2, lora_dropout=0.0, target_modules="all-linear"))
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.enable_input_require_grads()
    projection, bi_scores = None, None
    bi = None
    if args.method == "opd_impact":
        if args.bi_summary:
            bi = json.loads(Path(args.bi_summary).read_text())
            if bi["metadata"]["model"] != args.teacher or bi["metadata"].get("revision") != args.teacher_revision:
                raise ValueError("BI summary is from a different teacher/revision")
            if bi["metadata"]["sample_ids"] != [r["id"] for r in calibration]:
                raise ValueError("BI summary calibration sample IDs differ")
            if bi["metadata"]["max_length"] != args.bi_max_length or bi["metadata"]["text_mode"] != "prompt":
                raise ValueError("BI summary uses different calibration length/text mode")
            if bi["metadata"].get("resolved_commit") != getattr(teacher.config, "_commit_hash", None):
                raise ValueError("BI summary teacher commit differs")
        else:
            bi = collect(teacher, teacher_tok, calibration, out / "bi", args.teacher,
                         max_length=args.bi_max_length, top_k=args.top_k,
                         tau=args.bi_tau, seed=args.seed, revision=args.teacher_revision)
        bi_scores = torch.tensor([r["bi_mean"] for r in bi["layers"]], device=args.device)
        from .projection import compute_vocab_projection
        projection = nn.Parameter(compute_vocab_projection(teacher, student, teacher_tok,
                       student_tok, lambda_reg=1.0, max_overlap=args.projector_vocab).to(args.device))
    groups = [{"params": [p for p in student.parameters() if p.requires_grad], "lr": args.lr}]
    if projection is not None:
        groups.append({"params": [projection], "lr": args.projector_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    pointer, start_step = 0, 0
    if args.resume:
        state = torch.load(Path(args.resume) / "training_state.pt", map_location=args.device, weights_only=False)
        if state.get("objective_family") != "direct_opd":
            raise ValueError("Cannot resume an ETA/dual-space checkpoint as direct OPD")
        student.load_state_dict(state["student"], strict=False)
        if projection is not None:
            projection.data.copy_(state["projection"])
        optimizer.load_state_dict(state["optimizer"])
        start_step, pointer, order = state["step"], state["pointer"], state["order"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if torch.cuda.is_available() and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
    config = {**vars(args), "data_sha256": digest(Path(args.data).read_text()),
              "calibration_sha256": digest(Path(args.calibration).read_text()),
              "teacher_commit": getattr(teacher.config, "_commit_hash", None),
              "student_commit": getattr(student.config, "_commit_hash", None),
              "objective_family": "direct_opd", "checkpoint_schema": 2, "torch_version": torch.__version__,
              "probability_support": "all_actual_shared_tokenizer_ids; unassigned LM-head padding excluded",
              "tokenizer_vocab_size": len(valid_token_ids), "student_head_size": s_vocab, "teacher_head_size": t_vocab,
              "eligible_samples": len(rows), "excluded_samples": len(excluded),
              "loss": "(1-kd_rate)*reference_CE + kd_rate*direct_full_vocab_OPD + optional lambda*response-prefix_IMPACT"}
    if args.resume:
        old = json.loads((Path(args.resume).parent / "config.json").read_text())
        for key in ("objective_family", "method", "student", "teacher", "student_revision", "teacher_revision",
                    "data_sha256", "calibration_sha256", "teacher_commit", "student_commit", "seed",
                    "grad_accum", "kd_objective", "kd_rate", "kd_temperature", "selection", "weighting",
                    "impact_lambda", "top_k", "bi_tau", "bi_max_length", "lora_rank", "lr",
                    "projector_lr", "projector_vocab", "max_length", "max_prompt_length",
                    "max_new_tokens", "temperature", "top_p", "dtype"):
            if old.get(key) != config[key]:
                raise ValueError(f"Resume config mismatch: {key}")
    save_json(out / "config.json", config)
    write_jsonl(out / "training_samples.jsonl", [rows[i] for i in order])

    def checkpoint(step):
        path = out / f"checkpoint-{step}"
        path.mkdir(exist_ok=True)
        student.save_pretrained(path)
        student_tok.save_pretrained(path)
        student_state = ({k: v for k, v in student.state_dict().items() if "lora_" in k}
                         if args.lora_rank else student.state_dict())
        torch.save({"schema_version": 2, "objective_family": "direct_opd", "student": student_state,
                    "projection": projection.detach() if projection is not None else None,
                    "optimizer": optimizer.state_dict(), "step": step, "pointer": pointer,
                    "order": order, "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                   path / "training_state.pt")
        save_json(out / "latest.json", {"checkpoint": str(path.resolve()), "step": step})

    prior_seconds, prior_tokens = 0.0, 0
    history_path = out / "metrics.jsonl"
    if args.resume and history_path.exists():
        history = read_jsonl(history_path)
        if history:
            prior_seconds = history[-1].get("cumulative_loop_seconds", 0.0)
            prior_tokens = history[-1].get("cumulative_generated_tokens", 0)
    begin, total_tokens, step = time.monotonic(), 0, start_step
    try:
        with (out / "metrics.jsonl").open("a") as metrics, (out / "rollouts.jsonl").open("a") as rollouts:
            for step in range(start_step + 1, args.steps + 1):
                optimizer.zero_grad(set_to_none=True)
                sums = {}
                for micro in range(args.grad_accum):
                    row = rows[order[pointer % len(order)]]
                    pointer += 1
                    prompt = render_prompt(student_tok, row["prompt"])
                    enc = student_tok(prompt, add_special_tokens=False, return_tensors="pt").to(args.device)
                    if enc["input_ids"].shape[1] > args.max_prompt_length:
                        raise ValueError(f"Prompt {row['id']} exceeds --max-prompt-length (no silent truncation)")
                    student.eval()
                    for attempt in range(args.rollout_retries):
                        with torch.no_grad():
                            generated = student.generate(**enc, max_new_tokens=args.max_new_tokens,
                                do_sample=True, temperature=args.temperature, top_p=args.top_p, top_k=0,
                                eos_token_id=student_tok.eos_token_id, pad_token_id=student_tok.pad_token_id,
                                suppress_tokens=suppressed_ids,
                                use_cache=True)[0, enc["input_ids"].shape[1]:].tolist()
                        response = student_tok.decode(generated, skip_special_tokens=True,
                                                       clean_up_tokenization_spaces=False)
                        try:
                            batch = build_direct_batch(student_tok, [row["prompt"]], [generated],
                                                       args.device, args.max_length)
                            break
                        except ValueError as exc:
                            rollouts.write(json.dumps({"step": step, "micro": micro, "sample_id": row["id"],
                                "rejected": True, "attempt": attempt, "reason": str(exc),
                                "student_token_ids": generated, "response": response}) + "\n")
                    else:
                        raise ValueError(f"No usable rollout after {args.rollout_retries} attempts: {row['id']}")
                    student.train()
                    ce = torch.tensor(0., device=args.device)
                    if args.kd_rate < 1:
                        ce = supervised_loss(student, student_tok, row, args.device, args.max_length)
                        if not torch.isfinite(ce):
                            raise FloatingPointError("Nonfinite supervised loss")
                        # Release the reference forward graph before teacher scoring.
                        ((1 - args.kd_rate) * ce / args.grad_accum).backward()
                        ce = ce.detach()
                    so = student(**batch["op_input_batch"], output_hidden_states=projection is not None, use_cache=False)
                    with torch.no_grad():
                        to = teacher(**batch["op_teacher_input_batch"], output_hidden_states=projection is not None, use_cache=False)
                    kd = direct_opd_loss(so.logits, to.logits, batch["op_label_batch"]["label"],
                                         args.kd_objective, args.kd_temperature, shared_ids)
                    log = {"opd_loss": kd}
                    log["nll_loss"] = ce
                    loss = args.kd_rate * kd
                    if projection is not None:
                        impact = response_alignment_loss(so.hidden_states, to.hidden_states, batch,
                              student_tok, teacher_tok, projection, bi_scores, args.top_k, args.bi_tau,
                              args.selection, args.weighting, args.seed, allow_empty=True)
                        loss = loss + args.impact_lambda * impact
                        log["impact_loss"] = impact
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite loss at step {step}")
                    (loss / args.grad_accum).backward()
                    total_tokens += len(generated)
                    record = {"step": step, "micro": micro, "sample_id": row["id"],
                              "domain": row["domain"], "prompt": row["prompt"], "response": response,
                              "student_token_ids": generated, "n_tokens": len(generated),
                              "ended_with_eos": bool(generated and generated[-1] == student_tok.eos_token_id)}
                    rollouts.write(json.dumps(record, ensure_ascii=False) + "\n")
                    for key, value in {**log, "loss": loss + (1 - args.kd_rate) * ce}.items():
                        sums[key] = sums.get(key, 0.0) + float(value.detach()) / args.grad_accum
                    del so, to, loss, kd, log, batch
                params = [p for g in optimizer.param_groups for p in g["params"]]
                sums["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
                optimizer.step()
                record = {"step": step, **sums, "elapsed_seconds": time.monotonic() - begin,
                          "generated_tokens_this_session": total_tokens,
                          "cumulative_loop_seconds": prior_seconds + time.monotonic() - begin,
                          "cumulative_generated_tokens": prior_tokens + total_tokens,
                          "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None}
                metrics.write(json.dumps(record, allow_nan=False) + "\n")
                metrics.flush()
                rollouts.flush()
                print(json.dumps(record), flush=True)
                if step % args.save_every == 0:
                    checkpoint(step)
                if args.max_minutes and time.monotonic() - begin >= args.max_minutes * 60:
                    break
        checkpoint(step)
        save_json(out / "status.json", {"state": "completed" if step >= args.steps else "time_budget_reached",
                                        "step": step, "elapsed_seconds": time.monotonic() - begin})
    except Exception as exc:
        save_json(out / "status.json", {"state": "failed", "step": step, "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
