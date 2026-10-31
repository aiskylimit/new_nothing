"""Single-GPU comparison of direct OPD versus direct OPD + IMPACT.

The student samples fresh responses before each microbatch. Both methods use
the same objective, seed, prompts, optimizer, rollout budget, and checkpoints.
"""
import argparse
import json
import math
import random
import time
from collections import deque
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .recipes import TEACHER, STUDENT, UPSTREAM_COMMIT, RECIPES, apply_recipe, adapter_signature, load_teacher_adapter
from .alignment import response_alignment_loss
from .losses import require_shared_tokenizer, direct_opd_loss, build_direct_batch
from .bi import collect, render_prompt
from .data import digest, read_jsonl, save_json, write_jsonl
from .batching import rollout_inputs, trim_generated, supervised_batch_loss
from .resume import rewind_log
from .timing import deadline_timestamp, estimate_timing



def make_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--student", default=STUDENT)
    p.add_argument("--teacher", default=TEACHER)
    p.add_argument("--preset", choices=list(RECIPES), default="dskd-tiny")
    p.add_argument("--teacher-adapter", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--scheduler", choices=["constant", "cosine"], default=None)
    p.add_argument("--lora-alpha", type=int, default=None)
    p.add_argument("--lora-dropout", type=float, default=None)
    p.add_argument("--lora-targets", default=None)
    p.add_argument("--temperature-squared", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--truncate-prompts", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--student-revision", default=None)
    p.add_argument("--teacher-revision", default=None)
    p.add_argument("--method", choices=["opd", "opd_impact"], required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--bi-summary", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--steps", type=int, default=None, help="Optimizer updates")
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--attention-backend", choices=["sdpa", "eager"], default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--projector-lr", type=float, default=None)
    p.add_argument("--projector-vocab", type=int, default=4096)
    p.add_argument("--lora-rank", type=int, default=None, help="0 for full training")
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--max-prompt-length", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--rollout-retries", type=int, default=5)
    p.add_argument("--bi-max-length", type=int, default=None)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--kd-temperature", type=float, default=None)
    p.add_argument("--kd-rate", type=float, default=None, help="1 = pure OPD; values below 1 explicitly add reference CE")
    p.add_argument("--kd-objective", choices=["forward_kl", "reverse_kl", "js_divergence"], default=None)
    p.add_argument("--impact-lambda", type=float, default=0.25)
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--bi-tau", type=float, default=2.0)
    p.add_argument("--selection", choices=["bi", "first", "last", "random"], default="bi")
    p.add_argument("--weighting", choices=["bi", "uniform"], default="bi")
    p.add_argument("--save-every", type=int, default=64)
    p.add_argument("--max-minutes", type=float, default=0, help="0 = no deadline; saves at next update boundary")
    p.add_argument("--stop-at", default=None, help="Stop at an update boundary and save; timezone-aware ISO time")
    p.add_argument("--resume", default=None, help="Checkpoint directory, including optimizer/RNG state")
    return p


def supervised_loss(student, tokenizer, row, device, max_length, max_prompt_length=None):
    return supervised_batch_loss(student, tokenizer, [row], device, max_length, max_prompt_length)


def check_resume_batch(old, new):
    before = old.get("micro_batch_size", 1) * old["grad_accum"]
    after = new["micro_batch_size"] * new["grad_accum"]
    if before != after:
        raise ValueError(f"Resume effective batch changed: {before} -> {after}")


def main():
    args = apply_recipe(make_parser().parse_args())
    deadline = deadline_timestamp(args.stop_at)
    if deadline is not None and deadline <= time.time():
        raise ValueError("stop-at is in the past; provide a new deadline or omit it to continue")
    if args.steps is None and not args.epochs:
        args.steps = 256
    if min(args.steps if args.steps is not None else 1, args.grad_accum, args.micro_batch_size, args.save_every, args.projector_vocab, args.max_new_tokens,
           args.rollout_retries, args.max_length, args.max_prompt_length, args.top_k) < 1:
        raise ValueError("Step, length, vocabulary and accumulation budgets must be positive")
    if args.temperature <= 0 or not 0 < args.top_p <= 1 or args.bi_tau <= 0 or args.lora_rank < 0:
        raise ValueError("Invalid sampling, BI temperature or LoRA rank")
    if not 0 < args.kd_rate <= 1:
        raise ValueError("kd_rate must be in (0, 1]")
    if args.max_prompt_length + args.max_new_tokens > args.max_length:
        raise ValueError("max_prompt_length + max_new_tokens must fit max_length")
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
        lengths = [len(tok.encode(render_prompt(tok, row["prompt"], row.get("prompt_format", "auto")), add_special_tokens=False))
                   for tok in (student_tok, teacher_tok)]
        if max(lengths) > args.max_prompt_length and not args.truncate_prompts:
            excluded.append({"id": row["id"], "reason": "prompt_too_long", "lengths": lengths})
        else:
            eligible.append(row)
    if not eligible:
        raise ValueError("No prompts fit --max-prompt-length")
    rows = eligible
    effective_batch = args.micro_batch_size * args.grad_accum
    updates_per_epoch = len(rows) // effective_batch
    if args.epochs and updates_per_epoch < 1:
        raise ValueError("Need at least one full effective batch per epoch")
    epoch_steps = updates_per_epoch * args.epochs
    if args.steps is None:
        args.steps = epoch_steps
    if args.epochs and args.steps > epoch_steps:
        raise ValueError("--steps exceeds --epochs budget; increase epochs explicitly")
    write_jsonl(out / "excluded_samples.jsonl", excluded)
    dtype = getattr(torch, args.dtype)
    model_kwargs = {"attn_implementation": args.attention_backend} if args.attention_backend else {}
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, revision=args.teacher_revision,
                                torch_dtype=dtype, **model_kwargs).to(args.device).eval().requires_grad_(False)
    teacher = load_teacher_adapter(teacher, args.teacher_adapter)
    teacher_adapter_sha256 = adapter_signature(args.teacher_adapter)
    student = AutoModelForCausalLM.from_pretrained(args.student, revision=args.student_revision,
                                torch_dtype=dtype, **model_kwargs).to(args.device)
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
                    lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                    target_modules="all-linear" if args.lora_targets == "all-linear" else args.lora_targets.split(",")))
    if args.gradient_checkpointing:
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        student.enable_input_require_grads()
    projection, bi_scores = None, None
    bi = None
    if args.method == "opd_impact":
        if args.bi_summary:
            bi = json.loads(Path(args.bi_summary).read_text())
            if bi["metadata"].get("adapter_sha256") != teacher_adapter_sha256:
                raise ValueError("BI summary teacher adapter differs")
            if bi["metadata"]["model"] != args.teacher or bi["metadata"].get("revision") != args.teacher_revision:
                raise ValueError("BI summary is from a different teacher/revision")
            expected_inputs = [teacher_tok.encode(render_prompt(teacher_tok, r["prompt"],
                r.get("prompt_format", "auto")), add_special_tokens=False)[:args.bi_max_length] for r in calibration]
            expected_hash = digest("\n".join(digest(str(ids)) for ids in expected_inputs))
            if bi["metadata"]["inputs_sha256"] != expected_hash:
                raise ValueError("BI summary prompt/token inputs differ")
            if bi["metadata"]["sample_ids"] != [r["id"] for r in calibration]:
                raise ValueError("BI summary calibration sample IDs differ")
            if bi["metadata"]["max_length"] != args.bi_max_length or bi["metadata"]["text_mode"] != "prompt":
                raise ValueError("BI summary uses different calibration length/text mode")
            if bi["metadata"].get("resolved_commit") != getattr(teacher.config, "_commit_hash", None):
                raise ValueError("BI summary teacher commit differs")
        else:
            bi = collect(teacher, teacher_tok, calibration, out / "bi", args.teacher,
                         max_length=args.bi_max_length, top_k=args.top_k,
                         tau=args.bi_tau, seed=args.seed, revision=args.teacher_revision,
                         adapter_sha256=teacher_adapter_sha256)
        bi_scores = torch.tensor([r["bi_mean"] for r in bi["layers"]], device=args.device)
        from .projection import compute_vocab_projection
        projection = nn.Parameter(compute_vocab_projection(teacher, student, teacher_tok,
                       student_tok, lambda_reg=1.0, max_overlap=args.projector_vocab).to(args.device))
    groups = [{"params": [p for p in student.parameters() if p.requires_grad], "lr": args.lr}]
    if projection is not None:
        groups.append({"params": [projection], "lr": args.projector_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda step: .5 * (1 + math.cos(math.pi * min(step, args.steps) / args.steps))
        if args.scheduler == "cosine" else 1.)
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
        scheduler.load_state_dict(state["scheduler"])
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
              "reference_commit": UPSTREAM_COMMIT if args.preset == "dskd-tiny" else None,
              "bi_summary_sha256": digest(Path(args.bi_summary).read_text()) if args.bi_summary else None,
              "objective_family": "direct_opd", "checkpoint_schema": 4, "torch_version": torch.__version__,
              "effective_batch_size": effective_batch, "loss_reduction": "mean_of_per_sample_means",
              "impact_implementation": "vectorized_response_deltas_v1",
              "probability_support": "all_actual_shared_tokenizer_ids; unassigned LM-head padding excluded",
              "tokenizer_vocab_size": len(valid_token_ids), "student_head_size": s_vocab, "teacher_head_size": t_vocab,
              "teacher_adapter_sha256": teacher_adapter_sha256, "updates_per_epoch": updates_per_epoch,
              "rollout_schedule": "fresh student response from first update; no off-policy warmup",
              "eligible_samples": len(rows), "excluded_samples": len(excluded),
              "loss": "(1-kd_rate)*reference_CE + kd_rate*direct_full_vocab_OPD + optional lambda*response-prefix_IMPACT"}
    if args.resume:
        config_path = Path(args.resume) / "run_config.json"
        if not config_path.exists():
            config_path = Path(args.resume).parent / "config.json"
        old = json.loads(config_path.read_text())
        check_resume_batch(old, config)
        if args.scheduler == "cosine" and old["steps"] != args.steps:
            raise ValueError("Cosine resume requires the original total step budget")
        for key in ("objective_family", "method", "student", "teacher", "student_revision", "teacher_revision",
                    "data_sha256", "calibration_sha256", "teacher_commit", "student_commit", "seed",
                    "preset", "epochs", "scheduler", "bi_summary_sha256", "lora_alpha", "lora_dropout", "lora_targets",
                    "temperature_squared", "truncate_prompts", "teacher_adapter_sha256",
                    "kd_objective", "kd_rate", "kd_temperature", "selection", "weighting",
                    "impact_lambda", "top_k", "bi_tau", "bi_max_length", "lora_rank", "lr",
                    "projector_lr", "projector_vocab", "max_length", "max_prompt_length",
                    "max_new_tokens", "temperature", "top_p", "dtype"):
            if old.get(key) != config[key]:
                raise ValueError(f"Resume config mismatch: {key}")
        resume_id = f"step-{start_step}-{time.time_ns()}"
        save_json(out / "resume_history" / f"{resume_id}.json",
                  {"checkpoint": args.resume, "old_config": old, "new_config": config,
                   "note": "Batch regrouping preserves effective batch and sample weighting, not bitwise RNG/dropout trajectories."})
        for filename in ("metrics.jsonl", "rollouts.jsonl"):
            rewind_log(out / filename, start_step, out / "resume_history" / resume_id)
    save_json(out / "config.json", config)
    write_jsonl(out / "training_samples.jsonl", [rows[i] for i in order])

    def checkpoint(step):
        path = out / f"checkpoint-{step}"
        path.mkdir(exist_ok=True)
        student.save_pretrained(path)
        student_tok.save_pretrained(path)
        save_json(path / "run_config.json", config)
        student_state = ({k: v for k, v in student.state_dict().items() if "lora_" in k}
                         if args.lora_rank else student.state_dict())
        torch.save({"schema_version": 4, "objective_family": "direct_opd", "student": student_state,
                    "projection": projection.detach() if projection is not None else None,
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "step": step, "pointer": pointer,
                    "order": order, "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                   path / "training_state.pt")
        save_json(out / "latest.json", {"checkpoint": str(path.resolve()), "step": step})

    prior_seconds, prior_tokens = 0.0, 0
    history_path = out / "metrics.jsonl"
    if args.resume and history_path.exists():
        history = [r for r in read_jsonl(history_path) if r["step"] <= start_step]
        if history:
            prior_seconds = history[-1].get("cumulative_loop_seconds", 0.0)
            prior_tokens = history[-1].get("cumulative_generated_tokens", 0)
    begin, total_tokens, step = time.monotonic(), 0, start_step
    durations = deque(maxlen=50)
    stop_reason = "time_budget_reached"
    try:
        with (out / "metrics.jsonl").open("a") as metrics, (out / "rollouts.jsonl").open("a") as rollouts:
            for next_step in range(start_step + 1, args.steps + 1):
                if deadline is not None and time.time() >= deadline:
                    stop_reason = "deadline_reached"
                    break
                step = next_step
                step_begin, step_tokens, rollout_seconds, learning_seconds = time.monotonic(), 0, 0., 0.
                if args.epochs and (step - 1) % updates_per_epoch == 0:
                    epoch = (step - 1) // updates_per_epoch
                    order = list(range(len(rows)))
                    random.Random(args.seed + epoch).shuffle(order)
                    pointer = 0
                optimizer.zero_grad(set_to_none=True)
                sums = {}
                for micro in range(args.grad_accum):
                    micro_rows = [rows[order[(pointer + i) % len(order)]] for i in range(args.micro_batch_size)]
                    pointer += args.micro_batch_size
                    enc = rollout_inputs(student_tok, micro_rows, args.device,
                                         args.max_prompt_length if args.truncate_prompts else None)
                    if enc["input_ids"].shape[1] > args.max_prompt_length:
                        raise ValueError("Prompt exceeds --max-prompt-length (no silent truncation)")
                    student.eval()
                    rollout_begin = time.monotonic()
                    for attempt in range(args.rollout_retries):
                        with torch.no_grad():
                            generated_batch = student.generate(**enc, max_new_tokens=args.max_new_tokens,
                                do_sample=True, temperature=args.temperature, top_p=args.top_p, top_k=0,
                                eos_token_id=student_tok.eos_token_id, pad_token_id=student_tok.pad_token_id,
                                suppress_tokens=suppressed_ids,
                                use_cache=True)[:, enc["input_ids"].shape[1]:]
                        generated = trim_generated(generated_batch, student_tok.eos_token_id)
                        responses = student_tok.batch_decode(generated, skip_special_tokens=True,
                                                             clean_up_tokenization_spaces=False)
                        try:
                            batch = build_direct_batch(student_tok, [r["prompt"] for r in micro_rows], generated,
                                                       args.device, args.max_length,
                                                       prompt_format=[r.get("prompt_format", "auto") for r in micro_rows],
                                                       max_prompt_length=args.max_prompt_length if args.truncate_prompts else None)
                            break
                        except ValueError as exc:
                            for i, row in enumerate(micro_rows):
                                rollouts.write(json.dumps({"step": step, "micro": micro, "batch_index": i, "sample_id": row["id"],
                                    "rejected": True, "attempt": attempt, "reason": str(exc),
                                    "student_token_ids": generated[i], "response": responses[i]}) + "\n")
                    else:
                        raise ValueError(f"No usable rollout batch after {args.rollout_retries} attempts")
                    rollout_seconds += time.monotonic() - rollout_begin
                    learning_begin = time.monotonic()
                    student.train()
                    ce = torch.tensor(0., device=args.device)
                    if args.kd_rate < 1:
                        ce = supervised_batch_loss(student, student_tok, micro_rows, args.device, args.max_length,
                                                   args.max_prompt_length if args.truncate_prompts else None)
                        if not torch.isfinite(ce):
                            raise FloatingPointError("Nonfinite supervised loss")
                        # Release the reference forward graph before teacher scoring.
                        ((1 - args.kd_rate) * ce / args.grad_accum).backward()
                        ce = ce.detach()
                    so = student(**batch["op_input_batch"], output_hidden_states=projection is not None, use_cache=False)
                    with torch.no_grad():
                        to = teacher(**batch["op_teacher_input_batch"], output_hidden_states=projection is not None, use_cache=False)
                    kd = direct_opd_loss(so.logits, to.logits, batch["op_label_batch"]["label"],
                                         args.kd_objective, args.kd_temperature, shared_ids,
                                         temperature_squared=args.temperature_squared, reduction="sample_mean")
                    log = {"opd_loss": kd}
                    log["nll_loss"] = ce
                    loss = args.kd_rate * kd
                    if projection is not None:
                        impact = response_alignment_loss(so.hidden_states, to.hidden_states, batch,
                              student_tok, student_tok, projection, bi_scores, args.top_k, args.bi_tau,
                              args.selection, args.weighting, args.seed, allow_empty=True, reduction="sample_mean")
                        loss = loss + args.impact_lambda * impact
                        log["impact_loss"] = impact
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite loss at step {step}")
                    (loss / args.grad_accum).backward()
                    tokens = sum(map(len, generated))
                    total_tokens += tokens
                    step_tokens += tokens
                    for i, row in enumerate(micro_rows):
                        ids = generated[i]
                        record = {"step": step, "micro": micro, "batch_index": i, "sample_id": row["id"],
                                  "domain": row["domain"], "prompt": row["prompt"], "response": responses[i],
                                  "student_token_ids": ids, "n_tokens": len(ids),
                                  "ended_with_eos": bool(ids and ids[-1] == student_tok.eos_token_id)}
                        rollouts.write(json.dumps(record, ensure_ascii=False) + "\n")
                    for key, value in {**log, "loss": loss + (1 - args.kd_rate) * ce}.items():
                        sums[key] = sums.get(key, 0.0) + float(value.detach()) / args.grad_accum
                    learning_seconds += time.monotonic() - learning_begin
                    del so, to, loss, kd, log, batch
                params = [p for g in optimizer.param_groups for p in g["params"]]
                sums["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
                sums["lr"] = optimizer.param_groups[0]["lr"]
                optimizer.step()
                scheduler.step()
                if str(args.device).startswith("cuda"):
                    torch.cuda.synchronize(args.device)
                step_seconds = time.monotonic() - step_begin
                durations.append(step_seconds)
                record = {"step": step, **sums, "elapsed_seconds": time.monotonic() - begin,
                          **estimate_timing(step, args.steps, durations, time.time(), deadline),
                          "step_seconds": step_seconds, "rollout_seconds": rollout_seconds,
                          "learning_seconds": learning_seconds, "samples_per_second": effective_batch / step_seconds,
                          "generated_tokens_per_second": step_tokens / step_seconds,
                          "micro_batch_size": args.micro_batch_size, "grad_accum": args.grad_accum,
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
        save_json(out / "status.json", {"state": "completed" if step >= args.steps else stop_reason,
                                        "step": step, "elapsed_seconds": time.monotonic() - begin})
    except Exception as exc:
        save_json(out / "status.json", {"state": "failed", "step": step, "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
