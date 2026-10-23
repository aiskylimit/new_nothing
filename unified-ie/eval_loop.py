"""Evaluation shared by finetune.py and span_finetune.py (same idea as new_nothing/OpenED/ced_eval.py).

evaluate() makes a loss pass and, with --eval-gen, a generation pass over one split:
  - the loss pass computes logits only over the label window (label_window), the only
    positions the loss reads;
  - answers come from model.generate() (--gen-backend hf) or from vLLM (--gen-backend vllm,
    gen_backend.py), decoded and scored with the same code either way.
It returns (average loss, score). The score is the dev number the trainers select on: for event
rows the mean of per-dataset trigger F1 with --label-map (else the pooled trigger F1), for relation
rows (task "re") the mean of per-dataset triple F1.
"""
import json
import os
import shutil

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import GenerationConfig

from data_utils.lm_datasets import label_window
from ed_eval import ed_evaluate, ed_evaluate_by_dataset, format_by_dataset, load_label_map, re_evaluate_by_dataset
from gen_backend import (export_for_vllm, find_vllm_python, resolve_generation_config, row_seed, run_vllm,
                         unpadded_prompts, vllm_params)
from rouge_metric import compute_metrics
from utils import all_gather, get_rank, print_rank, save_rank


def eval_generation_config(args, tokenizer):
    return GenerationConfig(
        do_sample=args.do_sample,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        max_length=args.max_length,
        min_length=None,
        eos_token_id=[tokenizer.eos_token_id, 151643],
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=False
    )


def generation_kwargs(args):
    # transformers >= 4.50 fills config fields left at their defaults from Qwen3's generation_config.json
    # (do_sample=True), so do_sample goes in as a kwarg: without --do-sample decoding really is greedy
    return {"do_sample": args.do_sample}


def check_gen_backend(args, model, tokenizer):
    """With --gen-backend vllm, refuse to start a run whose evaluation vLLM could not do."""
    if args.gen_backend != "vllm":
        return
    if dist.get_world_size() != 1:
        raise ValueError("--gen-backend vllm evaluates in one process: run with world size 1")
    args.vllm_py, args.vllm_version = find_vllm_python()
    vllm_params(resolve_generation_config(model, eval_generation_config(args, tokenizer), **generation_kwargs(args)))
    print_rank(f"generation backend: vLLM {args.vllm_version} ({args.vllm_py})")


def _loader(args, dataset):
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False,
                                 rank=dist.get_rank(), num_replicas=dist.get_world_size())
    return DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size,
                      num_workers=args.num_workers, collate_fn=dataset.collate)


def _vllm_responses(args, tokenizer, model, loader, generation_config, split, epoch):
    """Each row's answer from vLLM, with the settings generate() would decode with."""
    requests = []
    for _, _, gen_data, _, _ in loader:
        width = gen_data["input_ids"].size(1)
        for prompt in unpadded_prompts(gen_data["input_ids"], gen_data["attention_mask"]):
            requests.append({"prompt_token_ids": prompt, "max_tokens": args.max_length - width,
                             "seed": row_seed(args.seed, split, epoch, len(requests))})
    params = vllm_params(resolve_generation_config(model, generation_config, **generation_kwargs(args)))
    work_dir = os.path.join(args.save, "vllm_tmp")
    model_dir, lora_dir, rank = export_for_vllm(model, args.model_path, work_dir)
    outputs = run_vllm(work_dir, model_dir, requests, params, lora_dir=lora_dir, max_lora_rank=rank,
                       max_model_len=args.max_length, vllm_py=getattr(args, "vllm_py", None))
    shutil.rmtree(work_dir, ignore_errors=True)
    return tokenizer.batch_decode([o["token_ids"] for o in outputs], skip_special_tokens=True)


def _hf_responses(args, tokenizer, model, loader, generation_config, device):
    dp_world_size = dist.get_world_size()
    all_response_ids = []
    with torch.no_grad():
        for it, (_, _, gen_data, _, _) in enumerate(tqdm(loader, desc="Generating", disable=(dist.get_rank() != 0))):
            print_rank(f"{it}/{len(loader)}")
            gen_data = {k: v.to(device) for k, v in gen_data.items()}
            width = gen_data["input_ids"].size(1)
            full_ids = model.generate(**gen_data, generation_config=generation_config,
                                      max_new_tokens=args.max_length - width, **generation_kwargs(args)).sequences
            full_ids = F.pad(full_ids, (0, args.max_length - full_ids.shape[1]), value=tokenizer.pad_token_id)
            all_response_ids.append(full_ids[:, width:])
    all_response_ids = torch.cat(all_response_ids, dim=0)
    all_response_ids = all_gather(all_response_ids, dim=1, world_size=dp_world_size, group=None, op="stack")
    all_response_ids = all_response_ids.view(-1, all_response_ids.size(-1))
    return tokenizer.batch_decode(all_response_ids, skip_special_tokens=True)


def evaluate(args, tokenizer, model, dataset, split, epoch, device, adaptive_threshold=None):
    dp_world_size = dist.get_world_size()
    loss_func = nn.CrossEntropyLoss()
    print_rank("dp size", dp_world_size)
    model.eval()
    loader = _loader(args, dataset)

    all_loss, step = 0.0, 0
    with torch.no_grad():
        for model_batch, no_model_batch, gen_data, _, _ in loader:
            dataset.move_to_device(model_batch, no_model_batch, gen_data, device)
            label = no_model_batch["label"]
            start, end = label_window(label)
            logits = model(**model_batch, use_cache=False,
                           logits_to_keep=torch.arange(start, end, device=label.device)).logits
            loss = loss_func(logits.reshape(-1, logits.shape[-1]), label[:, start:end].reshape(-1))
            dist.all_reduce(loss, dist.ReduceOp.SUM)
            all_loss += (loss / dp_world_size).item()
            step += 1

    responses = None
    if args.eval_gen:
        generation_config = eval_generation_config(args, tokenizer)
        if args.gen_backend == "vllm":
            responses = _vllm_responses(args, tokenizer, model, loader, generation_config, split, epoch)
        else:
            responses = _hf_responses(args, tokenizer, model, loader, generation_config, device)

    score = torch.tensor([-all_loss / step], device=device)
    if get_rank() == 0:
        res = {}
        if responses is not None:
            references = dataset.answers
            responses = responses[:len(references)]
            res = compute_metrics(responses, references)
            if dataset.raw[0].get("task") == "re":
                res["by_dataset"] = re_evaluate_by_dataset(responses, dataset.raw)
                score[0] = res["by_dataset"]["macro_triple_f1"]
                summary = format_by_dataset(res["by_dataset"], split, epoch)
                print_rank(summary)
                save_rank(summary, os.path.join(args.save, "log.txt"))
            else:
                res.update(ed_evaluate(responses, references))
                score[0] = res["trigger"]["f1"]
            if args.label_map is not None and dataset.raw[0].get("task") != "re":
                res["by_dataset"] = ed_evaluate_by_dataset(responses, dataset.raw, load_label_map(args.label_map))
                score[0] = res["by_dataset"]["macro_trigger_f1"]
                summary = format_by_dataset(res["by_dataset"], split, epoch)
                print_rank(summary)
                save_rank(summary, os.path.join(args.save, "log.txt"))
            eval_dir = os.path.join(args.save, "eval", str(epoch))
            os.makedirs(eval_dir, exist_ok=True)
            with open(os.path.join(eval_dir, f"{split}_answers.jsonl"), "w") as f:
                for resp in responses:
                    f.write(json.dumps({"text": resp}) + "\n")
        avg_loss = all_loss / step
        if "adaptive" in args.type:
            log_str = f"{split} | avg_loss: {avg_loss} | {res} | threshold: {adaptive_threshold}"
        else:
            log_str = f"{split} | avg_loss: {avg_loss} | {res}"
        print_rank(log_str)
        save_rank(log_str, os.path.join(args.save, "log.txt"))
    dist.broadcast(score, 0)
    return all_loss / step, score.item()


def evaluate_dev_and_test(args, tokenizer, model, dataset, epoch, device, best, adaptive_threshold=None):
    """Dev every time; test every time (--eval-gen-mode every) or only when the dev score beats the best
    so far (best). Returns (dev loss, new best score)."""
    dev_loss, dev_score = evaluate(args, tokenizer, model, dataset["dev"], "dev", epoch, device, adaptive_threshold)
    improved = dev_score > best
    if improved:
        best = dev_score
        line = f"BEST dev epoch {epoch} | score {100 * dev_score:.2f}"
        print_rank(line)
        save_rank(line, os.path.join(args.save, "log.txt"))
    if args.eval_gen_mode == "every" or improved:
        evaluate(args, tokenizer, model, dataset["test"], "test", epoch, device)
    return dev_loss, best
