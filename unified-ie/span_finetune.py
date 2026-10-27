import time
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
import deepspeed

import random
import json
from tqdm import tqdm
import math
import datetime

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    GenerationConfig)

from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR

from arguments import get_args

from data_utils.lm_datasets import LMTrainDataset, label_window, BalancedSampler, balanced_quota
from data_utils.data_utils import LLMDataset
from utils import get_optimizer_params, get_optimizer_params_peft, print_args, initialize
from utils import print_rank, get_rank
from utils import save_rank
from utils import all_gather
from utils import load_parallel, save_parallel
from utils import get_tokenizer, get_model

from distillm import forward_kl, reverse_kl, js_distance, tv_distance
from distillm import skewed_forward_kl, skewed_reverse_kl, csd
from distillm import SampleGenerator, ReplayBuffer

from rouge_metric import compute_metrics

from peft import PeftModel
from eval_loop import check_gen_backend, evaluate, evaluate_dev_and_test

torch.set_num_threads(4)


def get_teacher_model(args, device):
    config = AutoConfig.from_pretrained(args.teacher_model_path)
    if args.model_parallel:
        raise NotImplementedError
    else:
        config.is_model_parallel = False
        try: model = AutoModelForCausalLM.from_pretrained(args.teacher_model_path, config=config, device_map={"": device}, torch_dtype=torch.bfloat16)
        except:
            model = AutoModelForCausalLM.from_pretrained(args.teacher_model_path, config=config, device_map={"": device}, torch_dtype=torch.float32)
            model = model.half()
        
        if args.teacher_peft_paths is not None:
            # one adapter per source dataset on one copy of the base, not merged: teacher_forward()
            # switches to the row's adapter
            for k, (name, path) in enumerate(teacher_peft_paths(args).items()):
                if k == 0:
                    model = PeftModel.from_pretrained(model, path, adapter_name=name)
                else:
                    model.load_adapter(path, adapter_name=name)
                print_rank(f"teacher adapter {name}: {path}")
        elif args.teacher_peft_path is not None:
            model = PeftModel.from_pretrained(model, args.teacher_peft_path)
            model = model.merge_and_unload()
            print("merge_and_unload")

        if dist.get_rank() == 0:
            print(' > number of parameters: {}'.format(
                sum([p.nelement() for p in model.parameters()])), flush=True)

    model.eval()

    return model


def teacher_peft_paths(args):
    """--teacher-peft-paths "maven=dir,rams=dir" -> {"maven": "dir", "rams": "dir"}"""
    paths = {}
    for item in args.teacher_peft_paths.split(","):
        name, path = item.split("=", 1)
        paths[name.strip()] = path.strip()
    return paths


def teacher_forward(args, teacher_model, model_batch, datasets, window, hidden):
    """Teacher logits over the label window, and the hidden states (index as in HF hidden_states) when
    `hidden`. With --teacher-peft-paths every row goes through the adapter of its own dataset, as in BAM
    (Clark et al. 2019): one forward per dataset in the batch, results put back in row order. Only the
    --teacher_layer_mapping layers are kept; the span loss reads no other layer and compares spans within
    a row only, so rows from different teachers can share a batch."""
    if args.teacher_peft_paths is None:
        out = teacher_model(**model_batch, output_hidden_states=hidden, use_cache=False, logits_to_keep=window)
        return out.logits, out.hidden_states

    logits, hidden_states = None, None
    for name in sorted(set(datasets)):
        rows = torch.tensor([i for i, d in enumerate(datasets) if d == name], device=window.device)
        teacher_model.set_adapter(name)
        out = teacher_model(**{k: v[rows] for k, v in model_batch.items()}, output_hidden_states=hidden,
                            use_cache=False, logits_to_keep=window)
        if logits is None:
            logits = out.logits.new_empty((len(datasets),) + out.logits.shape[1:])
            if hidden:
                hidden_states = [None] * len(out.hidden_states)
                for l in args.teacher_layer_mapping:
                    hidden_states[l] = out.hidden_states[l].new_empty((len(datasets),) + out.hidden_states[l].shape[1:])
        logits[rows] = out.logits
        if hidden:
            for l in args.teacher_layer_mapping:
                hidden_states[l][rows] = out.hidden_states[l]
    return logits, hidden_states


def get_optimizer(args, model):
    """Set up the optimizer."""

    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, DDP):
        model = model.module

    if args.peft is not None:
        param_groups = get_optimizer_params_peft(args, model)
    else:
        param_groups = get_optimizer_params(args, model)

    # Use AdamW.
    optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    print_rank(f'Optimizer = {optimizer.__class__.__name__}')
    return optimizer


def get_learning_rate_scheduler(args, optimizer):
    if args.total_iters is None:
        args.total_iters = args.train_iters_per_epoch * args.epochs
    if args.lr_decay_style == "constant":
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters)
    elif args.lr_decay_style == "cosine":
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.total_iters,
            eta_min=args.lr_min)
    elif args.lr_decay_style == "noam":
        lr_scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters,
            num_training_steps=args.total_iters,
            power=0.5)
    elif args.lr_decay_style == "wrmup_cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_ratio * args.total_iters,
            num_training_steps=args.total_iters)
    else:
        raise ValueError(f"lr_scheduler of type {args.lr_decay_style} is not supported yet.")

    return lr_scheduler


def setup_model_and_optimizer(args, ds_config, device, set_optim=True):
    # get the model
    model = get_model(args, device)
    # get the optimizer and lr_scheduler
    if set_optim:
        optimizer = get_optimizer(args, model)
        lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    else:
        optimizer, lr_scheduler = None, None
        
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_scheduler,
        mpu=None,
        config_params=ds_config
    )
    
    # get the memory usage
    print_rank("Model mem\n", torch.cuda.memory_summary())
    return model, optimizer, lr_scheduler


def prepare_dataset(args, tokenizer):
    data = {}
    rng_sample = random.Random(args.seed)
    if args.do_train:
        data["train"] = LMTrainDataset(args, tokenizer, args.data_dir, "train", args.train_num, args.train_ratio, rng_sample)
        print_rank("train num", len(data["train"]))
        data["dev"] = LMTrainDataset(args, tokenizer, args.data_dir, "valid", args.dev_num, args.dev_ratio, rng_sample)

    data["test"] = LMTrainDataset(args, tokenizer, args.data_dir, "test", args.dev_num, args.dev_ratio, rng_sample)

        
    # pre-trained dataset
    if args.do_train and args.lm_data_dir is not None:
        data["pt_train"] = LMTrainDataset(args, tokenizer, args.lm_data_dir, "train", args.train_num, args.train_ratio, rng_sample)
        print_rank("train num", len(data["pt_train"]))
    return data


def pt_loss(args, model, model_batch, no_model_batch):
    loss_mask = (no_model_batch["label"] != -100).int()
    outputs = model(**model_batch, return_dict=True, use_cache=False)
    logits = outputs.logits
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    lm_loss = loss_fn(logits.view(-1, logits.size(-1)), no_model_batch["label"].view(-1))
    return lm_loss


def get_distil_loss(args, teacher_logits, no_model_batch, logits):
    if args.model_parallel:
        raise NotImplementedError
    else:
        if "sfkl" in args.type:
            distil_loss = skewed_forward_kl(logits, teacher_logits, no_model_batch, lam=args.skew_alpha)
        elif "srkl" in args.type:
            distil_loss = skewed_reverse_kl(logits, teacher_logits, no_model_batch, lam=args.skew_alpha)
        elif "jsd" in args.type:
            distil_loss = js_distance(logits, teacher_logits, no_model_batch)
        elif "tvd" in args.type:
            distil_loss = tv_distance(logits, teacher_logits, no_model_batch)
        elif "fkl" in args.type or args.type == "kd":
            distil_loss = forward_kl(logits, teacher_logits, no_model_batch)
        elif "rkl" in args.type:
            distil_loss = reverse_kl(logits, teacher_logits, no_model_batch)
        elif "csd" in args.type:
            distil_loss = csd(logits, teacher_logits, no_model_batch)
        elif "no" in args.type:
            distil_loss = 0.0
        else:
            raise NotImplementedError
    return distil_loss


def get_teacher_lm_loss(args, tokenizer, model, teacher_model, model_batch):
    with torch.no_grad():
        t_gen_out = teacher_model.generate(
            **model_batch,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_length=args.max_length,
            top_k=0,
            top_p=1,
            temperature=1.0,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=False)
    
    full_ids = t_gen_out.sequences
    
    input_ids = full_ids[:, :-1]
    mask = (input_ids != tokenizer.pad_token_id).long()
    labels = full_ids[:, 1:]    
    labels = torch.masked_fill(labels, mask==0, -100)
    labels[:, :model_batch["input_ids"].size(1)-1] = -100
    loss_mask = (labels != -100).float()
    
    new_batch = {
        "input_ids": input_ids,
        "attention_mask": mask,
    }
    
    if args.model_type in ["gpt2"]:
        position_ids = torch.cumsum(mask, dim=-1) - 1
        position_ids = torch.masked_fill(position_ids, mask==0, 0)    
        new_batch["position_ids"] = position_ids    
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    outputs = model(**new_batch, return_dict=True, use_cache=False)
    logits = outputs.logits
    lm_loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

    return lm_loss


def compute_token_weights(hidden_state, attention_mask):
    std = hidden_state.std(dim=-1, keepdim=True) + 1e-5
    Q = hidden_state / std
    K = hidden_state / std
    scores = torch.matmul(Q, K.transpose(-1, -2)) / (hidden_state.size(-1) ** 0.5)

    mask = attention_mask.unsqueeze(1).expand(-1, scores.size(-2), -1)
    scores = scores.masked_fill(mask == 0, float('-inf'))
    diag_mask = torch.eye(scores.size(-1), device=scores.device, dtype=torch.bool)
    scores = scores.masked_fill(diag_mask.unsqueeze(0), float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)  # [1, L, L]
    attn_weights = attn_weights * mask
    attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)

    # w_t = (1/N) sum_s alpha_{s->t} over the N real tokens, as in the EventKD paper. The original
    # attn_weights.mean(dim=1) also averaged the padding query rows (~2/3 of a 768-wide row), so the
    # weights were driven by the padding and changed with the pad width.
    query_mask = attention_mask.unsqueeze(-1).to(attn_weights.dtype)  # [B, L, 1]
    token_weights = (attn_weights * query_mask).sum(dim=1) / query_mask.sum(dim=1).clamp(min=1)  # [B, L]
    return token_weights.detach()

def prepare_span_indices_and_weights(t_layer_weights, s_layer_weights, 
                                     attention_mask, offsets_mapping, spans_offsets):
    device = attention_mask.device
    B_size, SeqLen = attention_mask.shape

    max_spans = max(len(s) for s in spans_offsets)
    if max_spans == 0:
        print(f"No spans found in the batch.")
        return None, None, None, None, None, None

    # (B_size, max_spans)
    padded_span_starts = torch.zeros(B_size, max_spans, dtype=torch.long, device=device)
    padded_span_ends = torch.zeros(B_size, max_spans, dtype=torch.long, device=device)
    padded_span_mask = torch.zeros(B_size, max_spans, dtype=torch.bool, device=device)

    for i in range(B_size):
        num_spans_i = len(spans_offsets[i])
        if num_spans_i > 0:
            spans_i = torch.tensor(spans_offsets[i], device=device, dtype=torch.long)
            padded_span_starts[i, :num_spans_i] = spans_i[:, 0]
            padded_span_ends[i, :num_spans_i] = spans_i[:, 1]
            padded_span_mask[i, :num_spans_i] = True
    
    if offsets_mapping.shape[1] != SeqLen:
        current_offsets_mapping = offsets_mapping[:, :SeqLen, :]
    else:
        current_offsets_mapping = offsets_mapping

    # (B_size, SeqLen, 1)
    offsets_start_expanded = current_offsets_mapping[..., 0].unsqueeze(2).to(device)
    offsets_end_expanded = current_offsets_mapping[..., 1].unsqueeze(2).to(device)
    
    # (B_size, 1, max_spans)
    span_starts_expanded = padded_span_starts.unsqueeze(1)
    span_ends_expanded = padded_span_ends.unsqueeze(1)

    token_in_span_map = (offsets_start_expanded + 1 >= span_starts_expanded) & \
                        (offsets_end_expanded <= span_ends_expanded)

    attention_mask_expanded = attention_mask.unsqueeze(2).bool()
    span_mask_expanded = padded_span_mask.unsqueeze(1) 

    final_token_to_span_map = token_in_span_map & attention_mask_expanded & span_mask_expanded

    if not final_token_to_span_map.any():
        print(f"No valid tokens found for any spans in the batch.")
        return torch.tensor(0.0, device=device)

    nonzero_indices = final_token_to_span_map.nonzero(as_tuple=False)
    
    batch_indices = nonzero_indices[:, 0] # (T_total)
    token_indices = nonzero_indices[:, 1] # (T_total)
    local_span_indices = nonzero_indices[:, 2] # (T_total)

    All_Indices = batch_indices * SeqLen + token_indices

    global_span_ids_flat = batch_indices * max_spans + local_span_indices
    _, Span_IDs = torch.unique(global_span_ids_flat, return_inverse=True) # (T_total)
    Max_Spans = Span_IDs.max().item() + 1 # Tổng số span duy nhất

    Batch_ID_for_Spans = torch.empty(Max_Spans, device=device, dtype=torch.long)
    Batch_ID_for_Spans.scatter_(0, Span_IDs, batch_indices)

    def gather_layer_weights(layer_weights):
        B_size, SeqLen = attention_mask.shape
        num_layers = layer_weights.shape[0]
        layer_weights_flat = layer_weights.view(num_layers, B_size * SeqLen)
        token_weights_unnorm = layer_weights_flat[:, All_Indices].float()
        batch_indices_expanded = batch_indices.unsqueeze(0).expand(num_layers, -1)
        sample_weight_sums = torch.zeros(num_layers, B_size, device=device, dtype=torch.float)
        sample_weight_sums.scatter_add_(1, batch_indices_expanded, token_weights_unnorm)
        sample_weight_sums = sample_weight_sums.clamp(min=1e-5)
        sample_weight_sums_gathered = torch.gather(sample_weight_sums, 1, batch_indices_expanded)
        Token_Weights_all = token_weights_unnorm / sample_weight_sums_gathered

        return Token_Weights_all

    T_Token_Weights_all = gather_layer_weights(t_layer_weights)
    S_Token_Weights_all = gather_layer_weights(s_layer_weights)

    return All_Indices, T_Token_Weights_all, S_Token_Weights_all, Span_IDs, Max_Spans, Batch_ID_for_Spans

def get_span_loss(attention_mask, s_hidden_states, t_hidden_states, 
                  offsets_mapping, spans_offsets, teacher_layer_mapping, student_layer_mapping):
    
    t_layer_weights = []
    s_layer_weights = []
    for i in teacher_layer_mapping:
        weights = compute_token_weights(t_hidden_states[i], attention_mask)  # (B, SeqLen)
        t_layer_weights.append(weights)
    for i in student_layer_mapping:
        weights = compute_token_weights(s_hidden_states[i], attention_mask)  # (B, SeqLen)
        s_layer_weights.append(weights)

    t_layer_weights = torch.stack(t_layer_weights)  # (num_layers, B, SeqLen)
    s_layer_weights = torch.stack(s_layer_weights)  # (num_layers, B, SeqLen)

    (All_Indices, T_Token_Weights_all, S_Token_Weights_all, 
     Span_IDs, Max_Spans, Batch_ID_for_Spans) =  prepare_span_indices_and_weights(t_layer_weights, s_layer_weights, 
                                                                                  attention_mask, offsets_mapping, spans_offsets)
    if All_Indices is None:
        return torch.tensor(0.0, device=attention_mask.device)
    
    final_loss = 0.0
    for i, (s_idx, t_idx) in enumerate(zip(student_layer_mapping, teacher_layer_mapping)):
        s_hidden = s_hidden_states[s_idx]
        t_hidden = t_hidden_states[t_idx]
        span_loss = compute_hidden_span_loss(s_hidden, t_hidden, All_Indices,
                                             S_Token_Weights_all[i], T_Token_Weights_all[i], 
                                             Span_IDs, Max_Spans, Batch_ID_for_Spans)
        final_loss += span_loss

    return final_loss

def compute_overall_span_loss(attention_mask, s_hidden_states, t_hidden_states, 
                              offsets_mapping, spans_offsets, args):
    
    s_span_mapping = args.student_layer_mapping
    t_span_mapping = args.teacher_layer_mapping
    span_loss = get_span_loss(attention_mask, s_hidden_states, t_hidden_states, 
                              offsets_mapping, spans_offsets, t_span_mapping, s_span_mapping)
    
    overall_loss = span_loss / len(args.student_layer_mapping)
    return overall_loss

def compute_hidden_span_loss(s_hidden_state, t_hidden_state, All_Indices, 
                             S_Token_Weights_all, T_Token_Weights_all, Span_IDs, Max_Spans, Batch_ID_for_Spans):
    D_hidden_s = s_hidden_state.size(-1)
    D_hidden_t = t_hidden_state.size(-1)
    device = t_hidden_state.device

    T_Hidden_Flat = t_hidden_state.flatten(0, 1) # (B*SeqLen, D_hidden_t)
    S_Hidden_Flat = s_hidden_state.flatten(0, 1) # (B*SeqLen, D_hidden_s)

    # 1. Trích xuất và Áp dụng Trọng số
    T_span_all = T_Hidden_Flat[All_Indices] # (T_total, D_hidden_t)
    S_span_all = S_Hidden_Flat[All_Indices] # (T_total, D_hidden_s)
    
    T_Token_Weights_expanded = T_Token_Weights_all.unsqueeze(-1) 
    S_Token_Weights_expanded = S_Token_Weights_all.unsqueeze(-1)
    
    T_span_weighted = T_span_all * T_Token_Weights_expanded # (T_total, D_hidden_t)
    S_span_weighted = S_span_all * S_Token_Weights_expanded # (T_total, D_hidden_s)

    Span_IDs_expanded_t = Span_IDs.unsqueeze(-1).expand(-1, D_hidden_t) 
    Span_IDs_expanded_s = Span_IDs.unsqueeze(-1).expand(-1, D_hidden_s) 

    T_span_sum = torch.zeros(Max_Spans, D_hidden_t, device=device)
    S_span_sum = torch.zeros(Max_Spans, D_hidden_s, device=device)
    T_Weight_sum_1d = torch.zeros(Max_Spans, device=device)
    S_Weight_sum_1d = torch.zeros(Max_Spans, device=device)

    T_span_sum.scatter_add_(0, Span_IDs_expanded_t, T_span_weighted)
    S_span_sum.scatter_add_(0, Span_IDs_expanded_s, S_span_weighted)

    T_Weight_sum_1d.scatter_add_(0, Span_IDs, T_Token_Weights_all) 
    T_Weight_sum = T_Weight_sum_1d.clamp(min=1e-5).unsqueeze(-1) # (Max_Spans, 1)
    S_Weight_sum_1d.scatter_add_(0, Span_IDs, S_Token_Weights_all)
    S_Weight_sum = S_Weight_sum_1d.clamp(min=1e-5).unsqueeze(-1) # (Max_Spans, 1)

    # Tính Trung bình (Mean)
    T_span_hidden_mean = T_span_sum / T_Weight_sum 
    S_span_hidden_mean = S_span_sum / S_Weight_sum

    S_normalized = F.normalize(S_span_hidden_mean, p=2, dim=-1)
    T_normalized = F.normalize(T_span_hidden_mean, p=2, dim=-1)
    S_Full_Sim_Matrix = S_normalized @ S_normalized.T
    T_Full_Sim_Matrix = T_normalized @ T_normalized.T

    Batch_IDs_col = Batch_ID_for_Spans.unsqueeze(1)
    Batch_IDs_row = Batch_ID_for_Spans.unsqueeze(0)
    Same_Batch_Mask = (Batch_IDs_col == Batch_IDs_row)
    Not_Self_Mask = ~torch.eye(Max_Spans, dtype=torch.bool, device=device)
    Final_Mask = Same_Batch_Mask & Not_Self_Mask

    S_intra_batch_similarities_flat = torch.masked_select(S_Full_Sim_Matrix, Final_Mask)
    T_intra_batch_similarities_flat = torch.masked_select(T_Full_Sim_Matrix, Final_Mask)

    Pair_Weights_Matrix = T_Weight_sum_1d.unsqueeze(1) * T_Weight_sum_1d.unsqueeze(0)
    Valid_Pair_Weights = torch.masked_select(Pair_Weights_Matrix, Final_Mask)

    span_loss = F.mse_loss(S_intra_batch_similarities_flat, T_intra_batch_similarities_flat, reduction='none')
    span_loss = (span_loss * Valid_Pair_Weights).sum() / Valid_Pair_Weights.sum().clamp(min=1e-5)

    return span_loss

def finetune(args, tokenizer: AutoTokenizer, model: deepspeed.DeepSpeedEngine, optimizer: AdamW, lr_scheduler, dataset, device, teacher_model=None):
    print_rank("Start Fine-tuning")

    # print_inspect(model, '*')
    if args.model_parallel:
        raise NotImplementedError
    else:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        dp_group = None
        loss_func = nn.CrossEntropyLoss()

    if args.balance_alpha is not None:
        sampler = BalancedSampler(dataset["train"], args.balance_alpha, rank=dp_rank, num_replicas=dp_world_size,
                                  seed=args.seed, log_dir=os.path.join(args.save, "balanced_epochs"))
    else:
        sampler = DistributedSampler(dataset["train"], shuffle=True, drop_last=True, rank=dp_rank, num_replicas=dp_world_size)
    train_dataloader = DataLoader(
        dataset['train'], sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=dataset["train"].collate)
    
    if "pt_train" in dataset:
        pt_sampler = DistributedSampler(dataset["pt_train"], shuffle=True, drop_last=True, rank=dp_rank, num_replicas=dp_world_size)
        pt_train_dataloader = DataLoader(
        dataset['pt_train'], sampler=pt_sampler, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=dataset["pt_train"].collate)
        pt_train_iter = iter(pt_train_dataloader)
        
    student_generator = SampleGenerator(args, tokenizer)

    step, global_step = 1, 1
    total_loss, total_distil_loss, total_time = 0.0, 0.0, 0.0
    
    adaptive_threshold = args.init_threshold if "adaptive" in args.type else -1.0
    # prev_avg_loss = evaluate(args, tokenizer, model, dataset["dev"], "dev", 0, device, adaptive_threshold)
    prev_avg_loss = 0.0
    best_dev = float("-inf")
    replay_buffer = ReplayBuffer(args)

    student_captured_hidden = []
    hook_handles = []
    def capture_hook_fn(module, input, output):
        if module.training: 
            if isinstance(output, tuple):
                student_captured_hidden.append(output[0])
            else:
                student_captured_hidden.append(output)

    # get_decoder(): the student may be a LoRA PeftModel or fully fine-tuned
    for layer in model.module.get_decoder().layers:
        h_layer = layer.register_forward_hook(capture_hook_fn)
        hook_handles.append(h_layer)
    
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)

        alloc_sum = alloc_count = 0

        model.train()
        for it, (model_batch, no_model_batch, gen_data, _, _) in enumerate(train_dataloader):
            dataset["train"].move_to_device(model_batch, no_model_batch, gen_data, device)
            student_captured_hidden.clear()
            student_captured_hidden.append(None)
            
            if args.lm_data_dir is not None:
                try:
                    pt_model_batch, pt_no_model_batch, pt_gen_data = next(pt_train_iter)
                    # pt_model_batch, pt_no_model_batch, pt_gen_data = pt_train_iter.next()
                except:
                    pt_train_iter = iter(pt_train_dataloader)
                    # pt_model_batch, pt_no_model_batch, pt_gen_data = pt_train_iter.next()
                    pt_model_batch, pt_no_model_batch, pt_gen_data = next(pt_train_iter)
                    
                dataset["pt_train"].move_to_device(pt_model_batch, pt_no_model_batch, pt_gen_data, device)
            
            torch.cuda.synchronize()
            st_time = time.time()
            
            # # sampling ratio:
            samp_threshold = adaptive_threshold * (1 - global_step / args.total_iters)
            if "adaptive" in args.type:
                if args.replay_ratio == "constant":
                    samp_threshold = adaptive_threshold * 0.5
                elif args.replay_ratio == "increasing":
                    samp_threshold = adaptive_threshold * global_step / args.total_iters
                else:
                    samp_threshold = adaptive_threshold * (1 - global_step / args.total_iters)
            
            # data generation
            if args.student_gen:
                r = np.random.uniform(0, 1)
                if "mixed" in args.type and r < args.mixed_alpha:
                    model_batch = student_generator.run_sample(model, gen_data)
                    no_model_batch["label"] = model_batch.pop("no_model_batch")
                    
                    replay_buffer.move_to_memory(model_batch, no_model_batch)
                    model_batch, no_model_batch, gen_data = replay_buffer.sample()
                    model_batch, no_model_batch = replay_buffer.move_to_device(model_batch, no_model_batch, gen_data, device)
                    
                elif "adaptive" in args.type and (r < samp_threshold or (r < adaptive_threshold and len(replay_buffer) < args.capacity)):

                    model_batch = student_generator.run_sample(model, gen_data)
                    no_model_batch["label"] = model_batch.pop("no_model_batch")
                    
                    if args.model_type in ["opt"]:
                        model_batch.pop('position_ids')
                        
                    replay_buffer.move_to_memory(model_batch, no_model_batch, gen_data)
                    
                elif "adaptive" in args.type and r < adaptive_threshold:
                    model_batch, no_model_batch, gen_data = replay_buffer.sample()
                    model_batch, no_model_batch, gen_data = replay_buffer.move_to_device(model_batch, no_model_batch, gen_data, device)
                    
                model.train()

            label = no_model_batch["label"]
            start, end = label_window(label)
            window = torch.arange(start, end, device=label.device)
            label_batch = {"label": label[:, start:end]}
            outputs = model(**model_batch, use_cache=False, logits_to_keep=window)
            
            logits = outputs.logits
            if args.model_parallel:
                raise NotImplementedError
            else:
                lm_loss = loss_func(logits.float().reshape(-1, logits.shape[-1]), label_batch["label"].reshape(-1))
            
            if teacher_model is not None:
                # --w-span-loss 0 skips the span loss and the teacher's hidden states
                use_span = args.w_span_loss > 0
                with torch.no_grad():
                    teacher_model.eval()
                    teacher_logits, teacher_hidden = teacher_forward(args, teacher_model, model_batch,
                                                                     no_model_batch["dataset"], window, use_span)

                distil_loss = get_distil_loss(args, teacher_logits, label_batch, logits)

                if use_span:
                    spans_offsets = no_model_batch["span_offsets"]
                    offset_mapping = no_model_batch["offset_mapping"]

                    span_loss = compute_overall_span_loss(model_batch['attention_mask'],
                                                          student_captured_hidden, teacher_hidden,
                                                          offset_mapping, spans_offsets, args)
                    span_loss = args.w_span_loss * span_loss
                    distil_loss = distil_loss + span_loss
                # loss = (1 - args.kd_ratio) * lm_loss + args.kd_ratio * (distil_loss + span_loss)
                # BAM teacher annealing: the target moves from the teacher to the gold labels as training goes
                kd_ratio = 1 - global_step / args.total_iters if args.kd_schedule == "anneal" else args.kd_ratio
                loss = (1 - kd_ratio) * lm_loss + kd_ratio * distil_loss
            else:
                loss = lm_loss
                
            if args.lm_data_dir is not None:
                assert args.lm_coef is not None
                loss += args.lm_coef * pt_loss(args, model, pt_model_batch, pt_no_model_batch)
                
            model.backward(loss)
            model.step()
             
            dist.all_reduce(loss, dist.ReduceOp.SUM, group=dp_group)
            global_loss = loss.item() / dp_world_size

            global_distil_loss = 0
            if teacher_model is not None:
                dist.all_reduce(distil_loss, dist.ReduceOp.SUM, group=dp_group)
                global_distil_loss = distil_loss.item() / dp_world_size
                total_distil_loss += global_distil_loss
    
            torch.cuda.synchronize()
            elapsed_time = time.time() - st_time

            total_loss += global_loss
            total_time += elapsed_time

            # Logging
            def get_log(log_loss, log_distil_loss, log_time):
                return "train | epoch {:3d} | Iter: {:6d}/{:6d} | global iter: {:6d}/{:6d} | loss: {:.4f} | ds_loss: {:.4f} | lr: {:.4e} | scale: {:10.4f} | micro time: {:.3f} | step time: {:.3f}".format(
                    epoch,
                    step,
                    args.total_iters * args.gradient_accumulation_steps,
                    global_step,
                    args.total_iters,
                    log_loss,
                    log_distil_loss,
                    lr_scheduler.get_last_lr()[0],
                    optimizer.cur_scale if hasattr(optimizer, "cur_scale") else 0,
                    elapsed_time,
                    log_time,
                )

            if args.mid_log_num > 0:
                mid_log_step = args.gradient_accumulation_steps // args.mid_log_num
                mid_log_step = 1 if mid_log_step == 0 else mid_log_step
                if step % mid_log_step == 0:
                    print_rank(get_log(global_loss, global_distil_loss, 0))

            if global_step % args.log_interval == 0 and step % args.gradient_accumulation_steps == 0:
                log_str = get_log(
                    total_loss / (args.log_interval * args.gradient_accumulation_steps),
                    total_distil_loss / (args.log_interval * args.gradient_accumulation_steps),
                    total_time / (args.log_interval))
                print_rank("*" * 100)
                print_rank(log_str)
                print_rank(args.save)
                print_rank("*" * 100)
                save_rank(log_str, os.path.join(args.save, "log.txt"))
                total_loss, total_distil_loss, total_time = 0.0, 0.0, 0.0

                # --- MEMORY MEASUREMENT BLOCK ---
                allocated   = torch.cuda.memory_allocated() / 1e9
                peak_alloc  = torch.cuda.max_memory_allocated() / 1e9

                alloc_sum   += allocated
                alloc_count += 1
                avg_alloc   = alloc_sum / alloc_count

                print_rank("train | avg_alloc {:.4f} GB | peak_alloc {:.4f} GB".format(avg_alloc, peak_alloc))

                # --- END MEMORY MEASUREMENT BLOCK ---
            
            # Checkpointing
            if args.save and args.save_interval and global_step % args.save_interval == 0 and step % args.gradient_accumulation_steps == 0:
                save_dir_path = os.path.join(args.save, str(global_step))
                if args.model_parallel:
                    raise NotImplementedError
                else:
                    if dist.get_rank() == 0:
                        os.makedirs(save_dir_path, exist_ok=True)
                        print_rank(f"Model save to {save_dir_path}")
                        tokenizer.save_pretrained(save_dir_path)
                        model.module.save_pretrained(save_dir_path, safe_serialization=False)
                dist.barrier()

            # Evaluation
            if args.eval_interval and global_step % args.eval_interval == 0 and step % args.gradient_accumulation_steps == 0:
                curr_avg_loss, best_dev = evaluate_dev_and_test(args, tokenizer, model, dataset, epoch, device,
                                                                best_dev, adaptive_threshold)
                if "adaptive" in args.type:
                    if curr_avg_loss >= prev_avg_loss + args.loss_eps:
                        adaptive_threshold += 0.1
                        adaptive_threshold = min(adaptive_threshold, 1.0)
                        prev_avg_loss = curr_avg_loss

                    
                model.train()
                
            step += 1
            if step % args.gradient_accumulation_steps == 0:
                global_step += 1
            
            if global_step > args.total_iters:
                break

    for h in hook_handles:
        h.remove()
            
    return model



def main():
    torch.backends.cudnn.enabled = False
    
    args = get_args()
    initialize(args)
    
    if dist.get_rank() == 0:
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f)
    
    device = torch.cuda.current_device()
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    save_rank("\n\n" + "="*30 + f" EXP at {cur_time} " + "="*30, os.path.join(args.save, "log.txt"))
    
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_clipping"] = args.clip_grad
    ds_config["steps_per_print"] = 10000000
    
    if not args.do_train:
        ds_config["zero_optimization"]["stage"] = 0
    
    args.fp32 = not ds_config["fp16"]["enabled"]  
    args.bf16 = "bf16" in ds_config and ds_config["bf16"]["enabled"]  
    args.deepspeed_config = None
    
    # get the tokenizer
    tokenizer = get_tokenizer(args)
    print(type(tokenizer))

    # the span loss is padding-invariant since compute_token_weights averages over real query rows only
    args.dynamic_pad_effective = args.dynamic_pad and not args.student_gen
    print_rank(f"dynamic padding: {args.dynamic_pad_effective}")
    dataset = prepare_dataset(
        args,
        tokenizer,
    )
    
    dp_world_size = dist.get_world_size()
    
    if args.do_train:
        train_rows = len(dataset["train"])
        if args.balance_alpha is not None:   # a balanced epoch is shorter than the train set
            train_rows = sum(balanced_quota([r["dataset"] for r in dataset["train"].raw[:train_rows]], args.balance_alpha).values())
        args.train_iters_per_epoch = int(train_rows / (args.batch_size * dp_world_size * args.gradient_accumulation_steps))
        print_rank("Train iters per epoch", args.train_iters_per_epoch)
        if args.total_iters is None:
            args.total_iters = args.train_iters_per_epoch * args.epochs
        if args.epochs is None:
            args.epochs = math.ceil(args.total_iters / args.train_iters_per_epoch)
        print_rank("total_iters", args.total_iters)
        
        if args.save_interval == -1:
            args.save_interval = args.train_iters_per_epoch
        
        if args.eval_interval == -1:
            args.eval_interval = args.train_iters_per_epoch
    
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args, ds_config, device, set_optim=args.do_train)
    check_gen_backend(args, model, tokenizer)
    
    if args.teacher_model_type is None:
        args.teacher_model_type = args.model_type
    
    if args.teacher_peft_paths is not None and args.do_train:
        # every train dataset needs its teacher, and a name that matches no dataset is a typo
        names, rows = set(teacher_peft_paths(args)), set(r["dataset"] for r in dataset["train"].raw)
        assert names == rows, f"--teacher-peft-paths has {sorted(names)}, the train rows are from {sorted(rows)}"
        # student-generated batches come from the replay buffer, which drops the rows' dataset
        assert "adaptive" not in args.type and "mixed" not in args.type, "--teacher-peft-paths needs gold batches"

    if args.teacher_model_path is not None:
        teacher_model = get_teacher_model(args, device)
    else:
        teacher_model = None
    
    if args.do_train:
        model = finetune(args, tokenizer, model, optimizer, lr_scheduler, dataset, device, teacher_model=teacher_model)
   
    if args.do_eval:
        evaluate(args, tokenizer, model, dataset["test"], "test", 0, device)
        
    
if __name__ == "__main__":
    main()
