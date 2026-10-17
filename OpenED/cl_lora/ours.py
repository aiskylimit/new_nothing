"""Our distillation on top of a Family-A CL-LoRA method (--ours in engine.py).

"Ours" is the f12_pl recipe that run_ced_v2.sh trains with ced_finetune.py, rebuilt inside the
CL-LoRA engine so the CL-LoRA method keeps its own adapters and regulariser:
  * data, before each task t > 0 (tools/ced_pseudo_label.py + tools/ced_oversample.py):
      pseudo-labels from the previous model with conflict dedup and the (trigger, type)
      lexicon of earlier tasks, no confidence filter; then replay rows oversampled x5
  * loss, on batches that contain replay rows (ced_finetune.py, --ced-kd-scope replay):
      (1 - 0.9) * CE + 0.9 * (SFKL(skew 0.1) on replay rows + 2.0 * span loss on their spans),
      span loss = cosine relational loss over layers 22, 25, 28; other batches: plain CE
  * teacher = the model at the start of the task. A new task's lora_B starts at zero, so a
    copy taken right after start_new_task() computes exactly the previous model.
Task 0 has nothing to distil and trains as the plain method does.

The span-loss functions are copied from ced_finetune.py, which cannot be imported here: it
pulls in deepspeed, and this engine runs without it. Kept line-for-line apart from the cosine
branch only and one return-value fix (marked).
"""
import copy
import importlib.util
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from distillm.losses import skewed_forward_kl

# tools/ has no __init__.py, so `import tools.ced_pseudo_label` resolves as a namespace package
# and any installed package called `tools` would shadow it; load the file by path instead
_PL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "tools", "ced_pseudo_label.py")
_spec = importlib.util.spec_from_file_location("ced_pseudo_label", _PL_PATH)
_pl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pl)
input_text_of, parse_events = _pl.input_text_of, _pl.parse_events

KD_RATIO = 0.9
W_SPAN = 2.0
SKEW = 0.1
LAYERS = (22, 25, 28)
BOOST = 5
PL_BATCH = 64
PL_MAX_NEW_TOKENS = 300
SUPPORTED = ("inclora", "olora", "tree")


def old_types_of(streams, task_id):
    old = set()
    for s in streams[:task_id]:
        old.update(s)
    return old


def events_of(row):
    return json.loads(row["response"]).get("events", [])


# ---------------------------------------------------------------- data (pseudo-labels + boost)
def build_lexicon(data_root, task_id, old_types):
    """(trigger, type) pairs of old types in the gold train data of tasks < task_id."""
    lexicon = set()
    for t_prev in range(task_id):
        with open(os.path.join(data_root, str(t_prev), "train.jsonl"), encoding="utf-8") as f:
            for line in f:
                for e in events_of(json.loads(line)):
                    if e[1] in old_types:
                        lexicon.add((e[0].lower(), e[1]))
    return lexicon


@torch.no_grad()
def pseudo_label(model, tok, rows, old_types, lexicon, device):
    """tools/ced_pseudo_label.py with --conflict-dedup 1 --lexicon-filter 1 --conf-filter none,
    on an in-memory model. Returns (new rows, stats)."""
    rows = [dict(r) for r in rows]
    cand_idx = [i for i, r in enumerate(rows) if {e[1] for e in events_of(r)} - old_types]
    side = tok.padding_side
    tok.padding_side = "left"
    model.eval()
    n_seen = n_dropped_conflict = n_aug_rows = n_aug_events = 0
    try:
        for b in range(0, len(cand_idx), PL_BATCH):
            idxs = cand_idx[b:b + PL_BATCH]
            prompts = [tok.apply_chat_template(
                [{"role": "system", "content": rows[i]["system_prompt"]},
                 {"role": "user", "content": rows[i]["user_prompt"]}],
                add_generation_prompt=True, tokenize=False, enable_thinking=False) for i in idxs]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                      max_length=1024).to(device)
            out = model.generate(**enc, max_new_tokens=PL_MAX_NEW_TOKENS, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for i, text in zip(idxs, texts):
                sent = input_text_of(rows[i]["user_prompt"]) or ""
                gold = events_of(rows[i])
                gold_keys = {(e[0], e[1]) for e in gold}
                gold_triggers = {str(e[0]).lower() for e in gold if isinstance(e, list) and e}
                kept = []
                for e in parse_events(text):
                    if not isinstance(e, list) or len(e) < 2:
                        continue
                    trig, ty = e[0], e[1]
                    if not isinstance(ty, str) or ty not in old_types:
                        continue
                    if not isinstance(trig, str) or trig not in sent:
                        continue
                    if (trig, ty) in gold_keys:
                        continue
                    tl = trig.lower()
                    if any(tl == g or tl in g or g in tl for g in gold_triggers):
                        n_dropped_conflict += 1
                        continue
                    if (tl, ty) not in lexicon:
                        continue
                    args_clean = []
                    if len(e) > 2 and isinstance(e[2], list):
                        for a in e[2]:
                            if isinstance(a, list) and len(a) >= 2 \
                                    and isinstance(a[0], str) and isinstance(a[1], str):
                                args_clean.append([a[0], a[1]])
                    kept.append([trig, ty, args_clean,
                                 e[3] if len(e) > 3 and isinstance(e[3], str) else ""])
                    gold_keys.add((trig, ty))
                    gold_triggers.add(tl)
                    n_seen += 1
                if kept:
                    rows[i]["response"] = json.dumps({"events": gold + kept})
                    n_aug_rows += 1
                    n_aug_events += len(kept)
    finally:
        tok.padding_side = side
    stats = {"candidates": len(cand_idx), "aug_rows": n_aug_rows, "aug_events": n_aug_events,
             "dropped_conflict": n_dropped_conflict}
    return rows, stats


def oversample(rows, old_types, boost):
    """tools/ced_oversample.py: repeat replay rows (all event types old) boost times."""
    out, n_replay = [], 0
    for r in rows:
        out.append(r)
        types = {e[1] for e in events_of(r)}
        if types and types <= old_types:
            n_replay += 1
            out.extend([r] * (boost - 1))
    return out, n_replay


def prepare_task_data(a, model, tok, task_id, streams, device, out_dir):
    """Write out_dir/train.jsonl = boost(pseudo_label(raw train)); return its path."""
    old_types = old_types_of(streams, task_id)
    with open(os.path.join(a.data_root, str(task_id), "train.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if a.limit > 0:
        rows = rows[:a.limit]
    lexicon = build_lexicon(a.data_root, task_id, old_types)
    rows, stats = pseudo_label(model, tok, rows, old_types, lexicon, device)
    rows, n_replay = oversample(rows, old_types, BOOST)
    stats.update(task_id=task_id, lexicon=len(lexicon), replay_rows=n_replay,
                 boost=BOOST, train_rows=len(rows))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "train.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "pl_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f)
    print(f"[ours] task{task_id} data: {json.dumps(stats)}", flush=True)
    return path


class OursTrainSet(Dataset):
    """Wraps the engine's JsonlED: adds the replay flag, the field-span char offsets and the
    token offsets that the span loss needs (data_utils/lm_datasets.py get_span_offsets /
    get_replay_flags, on the same prompt + response text the engine tokenizes)."""

    def __init__(self, base, old_types):
        self.base = base
        tok = base.tok
        self.flags, self.spans, self.offsets = [], [], []
        for r in base.rows:
            events = events_of(r)
            types = {e[1] for e in events}
            self.flags.append(bool(types) and types <= old_types)
            prompt_text = base._prompt_text(r)
            full_text = prompt_text + r["response"]
            spans = []
            # a truncated prompt shifts every token position; such rows get no span loss
            if len(tok.encode(prompt_text, add_special_tokens=False)) <= base.max_prompt_length:
                values = []
                for e in events:
                    if not isinstance(e, list) or len(e) < 2:
                        continue
                    values += [e[0], e[1]]
                    if len(e) > 3:
                        for arg in e[2]:
                            if isinstance(arg, list) and len(arg) >= 2:
                                values += [arg[0], arg[1]]
                        values.append(e[3])
                    elif len(e) == 3:
                        values.append(e[2])
                start = 0
                for val in values:
                    cs = full_text.find(f"{val}", start)
                    if cs != -1:
                        spans.append((cs, cs + len(val)))
                        start = cs + len(val) + 1
            self.spans.append(spans)
            self.offsets.append(tok(full_text, return_offsets_mapping=True,
                                    add_special_tokens=False)["offset_mapping"])
        print(f"[ours] replay rows {sum(self.flags)}/{len(self.flags)}", flush=True)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = dict(self.base[i])
        item["index"] = i
        return item

    def collate_train(self, batch):
        mb, nmb = self.base.collate_train(batch)
        bs, seq = mb["input_ids"].shape
        offsets = torch.zeros(bs, seq, 2, dtype=torch.long)
        for row, b in enumerate(batch):
            om = self.offsets[b["index"]][:seq]
            if om:
                offsets[row, :len(om)] = torch.tensor(om, dtype=torch.long)
        nmb["is_replay"] = torch.tensor([self.flags[b["index"]] for b in batch], dtype=torch.bool)
        nmb["spans"] = [self.spans[b["index"]] for b in batch]
        nmb["offset_mapping"] = offsets
        return mb, nmb


# ---------------------------------------------------------------- loss (from ced_finetune.py)
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

    token_weights = attn_weights.mean(dim=1).squeeze(0)  # [L]
    return token_weights.detach()


def prepare_span_indices_and_weights(t_layer_weights, s_layer_weights,
                                     attention_mask, offsets_mapping, spans_offsets):
    device = attention_mask.device
    B_size, SeqLen = attention_mask.shape

    max_spans = max(len(s) for s in spans_offsets)
    if max_spans == 0:
        return None, None, None, None, None, None

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

    offsets_start_expanded = current_offsets_mapping[..., 0].unsqueeze(2).to(device)
    offsets_end_expanded = current_offsets_mapping[..., 1].unsqueeze(2).to(device)

    span_starts_expanded = padded_span_starts.unsqueeze(1)
    span_ends_expanded = padded_span_ends.unsqueeze(1)

    token_in_span_map = (offsets_start_expanded + 1 >= span_starts_expanded) & \
                        (offsets_end_expanded <= span_ends_expanded)

    attention_mask_expanded = attention_mask.unsqueeze(2).bool()
    span_mask_expanded = padded_span_mask.unsqueeze(1)

    final_token_to_span_map = token_in_span_map & attention_mask_expanded & span_mask_expanded

    if not final_token_to_span_map.any():
        # ced_finetune.py returns a single tensor here, which its caller cannot unpack
        return None, None, None, None, None, None

    nonzero_indices = final_token_to_span_map.nonzero(as_tuple=False)

    batch_indices = nonzero_indices[:, 0]
    token_indices = nonzero_indices[:, 1]
    local_span_indices = nonzero_indices[:, 2]

    All_Indices = batch_indices * SeqLen + token_indices

    global_span_ids_flat = batch_indices * max_spans + local_span_indices
    _, Span_IDs = torch.unique(global_span_ids_flat, return_inverse=True)
    Max_Spans = Span_IDs.max().item() + 1

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


def compute_hidden_span_loss(s_hidden_state, t_hidden_state, All_Indices,
                             S_Token_Weights_all, T_Token_Weights_all, Span_IDs, Max_Spans, Batch_ID_for_Spans):
    D_hidden_s = s_hidden_state.size(-1)
    D_hidden_t = t_hidden_state.size(-1)
    device = t_hidden_state.device

    T_Hidden_Flat = t_hidden_state.flatten(0, 1)
    S_Hidden_Flat = s_hidden_state.flatten(0, 1)

    T_span_all = T_Hidden_Flat[All_Indices]
    S_span_all = S_Hidden_Flat[All_Indices]

    T_Token_Weights_expanded = T_Token_Weights_all.unsqueeze(-1)
    S_Token_Weights_expanded = S_Token_Weights_all.unsqueeze(-1)

    T_span_weighted = T_span_all * T_Token_Weights_expanded
    S_span_weighted = S_span_all * S_Token_Weights_expanded

    Span_IDs_expanded_t = Span_IDs.unsqueeze(-1).expand(-1, D_hidden_t)
    Span_IDs_expanded_s = Span_IDs.unsqueeze(-1).expand(-1, D_hidden_s)

    T_span_sum = torch.zeros(Max_Spans, D_hidden_t, device=device)
    S_span_sum = torch.zeros(Max_Spans, D_hidden_s, device=device)
    T_Weight_sum_1d = torch.zeros(Max_Spans, device=device)
    S_Weight_sum_1d = torch.zeros(Max_Spans, device=device)

    T_span_sum.scatter_add_(0, Span_IDs_expanded_t, T_span_weighted)
    S_span_sum.scatter_add_(0, Span_IDs_expanded_s, S_span_weighted)

    T_Weight_sum_1d.scatter_add_(0, Span_IDs, T_Token_Weights_all)
    T_Weight_sum = T_Weight_sum_1d.clamp(min=1e-5).unsqueeze(-1)
    S_Weight_sum_1d.scatter_add_(0, Span_IDs, S_Token_Weights_all)
    S_Weight_sum = S_Weight_sum_1d.clamp(min=1e-5).unsqueeze(-1)

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


def span_loss_over_layers(attention_mask, s_hidden_states, t_hidden_states, offsets_mapping, spans_offsets):
    """get_span_loss + compute_overall_span_loss of ced_finetune.py, same layer list both sides."""
    t_layer_weights = torch.stack([compute_token_weights(t_hidden_states[i], attention_mask) for i in LAYERS])
    s_layer_weights = torch.stack([compute_token_weights(s_hidden_states[i], attention_mask) for i in LAYERS])
    (All_Indices, T_Token_Weights_all, S_Token_Weights_all,
     Span_IDs, Max_Spans, Batch_ID_for_Spans) = prepare_span_indices_and_weights(
        t_layer_weights, s_layer_weights, attention_mask, offsets_mapping, spans_offsets)
    if All_Indices is None:
        return torch.tensor(0.0, device=attention_mask.device)
    final_loss = 0.0
    for i, idx in enumerate(LAYERS):
        final_loss += compute_hidden_span_loss(
            s_hidden_states[idx], t_hidden_states[idx], All_Indices,
            S_Token_Weights_all[i], T_Token_Weights_all[i], Span_IDs, Max_Spans, Batch_ID_for_Spans)
    return final_loss / len(LAYERS)


class Ours:
    """Holds the frozen previous-model teacher and the student hidden-state hooks."""

    def __init__(self, model):
        self.teacher = None
        # Student hidden states come from forward hooks on the decoder layers, teacher ones from
        # output_hidden_states, exactly as in ced_finetune.py: index i is the output of layer
        # i-1 on both sides (index 0 is a placeholder / the embeddings).
        self.captured = []

        def hook(module, inputs, output):
            if module.training:
                self.captured.append(output[0] if isinstance(output, tuple) else output)

        for layer in model.base_model.model.model.layers:
            layer.register_forward_hook(hook)

    def start_task(self, model):
        """Freeze a copy of the model as it stands: the previous model (new lora_B is zero)."""
        self.teacher = copy.deepcopy(model).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    def end_task(self):
        self.teacher = None
        torch.cuda.empty_cache()

    def before_forward(self):
        self.captured.clear()
        self.captured.append(None)

    def loss(self, mb, nmb, logits, labels, ce):
        """(1 - KD_RATIO) * CE + KD_RATIO * (KD + W_SPAN * span) when the batch has replay rows."""
        if self.teacher is None:
            return ce
        device = logits.device
        is_replay = nmb["is_replay"].to(device)
        if not bool(is_replay.any()):
            return ce
        with torch.no_grad():
            t_out = self.teacher(**mb, output_hidden_states=True, use_cache=False)
        kd_label = labels[:, 1:].clone()
        kd_label[~is_replay] = -100
        distil = torch.tensor(0.0, device=device)
        if bool((kd_label != -100).any()):
            distil = skewed_forward_kl(logits[:, :-1], t_out.logits[:, :-1], {"label": kd_label}, lam=SKEW)
        spans = [s if f else [] for s, f in zip(nmb["spans"], is_replay.tolist())]
        span = span_loss_over_layers(mb["attention_mask"], self.captured, t_out.hidden_states,
                                     nmb["offset_mapping"].to(device), spans)
        distil = distil + W_SPAN * span
        return (1 - KD_RATIO) * ce + KD_RATIO * distil
