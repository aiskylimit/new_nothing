import random
import re
import torch
import os
import json
import pickle
import numpy as np
from torch.utils.data import Dataset
from .distributed_indexed import DistributedMMapIndexedDataset

from torch.distributed import get_rank, get_world_size, barrier
from utils import print_rank
from utils import save_rank


class LMTrainDataset(Dataset):
    def __init__(self, args, tokenizer, path, split, num, ratio, rng_sample: random.Random):
        self.args = args
        self.tokenizer = tokenizer
        self.split = split
        self.pad_id = self.tokenizer.eos_token_id
        self.ratio = ratio
        self.max_length = args.max_length
        self.max_prompt_length = args.max_prompt_length
        self.rng_sample = rng_sample
        self.lm_ctx = DistributedMMapIndexedDataset(path, f"{split}", get_rank(), get_world_size())
        self.t_lm_ctx = None

        if os.path.exists(os.path.join(path, f"teacher_train_0.bin")) and split == "train":
            self.t_lm_ctx = DistributedMMapIndexedDataset(path, f"teacher_train", get_rank(), get_world_size())

        # --datasets: keep only those source datasets. Row i of {split}.jsonl is row i of the .bin
        # (tools/process_data.py writes both in one loop), so self.keep maps a kept row to its .bin row.
        self.keep = None
        if os.path.exists(os.path.join(path, f"{split}.jsonl")):
            with open(os.path.join(path, f"{split}.jsonl")) as f:
                self.raw = [json.loads(line) for line in f.readlines()]
                if args.datasets is not None:
                    wanted = set(args.datasets.split(","))
                    self.keep = [i for i, x in enumerate(self.raw) if x["dataset"] in wanted]
                    assert self.keep, f"no {split} rows for --datasets {args.datasets}"
                    self.raw = [self.raw[i] for i in self.keep]
                self.answers = [x["response"] if isinstance(x["response"], list) else [x["response"]] for x in self.raw]
                self.full_texts = [x["prompt"] + x["response"] for x in self.raw]
                self.offset_mapping = [tokenizer(text, return_offsets_mapping=True, truncation=True, 
                                                 max_length=self.max_length, padding="max_length",
                                                 add_special_tokens=False, return_tensors="pt")["offset_mapping"]
                                       for text in self.full_texts]
                
                self.get_span_offsets()
                self.get_unknown_arg_tokens()
        
        rows = len(self.lm_ctx) if self.keep is None else len(self.keep)
        print_rank(rows)
        if num == -1:
            self.num = rows
        else:
            self.num = num

        print_rank(f"Num LM instances: {rows}")

    def get_span_offsets(self):
        self.span_offsets = []
        for item, full_text in zip(self.raw, self.full_texts):
            response_str = item["response"]
            response_json = json.loads(response_str)

            values_to_find = []
            for event in response_json.get("events", []):
                values_to_find.append(event[0])  # 1. trigger
                values_to_find.append(event[1])  # 2. event_type
                
                if len(event) > 2 and isinstance(event[2], list):
                    for arg in event[2]:             # 3. Duyệt qua các arguments
                        values_to_find.append(arg[0])  # arg_span
                        values_to_find.append(arg[1])  # arg_role

                if len(event) > 3:
                    values_to_find.append(event[3])  # 4. description
                elif len(event) == 3 and not isinstance(event[2], list):
                    values_to_find.append(event[2])  # 3. description
                # general_domain rows have no description: [trigger, type] or [trigger, type, arguments]

            result_tuples = []
            # search_start_idx = len(full_text) - len(response_str)
            search_start_idx = 0

            for val in values_to_find:
                search_str = f'{val}'
                char_start = full_text.find(search_str, search_start_idx)
                
                if char_start != -1:
                    char_end = char_start + len(val)
                    result_tuples.append((char_start, char_end))
                    search_start_idx = char_end + 1

            self.span_offsets.append(result_tuples)

    def get_unknown_arg_tokens(self):
        """Rows with "unknown_args" (MAVEN, RAMS: no argument annotation) write every event as
        [trigger, type, []]. That [] means "not annotated", not "no arguments", so the tokens that
        overlap it get no loss (CE and KD both read label != -100). With the Qwen3 tokenizer the
        slot is the single token ' []'. Returns, per row, the token indices in prompt + response."""
        self.unknown_arg_tokens = []
        for item, full_text, offsets in zip(self.raw, self.full_texts, self.offset_mapping):
            tokens = []
            if item.get("unknown_args"):
                start = len(item["prompt"])
                spans = [(start + m.start(1), start + m.end(1))
                         for m in re.finditer(r'", (\[\])\]', item["response"])]
                for k, (a, b) in enumerate(offsets[0].tolist()):
                    if b > a and any(a < e and s < b for s, e in spans):
                        tokens.append(k)
            self.unknown_arg_tokens.append(tokens)

    def __len__(self):
        return self.num
   
    def __getitem__(self, index):
        return self._get_lm(index)
    
    def _get_lm(self, index):
        row = index if self.keep is None else self.keep[index]   # raw, span_offsets etc. are already filtered
        data = self.lm_ctx[row]
        input_ids = data.astype(int)

        t_input_ids = None
        if self.t_lm_ctx is not None:
            t_data = self.t_lm_ctx[row]
            t_input_ids = t_data.astype(int)

        return {
            "input_ids": input_ids,
            "t_input_ids": t_input_ids,
            "span_offsets": self.span_offsets[index],
            "offset_mapping": self.offset_mapping[index],
            "unknown_arg_tokens": self.unknown_arg_tokens[index],
            "dataset": self.raw[index].get("dataset")   # picks the teacher with --teacher-peft-paths
        }

    def _process_lm(self, i, samp, model_data, no_model_data, gen_data):
        input_ids = samp["input_ids"]
        source_len = 1
        
        prompt = None
        if self.args.model_type in ["qwen"] and 4294967295 in input_ids:
            source_len = np.where(input_ids==4294967295)[0][0]
            prompt = input_ids[:source_len]
            input_ids = np.concatenate([input_ids[:source_len], input_ids[source_len+1:]], axis=0)
        elif 65535 in input_ids:
            source_len = np.where(input_ids==65535)[0][0]
            prompt = input_ids[:source_len]
            input_ids = np.concatenate([input_ids[:source_len], input_ids[source_len+1:]], axis=0)
        
        input_ids = input_ids[:self.max_length]
        input_len = len(input_ids)
        model_data["input_ids"][i][:input_len-1] = torch.tensor(input_ids[:-1], dtype=torch.long)
        model_data["attention_mask"][i][:input_len-1] = 1.0
        if self.args.model_type in ["gpt2"]:
            model_data["position_ids"][i][:input_len-1] = torch.arange(0, input_len-1, dtype=torch.long)
        no_model_data["label"][i][:input_len-1] = torch.tensor(input_ids[1:], dtype=torch.long)
        no_model_data["label"][i][:source_len-1] = -100
        if "loss_mask" in no_model_data:
            no_model_data["loss_mask"][i][:input_len-1] = 1.0
            no_model_data["loss_mask"][i][:source_len-1] = 0
        
        if prompt is not None and gen_data is not None:
            gen_data["input_ids"][i][-len(prompt):] = torch.tensor(prompt, dtype=torch.long)
            gen_data["attention_mask"][i][-len(prompt):] = 1.0

    def move_to_device(self, model_data, no_model_data, gen_data, device):
        for k in model_data:
            model_data[k] = model_data[k].to(device)

        for k in no_model_data:
            if isinstance(no_model_data[k], torch.Tensor):
                no_model_data[k] = no_model_data[k].to(device)

        if gen_data is not None:
            for k in gen_data:
                gen_data[k] = gen_data[k].to(device)

        return model_data, no_model_data, gen_data

    def batch_width(self, samples):
        """Columns of the model inputs: --max-length, or with dynamic padding the longest row of the
        batch rounded up to a multiple of 64 (at most --max-length). Rows are right padded and attention
        is causal, so a row's logits do not depend on its padding. From new_nothing/OpenED."""
        if not getattr(self.args, "dynamic_pad_effective", False):
            return self.max_length
        need = 1
        for sample in samples:
            ids = sample["input_ids"]
            n = len(ids) - 1 if (4294967295 in ids or 65535 in ids) else len(ids)
            need = max(need, min(n, self.max_length) - 1)
        return min(self.max_length, -(-need // 64) * 64)

    def collate(self, samples):
        bs = len(samples)

        max_length = self.batch_width(samples)
        
        model_data = {
            "input_ids": torch.ones(bs, max_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, max_length),
        }
        
        if self.args.model_type in ["gpt2"]:
            model_data["position_ids"] = torch.zeros(bs, max_length, dtype=torch.long)
            
        no_model_data = {
            "label": torch.ones(bs, max_length, dtype=torch.long) * -100,
            "loss_mask": torch.zeros(bs, max_length),
            "span_offsets": [sample["span_offsets"] for sample in samples],
            "offset_mapping": torch.concat([sample["offset_mapping"] for sample in samples]),
            "dataset": [sample["dataset"] for sample in samples]
        }
        
        gen_data = {
            "input_ids": torch.ones(bs, self.max_prompt_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, self.max_prompt_length, dtype=torch.long),
        }

        for i, samp in enumerate(samples):
            self._process_lm(i, samp, model_data, no_model_data, gen_data)
            # token k is predicted at label position k - 1
            for k in samp["unknown_arg_tokens"]:
                if 0 < k <= max_length:
                    no_model_data["label"][i][k - 1] = -100
                    no_model_data["loss_mask"][i][k - 1] = 0

        t_model_data, t_no_model_data = None, None
        if samples[0]["t_input_ids"] is not None:
            t_model_data = {
                "input_ids": torch.ones(bs, self.args.t_max_length, dtype=torch.long) * self.pad_id,
                "attention_mask": torch.zeros(bs, self.args.t_max_length),
            }
            
            if self.args.model_type in ["gpt2"]:
                t_model_data["position_ids"] = torch.zeros(bs, self.args.t_max_length, dtype=torch.long)
                
            t_no_model_data = {
                "label": torch.ones(bs, self.args.t_max_length, dtype=torch.long) * -100,
            }

            for i, samp in enumerate(samples):
                self._process_lm(i, {"input_ids": samp["t_input_ids"]}, t_model_data, t_no_model_data, None)
        
        return model_data, no_model_data, gen_data, t_model_data, t_no_model_data


class LMEvalDataset(Dataset):
    def __init__(self, args, tokenizer, path, split, rng_sample: random.Random):
        self.args = args
        self.tokenizer = tokenizer
        self.split = split
        self.pad_id = self.tokenizer.eos_token_id
        self.max_length = args.max_length
        self.max_prompt_length = args.max_prompt_length
        self.rng_sample = rng_sample
        self.lm_ctx = DistributedMMapIndexedDataset(path, f"{split}", 0, 1)

        if os.path.exists(os.path.join(path, f"{split}.jsonl")):
            with open(os.path.join(path, f"{split}.jsonl")) as f:
                self.raw = [json.loads(line) for line in f.readlines()]
                self.answers = [x["response"] if isinstance(x["response"], list) else [x["response"]] for x in self.raw]
        
        self.num = len(self.lm_ctx)

        print(f"Num LM instances: {len(self.lm_ctx)}")

    def __len__(self):
        return self.num
   
    def __getitem__(self, index):
        return self._get_lm(index)
    
    def _get_lm(self, index):
        data = self.lm_ctx[index]
        input_ids = data.astype(int)
        return {
            "input_ids": input_ids
        }

    def _process_lm(self, i, samp, model_data, no_model_data, gen_data):
        input_ids = samp["input_ids"]
        source_len = 1
        
        prompt = None
        if self.args.model_type in ["qwen"] and 4294967295 in input_ids:
            source_len = np.where(input_ids==4294967295)[0][0]
            prompt = input_ids[:source_len]
            input_ids = np.concatenate([input_ids[:source_len], input_ids[source_len+1:]], axis=0)
        elif 65535 in input_ids:
            source_len = np.where(input_ids==65535)[0][0]
            prompt = input_ids[:source_len]
            input_ids = np.concatenate([input_ids[:source_len], input_ids[source_len+1:]], axis=0)
        
        input_ids = input_ids[:self.max_length]
        input_len = len(input_ids)
        model_data["input_ids"][i][:input_len-1] = torch.tensor(input_ids[:-1], dtype=torch.long)
        model_data["attention_mask"][i][:input_len-1] = 1.0
        if self.args.model_type in ["gpt2"]:
            model_data["position_ids"][i][:input_len-1] = torch.arange(0, input_len-1, dtype=torch.long)
        no_model_data["label"][i][:input_len-1] = torch.tensor(input_ids[1:], dtype=torch.long)
        no_model_data["label"][i][:source_len-1] = -100
        no_model_data["loss_mask"][i][:input_len-1] = 1.0
        no_model_data["loss_mask"][i][:source_len-1] = 0
        
        if prompt is not None:
            gen_data["input_ids"][i][-len(prompt):] = torch.tensor(prompt, dtype=torch.long)
            gen_data["attention_mask"][i][-len(prompt):] = 1.0

    def move_to_device(self, model_data, no_model_data, gen_data, device):
        for k in model_data:
            model_data[k] = model_data[k].to(device)

        for k in no_model_data:
            no_model_data[k] = no_model_data[k].to(device)

        for k in gen_data:
            gen_data[k] = gen_data[k].to(device)

        return model_data, no_model_data, gen_data

    def collate(self, samples):
        bs = len(samples)

        max_length = self.max_length
        
        model_data = {
            "input_ids": torch.ones(bs, max_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, max_length),
        }
        
        if self.args.model_type in ["gpt2"]:
            model_data["position_ids"] = torch.zeros(bs, max_length, dtype=torch.long)
            
        no_model_data = {
            "label": torch.ones(bs, max_length, dtype=torch.long) * -100,
            "loss_mask": torch.zeros(bs, max_length)
        }
        
        gen_data = {
            "input_ids": torch.ones(bs, self.max_prompt_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, self.max_prompt_length, dtype=torch.long),
        }

        for i, samp in enumerate(samples):
            self._process_lm(i, samp, model_data, no_model_data, gen_data)
        
        return model_data, no_model_data, gen_data



def label_window(label):
    """[start, end) over every column of `label` (rows x columns) that holds a label in some row.
    CE and KD only read logits at labelled positions, so the forward can skip the LM head outside
    this window (logits_to_keep). A batch with no label at all gets a 1-column window.
    From new_nothing/OpenED/ced_losses.py."""
    cols = (label != -100).any(dim=0).nonzero().flatten()
    if cols.numel() == 0:
        return 0, 1
    return int(cols[0]), int(cols[-1]) + 1


def balanced_quota(names, alpha):
    """Rows per epoch of each source dataset for --balance-alpha. Dataset d gets a share prop. to
    n_d**alpha; the epoch is as long as that allows without repeating a row, so the dataset with the
    largest share per row is used in full. alpha=1 gives every row once (a normal epoch)."""
    sizes = {}
    for d in names:
        sizes[d] = sizes.get(d, 0) + 1
    weight = {d: n ** alpha for d, n in sizes.items()}
    total = sum(weight.values())
    epoch_rows = min(n * total / weight[d] for d, n in sizes.items())
    return {d: min(n, round(epoch_rows * weight[d] / total)) for d, n in sorted(sizes.items())}


class BalancedSampler(torch.utils.data.Sampler):
    """--balance-alpha: each epoch takes balanced_quota() rows from every source dataset, without
    repeating a row inside the epoch. Each dataset is walked through a shuffled order, epoch after
    epoch, so every row is used before any row is used again; when a dataset's pass runs out, the
    next pass is reshuffled with the rows already taken this epoch moved to its end. The seed fixes
    the order only, so every rank (and a rerun) gets the same rows. The epoch is shuffled and split
    across ranks like DistributedSampler with drop_last=True."""

    def __init__(self, dataset, alpha, rank, num_replicas, seed=0, log_dir=None):
        names = [r["dataset"] for r in dataset.raw[:len(dataset)]]
        self.groups = {}
        for i, d in enumerate(names):
            self.groups.setdefault(d, []).append(i)
        self.quota = balanced_quota(names, alpha)
        # trim the largest dataset so the epoch splits evenly over ranks: no taken row is left out
        big = max(self.quota, key=self.quota.get)
        self.quota[big] -= sum(self.quota.values()) % num_replicas
        self.rank, self.num_replicas, self.seed, self.epoch = rank, num_replicas, seed, 0
        self.num_samples = sum(self.quota.values()) // num_replicas
        self.log_dir = log_dir
        print_rank(f"BalancedSampler alpha={alpha}: rows per epoch " + ", ".join(
            f"{d} {q}/{len(self.groups[d])}" for d, q in self.quota.items()) + f" (total {sum(self.quota.values())})")

    def set_epoch(self, epoch):
        self.epoch = epoch

    def epoch_indices(self, epoch):
        rng = random.Random(self.seed)
        queue = {d: [] for d in self.quota}
        for _ in range(epoch + 1):            # replay earlier epochs: the walk carries over
            chosen = []
            for d, q in self.quota.items():
                take = queue[d][:q]
                queue[d] = queue[d][q:]
                if len(take) < q:
                    order = self.groups[d][:]
                    rng.shuffle(order)
                    taken = set(take)
                    order = [i for i in order if i not in taken] + [i for i in order if i in taken]
                    need = q - len(take)
                    take, queue[d] = take + order[:need], order[need:]
                chosen += take
            rng.shuffle(chosen)
        return chosen[:self.num_samples * self.num_replicas]

    def __iter__(self):
        indices = self.epoch_indices(self.epoch)
        if self.log_dir is not None and self.rank == 0:
            os.makedirs(self.log_dir, exist_ok=True)
            with open(os.path.join(self.log_dir, f"epoch_{self.epoch}_indices.json"), "w") as f:
                json.dump(indices, f)
        return iter(indices[self.rank::self.num_replicas])

    def __len__(self):
        return self.num_samples
