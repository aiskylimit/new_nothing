import json
import os
import pickle
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import HfArgumentParser

from evaluation.mmeb_baselines.eval_utils import get_pred
from src.arguments import DataArguments, ModelArguments, TrainingArguments
from src.centroid import CentroidArguments, SinkhornCentroidDescriptor
from src.centroid.utils import encode_centroid_descriptor, make_runtime_data_args, move_to_device
from src.data.collator.eval_collator import EvalCollator
from src.data.dataset.mmeb_dataset import EvalDataset
from train_centroid_ddp import get_hidden_dim, load_teacher


POS_MOD_CLASS_LABEL = "Represent the class label: "
POS_MOD_IMAGE_CAPTION = "Represent the image caption: "
POS_MOD_ANSWER = "Represent the answer: "

POS_MOD_DICT = {
    "ImageNet-1K": POS_MOD_CLASS_LABEL,
    "ImageNet_1K": POS_MOD_CLASS_LABEL,
    "HatefulMemes": POS_MOD_CLASS_LABEL,
    "SUN397": POS_MOD_CLASS_LABEL,
    "N24News": POS_MOD_CLASS_LABEL,
    "VOC2007": POS_MOD_CLASS_LABEL,
    "Place365": POS_MOD_CLASS_LABEL,
    "ImageNet-A": POS_MOD_CLASS_LABEL,
    "ImageNet-R": POS_MOD_CLASS_LABEL,
    "ObjectNet": POS_MOD_CLASS_LABEL,
    "Country211": POS_MOD_CLASS_LABEL,
    "OK-VQA": POS_MOD_ANSWER,
    "A-OKVQA": POS_MOD_ANSWER,
    "DocVQA": POS_MOD_ANSWER,
    "InfographicsVQA": POS_MOD_ANSWER,
    "ChartQA": POS_MOD_ANSWER,
    "Visual7W": POS_MOD_ANSWER,
    "ScienceQA": POS_MOD_ANSWER,
    "GQA": POS_MOD_ANSWER,
    "TextVQA": POS_MOD_ANSWER,
    "VizWiz": POS_MOD_ANSWER,
    "MSCOCO_i2t": POS_MOD_IMAGE_CAPTION,
    "VisualNews_i2t": POS_MOD_IMAGE_CAPTION,
}


def ddp_setup():
    if "LOCAL_RANK" not in os.environ:
        return
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_count = torch.cuda.device_count()
    if local_rank >= device_count:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but only {device_count} CUDA device(s) are visible. "
            "Lower --nproc_per_node/NUM_GPUS_PER_NODE or set CUDA_VISIBLE_DEVICES to expose more GPUs."
        )
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")


def is_main_process():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def get_device():
    if torch.cuda.is_available():
        return torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
    return torch.device("cpu")


def load_centroid_head(centroid_args, model_args, teacher, device):
    state = torch.load(centroid_args.centroid_checkpoint, map_location="cpu")
    saved_args = state.get("centroid_args", {})
    model_state = state["model"]
    input_dim = state.get("input_dim", get_hidden_dim(model_args, teacher))
    hidden_dim = state.get(
        "hidden_dim",
        saved_args.get("centroid_hidden_dim", model_state["centroids"].shape[1]),
    )
    centroid_head = SinkhornCentroidDescriptor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_centroids=saved_args.get("num_centroids", centroid_args.num_centroids),
        centroid_dim=saved_args.get("centroid_dim", centroid_args.centroid_dim),
        sinkhorn_epsilon=saved_args.get("sinkhorn_epsilon", centroid_args.sinkhorn_epsilon),
        sinkhorn_iters=saved_args.get("sinkhorn_iters", centroid_args.sinkhorn_iters),
        dustbin_mass=saved_args.get("dustbin_mass", centroid_args.dustbin_mass),
    ).to(device)
    centroid_head.load_state_dict(model_state)
    centroid_head.eval()
    return centroid_head


def encode_loader(teacher, centroid_head, dataloader, tokenizer, centroid_args, device, desc):
    encoded = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            batch = move_to_device(batch, device)
            descs, _ = encode_centroid_descriptor(
                teacher,
                centroid_head,
                batch,
                tokenizer=tokenizer,
                layer_idx=centroid_args.centroid_layer_idx,
                drop_special_tokens=centroid_args.drop_special_tokens,
            )
            encoded.append(descs.cpu().detach().float().numpy())
    return np.concatenate(encoded, axis=0)


def build_eval_loader(data_args, model_args, processor, subset, text_field, img_path_field, batch_size, mod_instruction=None):
    dataset = EvalDataset(
        data_args=data_args,
        model_args=model_args,
        subset=subset,
        text_field=text_field,
        img_path_field=img_path_field,
        mod_instruction=mod_instruction,
    )
    collator = EvalCollator(
        data_args=data_args,
        model_args=model_args,
        processor=processor,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    return dataset, dataloader


def encode_subset(teacher, centroid_head, processor, tokenizer, data_args, teacher_args, training_args, centroid_args, subset):
    encode_qry_path = os.path.join(data_args.encode_output_path, f"{subset}_qry")
    encode_tgt_path = os.path.join(data_args.encode_output_path, f"{subset}_tgt")
    if os.path.exists(encode_qry_path) and os.path.exists(encode_tgt_path):
        return

    batch_size = training_args.per_device_eval_batch_size
    eval_qry_dataset, eval_qry_loader = build_eval_loader(
        data_args,
        teacher_args,
        processor,
        subset,
        text_field="qry_text",
        img_path_field="qry_img_path",
        batch_size=batch_size,
    )
    eval_tgt_dataset, eval_tgt_loader = build_eval_loader(
        data_args,
        teacher_args,
        processor,
        subset,
        text_field="tgt_text",
        img_path_field="tgt_img_path",
        batch_size=batch_size,
        mod_instruction=POS_MOD_DICT.get(subset, None) if data_args.tgt_prefix_mod else None,
    )

    device = next(centroid_head.parameters()).device
    if not os.path.exists(encode_qry_path):
        qry_tensor = encode_loader(
            teacher,
            centroid_head,
            eval_qry_loader,
            tokenizer,
            centroid_args,
            device,
            desc=f"Encode query - {subset}",
        )
        with open(encode_qry_path, "wb") as f:
            pickle.dump((qry_tensor, eval_qry_dataset.paired_data), f)

    if not os.path.exists(encode_tgt_path):
        tgt_tensor = encode_loader(
            teacher,
            centroid_head,
            eval_tgt_loader,
            tokenizer,
            centroid_args,
            device,
            desc=f"Encode target - {subset}",
        )
        with open(encode_tgt_path, "wb") as f:
            pickle.dump((tgt_tensor, eval_tgt_dataset.paired_data), f)


def score_subset(data_args, teacher_args, subset):
    score_path = os.path.join(data_args.encode_output_path, f"{subset}_score.json")
    if os.path.exists(score_path):
        with open(score_path, "r") as f:
            score_dict = json.load(f)
        print(f"Found previous eval score, skipping {subset}")
        print(score_dict)
        return score_dict

    encode_qry_path = os.path.join(data_args.encode_output_path, f"{subset}_qry")
    encode_tgt_path = os.path.join(data_args.encode_output_path, f"{subset}_tgt")
    with open(encode_qry_path, "rb") as f:
        qry_tensor, qry_index = pickle.load(f)
    with open(encode_tgt_path, "rb") as f:
        tgt_tensor, tgt_index = pickle.load(f)

    eval_data = load_dataset(
        data_args.dataset_name,
        subset,
        split=data_args.dataset_split,
    )
    if (subset == "WebQA" or subset == "EDIS") and "qry_text" in eval_data.column_names and teacher_args.model_backbone == "llava_qwen2":
        eval_data = eval_data.map(lambda x: {"qry_text": x["qry_text"].replace("<|image_1|>", "").strip()})

    qry_key2emb, tgt_key2emb = OrderedDict(), OrderedDict()
    for qry_t, tt in zip(qry_tensor, qry_index):
        qry_key2emb[(tt["text"], tt["img_path"])] = qry_t
    for tgt_t, tt in zip(tgt_tensor, tgt_index):
        tgt_key2emb[(tt["text"], tt["img_path"])] = tgt_t

    n_correct = 0
    all_pred = []
    for row in tqdm(eval_data, desc=f"calculate score for {subset}"):
        qry_t = qry_key2emb[(row["qry_text"], row["qry_img_path"])]
        tgt_t, all_candidates = [], []
        for tt in zip(row["tgt_text"], row["tgt_img_path"]):
            tgt_t.append(tgt_key2emb[tt])
            all_candidates.append(tt)
        _, pred = get_pred(
            np.asarray(qry_t),
            np.stack(tgt_t, axis=0),
            normalization=True,
        )
        if pred == 0:
            n_correct += 1
        all_pred.append(all_candidates[pred])

    acc = n_correct / len(eval_data)
    score_dict = {
        "acc": acc,
        "num_correct": n_correct,
        "num_pred": len(all_pred),
        "num_data": len(eval_data),
    }
    print(f"\033[91m{subset} accuracy: {acc}\033[0m")
    print(score_dict)
    with open(score_path, "w") as f:
        json.dump(score_dict, f, indent=4)
    with open(os.path.join(data_args.encode_output_path, f"{subset}_pred.txt"), "w") as f:
        for item in all_pred:
            f.write(f"{item}\n")
    return score_dict


def main():
    for arg in sys.argv:
        if arg.startswith("--local-rank="):
            rank = arg.split("=")[1]
            sys.argv.remove(arg)
            sys.argv.append("--local_rank")
            sys.argv.append(rank)

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, CentroidArguments))
    model_args, data_args, training_args, centroid_args = parser.parse_args_into_dataclasses()
    if centroid_args.centroid_checkpoint is None:
        raise ValueError("--centroid_checkpoint is required for eval_centroid_ddp.py")

    if data_args.encode_output_path is None:
        data_args.encode_output_path = os.path.join(training_args.output_dir, "centroid_mmeb_encode")
    os.makedirs(data_args.encode_output_path, exist_ok=True)

    if not is_main_process():
        return

    runtime_data_args = make_runtime_data_args(data_args)
    device = get_device()
    teacher, teacher_args, processor, tokenizer = load_teacher(model_args, device)
    teacher.eval()
    centroid_head = load_centroid_head(centroid_args, model_args, teacher, device)

    for idx, subset in enumerate(runtime_data_args.subset_name):
        score_path = os.path.join(runtime_data_args.encode_output_path, f"{subset}_score.json")
        if os.path.exists(score_path):
            print(f"Found previous eval score, skipping encode for {subset}")
            continue
        print(f"\033[91m{idx + 1}/{len(runtime_data_args.subset_name)}: Processing {subset} now!\033[0m")
        encode_subset(
            teacher,
            centroid_head,
            processor,
            tokenizer,
            runtime_data_args,
            teacher_args,
            training_args,
            centroid_args,
            subset,
        )

    for subset in tqdm(runtime_data_args.subset_name, desc="Iterate datasets to calculate scores"):
        print(f"\033[91m{subset}: Calculating score now!\033[0m")
        score_subset(runtime_data_args, teacher_args, subset)


if __name__ == "__main__":
    ddp_setup()
    main()
    if dist.is_initialized():
        dist.destroy_process_group()
