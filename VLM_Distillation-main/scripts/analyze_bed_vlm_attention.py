"""Generate BeD-VLM attention overlays and adjacent-layer statistics for GQA."""

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean, pstdev

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.arguments import DataArguments, ModelArguments
from src.criterions.scva import _grid_from_inputs
from src.data.dataset import VlmDistillDataCollator, _load_and_prepare_image, _message_content
from src.model.model import VLMModel
from src.model.processor import load_processor
from src.model.scva_attention import resolve_layer_pairs


def to_device(inputs, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}


def prompt(image, question):
    return [{"role": "user", "content": _message_content("user", question, image)}]


def attention_rows(model, collator, messages, answer, device):
    full = messages + [{"role": "assistant", "content": _message_content("assistant", answer, None)}]
    instance = {"messages": full, "id": None, "image_path": None}
    inputs = to_device(collator([instance])["student_inputs"], device)
    output = model(**inputs)
    patches = output.vision_feature_mask[0].nonzero(as_tuple=True)[0]
    response = inputs["labels"][0].ne(-100).nonzero(as_tuple=True)[0]
    if not patches.numel() or not response.numel() or not output.attentions:
        raise ValueError("The example needs vision tokens, assistant tokens, and decoder attentions.")
    rows = torch.stack([
        layer[0][:, response[:, None], patches].float().mean(dim=0)
        for layer in output.attentions
    ])
    return F.normalize(rows, p=1, dim=-1), _grid_from_inputs(inputs, 0, patches.numel())


def save_overlay(image, patch_attention, grid, path):
    intensity = patch_attention.reshape(grid).cpu().numpy()
    low, high = float(intensity.min()), float(intensity.max())
    intensity = (intensity - low) / max(high - low, 1e-8)
    opacity = Image.fromarray((180 * intensity).astype(np.uint8)).resize(image.size, Image.Resampling.BILINEAR)
    highlight = Image.new("RGB", image.size, (255, 48, 0))
    Image.composite(highlight, image, opacity).save(path)


@torch.inference_mode()
def analyze_model(label, base, checkpoint, examples, args):
    model_args = ModelArguments(
        model_name=base,
        checkpoint_path=checkpoint,
        lora=bool(checkpoint),
    )
    data_args = DataArguments(image_resolution=args.image_resolution)
    model = VLMModel.load(model_args, is_trainable=False, output_attentions=True,
                          vision_output_attentions=False).to(args.device).eval()
    processor = load_processor(model_args, data_args)
    tokenizer = getattr(processor, "tokenizer", processor)
    collator = VlmDistillDataCollator(processor, data_args, model_args)
    layer_scores = {}
    for index, sample in enumerate(examples):
        image = _load_and_prepare_image(str(args.examples.parent / sample["image"]), args.image_resolution)
        messages = prompt(image, sample["question"])
        generate_inputs = processor.apply_chat_template(
            [messages], tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        generated = model.generate(**to_device(generate_inputs, args.device), do_sample=False,
                                   max_new_tokens=args.max_new_tokens, output_attentions=False,
                                   output_hidden_states=False)
        answer = tokenizer.decode(generated[0, generate_inputs["input_ids"].size(1):], skip_special_tokens=True)
        sample_dir = args.output_dir / f"example-{index + 1:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        if label == "teacher":
            image.save(sample_dir / "input_image.png")
            metadata = {"question": sample["question"], "ground_truth": sample["answer"], "predictions": {}}
        else:
            metadata = json.loads((sample_dir / "answers.json").read_text(encoding="utf-8"))
        metadata["predictions"][label] = answer
        (sample_dir / "answers.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        rows, grid = attention_rows(model, collator, messages, answer, args.device)
        layers = [left for left, _ in resolve_layer_pairs(len(rows), len(rows), 4, 0.0, 0.4, half_open=True)]
        save_overlay(image, rows[layers].mean(dim=(0, 1)), grid, sample_dir / f"{label}_attention_overlay.png")
        if label == "base_student":
            rows, _ = attention_rows(model, collator, messages, sample["answer"], args.device)
            scores = F.cosine_similarity(rows[:-1], rows[1:], dim=-1).mean(dim=1).tolist()
            for layer, score in enumerate(scores, start=1):
                depth = layer / len(rows)
                layer_scores.setdefault(depth, []).append(score)
    return layer_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", required=True, type=Path, help="JSONL with image, question, and answer fields.")
    parser.add_argument("--student", default="KamilaMila/FastVLM-0.5B")
    parser.add_argument("--teacher", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--checkpoint", required=True, help="BeD-VLM LoRA adapter directory.")
    parser.add_argument("--output-dir", default=Path("outputs/bed_vlm_attention"), type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", default=24, type=int)
    parser.add_argument("--image-resolution", default="low")
    args = parser.parse_args()
    examples = [json.loads(line) for line in args.examples.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not examples:
        parser.error("--examples must contain at least one GQA example")
    analyze_model("teacher", args.teacher, None, examples, args)
    layer_scores = analyze_model("base_student", args.student, None, examples, args)
    analyze_model("bed_vlm", args.student, args.checkpoint, examples, args)
    with (args.output_dir / "attention_adjacent_layer_similarity.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("normalized_depth", "mean_cosine", "std_cosine", "examples"))
        writer.writerows((depth, fmean(scores), pstdev(scores), len(scores))
                         for depth, scores in sorted(layer_scores.items()))


if __name__ == "__main__":
    main()
