import argparse
import os
import sys
from pathlib import Path

import logging

import torch
from PIL import Image


def disable_verbose_logs():
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["HF_HUB_VERBOSITY"] = "error"

    for name in [
        "httpcore",
        "httpcore.http11",
        "httpcore.connection",
        "httpx",
        "urllib3",
        "huggingface_hub",
        "transformers",
    ]:
        logger = logging.getLogger(name)
        logger.setLevel(logging.WARNING)
        logger.propagate = False


disable_verbose_logs()


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.arguments import ModelArguments
from src.model.model import VLMModel
from src.model.processor import load_processor


DEFAULT_IMAGE_PATH = PROJECT_ROOT / "assets" / "a_cute_dog.png"
DEFAULT_QUESTION = "what is the animal in the image? Only answer the animal name, no other text"
DEFAULT_QUESTION_2 = "what is the animal in the image? Only answer the animal name"


def parse_args(model_id):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=model_id)
    parser.add_argument("--image-path", default=str(DEFAULT_IMAGE_PATH))
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--question-2", default=DEFAULT_QUESTION_2)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--use-device-map", action="store_true")
    parser.add_argument("--disable-device-map", action="store_true")
    return parser.parse_args()


def build_messages(image, question):
    return [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a helpful assistant."},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        },
    ]


def prepare_inputs(processor, image, questions):
    """
    Build a batch of Qwen3-VL style multimodal inputs.

    Batch size = len(questions).
    All samples use the same image, but each sample can have a different question.
    """
    conversations = [
        build_messages(image=image, question=question)
        # build_messages(image=None, question=question)
        for question in questions
    ]

    if not hasattr(processor, "apply_chat_template"):
        raise RuntimeError(
            "This processor does not support apply_chat_template. "
            "Qwen3-VL requires a processor with apply_chat_template support."
        )

    return processor.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )


def has_device_map(module):
    return hasattr(module, "hf_device_map") or hasattr(getattr(module, "encoder", None), "hf_device_map")


def maybe_place_model(model):
    if torch.cuda.is_available() and not has_device_map(model):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        model.to(torch.device("cuda", local_rank))
    return model


def move_batch_to_model(batch, model):
    """
    Move tensor inputs to the model device.

    If the model is loaded with device_map, do not manually move the batch because
    Hugging Face Accelerate will handle placement.
    """
    if has_device_map(model):
        return batch

    device = next(model.parameters()).device
    moved = {}

    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value

    return moved


def check_outputs(outputs):
    required = [
        "logits",
        "hidden_states",
        "attentions",
        "vision_feature_mask",
        "text_feature_mask",
    ]

    missing = [name for name in required if not hasattr(outputs, name)]
    if missing:
        raise AssertionError(f"Missing output fields: {missing}")

    if outputs.logits is None:
        raise AssertionError("outputs.logits is None")
    if outputs.hidden_states is None:
        raise AssertionError("outputs.hidden_states is None")
    if outputs.attentions is None:
        raise AssertionError("outputs.attentions is None")
    if outputs.vision_feature_mask is None:
        raise AssertionError("outputs.vision_feature_mask is None")
    if outputs.text_feature_mask is None:
        raise AssertionError("outputs.text_feature_mask is None")

    if outputs.vision_feature_mask.sum().item() == 0:
        print("No vision tokens marked by vision_feature_mask")
        # raise AssertionError("vision_feature_mask does not mark any token")
    if outputs.text_feature_mask.sum().item() == 0:
        print("Warning: text_feature_mask does not mark any token")
        # raise AssertionError("text_feature_mask does not mark any token")


def decode_generated(processor, batch, generated):
    """
    Decode only newly generated tokens for each sample in the batch.
    """
    if "input_ids" not in batch:
        return processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    generated_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(batch["input_ids"], generated)
    ]

    return processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def print_batch_shapes(batch):
    print("batch keys:", list(batch.keys()))
    for key, value in batch.items():
        if torch.is_tensor(value):
            print(f"{key}: shape={tuple(value.shape)}, dtype={value.dtype}")
            if key == "image_grid_thw":
                print(f"sample image_grid_thw[0]: {value[0]}")
                print(f"sample image_grid_thw[-1]: {value[-1]}")
        else:
            print(f"{key}: {type(value)}")


def run_load_test(model_id):
    args = parse_args(model_id)
    model_args = ModelArguments(model_name=args.model_name)

    processor = load_processor(model_args)
    model = VLMModel.build(
        model_args,
        use_device_map=args.use_device_map,
        disable_device_map=args.disable_device_map,
    )
    model.eval()
    maybe_place_model(model)

    image = Image.open(args.image_path).convert("RGB")

    questions = [
        args.question,
        args.question_2,
    ]

    print("questions:")
    for idx, question in enumerate(questions):
        print(f"[{idx}] {question}")

    batch = prepare_inputs(
        processor=processor,
        image=image,
        questions=questions,
    )

    print_batch_shapes(batch)

    batch = move_batch_to_model(batch, model)

    with torch.no_grad():
        outputs = model(**batch)

    check_outputs(outputs)

    print(f"Loaded: {args.model_name}")
    print(f"logits: {tuple(outputs.logits.shape)}")
    print(f"hidden_states: {len(outputs.hidden_states)} layers")
    print(f"attentions: {len(outputs.attentions)} layers")

    if outputs.vision_feature_mask.ndim >= 2:
        vision_tokens_per_sample = outputs.vision_feature_mask.sum(dim=1).tolist()
        text_tokens_per_sample = outputs.text_feature_mask.sum(dim=1).tolist()
        print("vision tokens per sample:", [int(x) for x in vision_tokens_per_sample])
        print("text tokens per sample:", [int(x) for x in text_tokens_per_sample])
    else:
        print(f"vision tokens: {int(outputs.vision_feature_mask.sum().item())}")
        print(f"text tokens: {int(outputs.text_feature_mask.sum().item())}")

    if not args.skip_generate:
        with torch.no_grad():
            generated = model.generate(
                **batch,
                max_new_tokens=args.max_new_tokens,
            )

        decoded = decode_generated(processor, batch, generated)

        for idx, text in enumerate(decoded):
            print(f"generated[{idx}]:", text)


if __name__ == "__main__":
    run_load_test("Qwen/Qwen3-VL-2B-Instruct")
