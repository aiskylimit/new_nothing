import logging
import json
from pathlib import Path
from typing import Any

import numpy as np
import PIL
import torch
from transformers import AutoConfig, AutoProcessor, AutoTokenizer
from transformers.image_utils import ChannelDimension

from .vlm_backbone.fast_vlm import FastVlmForConditionalGeneration
from .vlm_backbone.llava_next import LlavaNextForConditionalGeneration
from .vlm_backbone.llava_onevision import LlavaOnevisionForConditionalGeneration
from .vlm_backbone.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from .vlm_backbone.qwen2_vl import Qwen2VLForConditionalGeneration
from .vlm_backbone.qwen3_vl import Qwen3VLForConditionalGeneration


logger = logging.getLogger(__name__)

FAST_VLM = "fast_vlm"
FAST_VLM_EOS_TOKEN = "<|im_end|>"
FAST_VLM_PAD_TOKEN = "<|endoftext|>"
LLAVA_NEXT = "llava_next"
LLAVA_ONEVISION = "llava_onevision"
QWEN2_VL = "qwen2_vl"
QWEN2_5_VL = "qwen2_5_vl"
QWEN3_VL = "qwen3_vl"


MODEL2BACKBONE = {
    FAST_VLM: FAST_VLM,
    LLAVA_NEXT: LLAVA_NEXT,
    LLAVA_ONEVISION: LLAVA_ONEVISION,
    QWEN2_VL: QWEN2_VL,
    QWEN2_5_VL: QWEN2_5_VL,
    QWEN3_VL: QWEN3_VL,
}

SUPPORTED_MODELS = set(MODEL2BACKBONE)

VLM_IMAGE_TOKENS = {
    FAST_VLM: "<image>",
    LLAVA_NEXT: "<image>",
    LLAVA_ONEVISION: "<image>",
    QWEN2_VL: "<|image_pad|>",
    QWEN2_5_VL: "<|image_pad|>",
    QWEN3_VL: "<|image_pad|>",
}

VLM_VIDEO_TOKENS = {
    FAST_VLM: "",
    LLAVA_NEXT: "<image>",
    LLAVA_ONEVISION: "<video>",
    QWEN2_VL: "<|video_pad|>",
    QWEN2_5_VL: "<|video_pad|>",
    QWEN3_VL: "<|video_pad|>",
}

backbone2model = {
    FAST_VLM: FastVlmForConditionalGeneration,
    LLAVA_NEXT: LlavaNextForConditionalGeneration,
    LLAVA_ONEVISION: LlavaOnevisionForConditionalGeneration,
    QWEN2_VL: Qwen2VLForConditionalGeneration,
    QWEN2_5_VL: Qwen2_5_VLForConditionalGeneration,
    QWEN3_VL: Qwen3VLForConditionalGeneration,
}


def _get_arg(args: Any, name: str, default=None):
    return getattr(args, name, default) if args is not None else default


def _get_model_name_or_path(model_args):
    return _get_arg(model_args, "checkpoint_path") or _get_arg(model_args, "processor_name") or _get_arg(model_args, "model_name")


def _require_token_id(tokenizer, token):
    token_id = tokenizer.convert_tokens_to_ids(token)
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    if token_id is None or token_id < 0 or (unk_token_id is not None and token_id == unk_token_id):
        raise ValueError(f"FastVLM tokenizer does not contain required special token {token!r}")
    return int(token_id)


def normalize_fast_vlm_tokenizer(tokenizer):
    """Match Apple's FastVLM chat semantics without modifying the HF cache."""
    eos_token_id = _require_token_id(tokenizer, FAST_VLM_EOS_TOKEN)
    pad_token_id = _require_token_id(tokenizer, FAST_VLM_PAD_TOKEN)
    tokenizer.eos_token = FAST_VLM_EOS_TOKEN
    tokenizer.pad_token = FAST_VLM_PAD_TOKEN
    if tokenizer.eos_token_id != eos_token_id or tokenizer.pad_token_id != pad_token_id:
        raise ValueError("FastVLM tokenizer rejected the normalized EOS/PAD tokens")
    return tokenizer


def normalize_fast_vlm_processor(processor):
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("FastVLM processor has no tokenizer")
    normalize_fast_vlm_tokenizer(tokenizer)
    return processor


def normalize_fast_vlm_config(config, eos_token_id, pad_token_id):
    """Apply the FastVLM EOS/PAD invariant to model and generation configs."""
    for target in (config, getattr(config, "text_config", None)):
        if target is not None:
            target.eos_token_id = int(eos_token_id)
            target.pad_token_id = int(pad_token_id)
    return config


def normalize_fast_vlm_model(model, tokenizer):
    eos_token_id = _require_token_id(tokenizer, FAST_VLM_EOS_TOKEN)
    pad_token_id = _require_token_id(tokenizer, FAST_VLM_PAD_TOKEN)
    normalize_fast_vlm_config(model.config, eos_token_id, pad_token_id)
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.eos_token_id = eos_token_id
        generation_config.pad_token_id = pad_token_id
    return model


def save_processor(processor, output_dir, model_backbone=None):
    """Save a processor and persist the normalized FastVLM token contract."""
    if model_backbone == FAST_VLM:
        normalize_fast_vlm_processor(processor)
    processor.save_pretrained(output_dir)
    if model_backbone == FAST_VLM:
        special = {
            "additional_special_tokens": ["<|im_start|>", FAST_VLM_EOS_TOKEN],
            "eos_token": {
                "content": FAST_VLM_EOS_TOKEN,
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
            },
            "pad_token": {
                "content": FAST_VLM_PAD_TOKEN,
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
            },
        }
        path = Path(output_dir) / "special_tokens_map.json"
        path.write_text(json.dumps(special, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _apply_resize_config(processor, data_args=None):
    if data_args is None:
        return processor

    min_pixels = _get_arg(data_args, "resize_min_pixels")
    max_pixels = _get_arg(data_args, "resize_max_pixels")
    do_resize = _get_arg(data_args, "resize_use_processor")

    for component_name in ("image_processor", "video_processor"):
        component = getattr(processor, component_name, None)
        if component is None:
            continue
        if min_pixels is not None and hasattr(component, "min_pixels"):
            component.min_pixels = min_pixels
        if max_pixels is not None and hasattr(component, "max_pixels"):
            component.max_pixels = max_pixels
        if do_resize is not None and hasattr(component, "do_resize"):
            component.do_resize = do_resize

    return processor


def _maybe_lora_base_model(model_args, model_name_or_path):
    if not _get_arg(model_args, "init_lora_model", False):
        return model_name_or_path
    try:
        from peft import PeftConfig

        return PeftConfig.from_pretrained(_get_arg(model_args, "model_name")).base_model_name_or_path
    except Exception as exc:
        logger.warning("Could not resolve LoRA base model for processor loading: %s", exc)
        return model_name_or_path


def load_processor(model_args, data_args=None):
    """
    Load the matching processor for a VLM_Distill backbone.
    """
    model_name_or_path = _get_model_name_or_path(model_args)
    if model_name_or_path is None:
        raise ValueError("model_args must provide checkpoint_path, processor_name, or model_name")

    model_backbone = _get_arg(model_args, "model_backbone")
    if not model_backbone:
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        model_backbone = get_backbone_name(config, model_type=_get_arg(model_args, "model_type"))
        setattr(model_args, "model_backbone", model_backbone)

    logger.info("Loading processor for backbone [%s] from %s", model_backbone, model_name_or_path)

    if model_backbone == LLAVA_NEXT:
        from .vlm_backbone.llava_next.processing_llava_next import LlavaNextProcessor

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        tokenizer.padding_side = "left"
        processor = LlavaNextProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            tokenizer=tokenizer,
        )

    elif model_backbone == LLAVA_ONEVISION:
        from .vlm_backbone.llava_onevision.processing_llava_onevision import LlavaOnevisionProcessor

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        tokenizer.padding_side = "left"
        processor = LlavaOnevisionProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            tokenizer=tokenizer,
        )

    elif model_backbone == QWEN2_VL:
        from .vlm_backbone.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
        from .vlm_backbone.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
        from .vlm_backbone.qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor

        model_name_or_path = _maybe_lora_base_model(model_args, model_name_or_path)
        image_processor = Qwen2VLImageProcessor.from_pretrained(model_name_or_path)
        video_processor = Qwen2VLVideoProcessor.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        tokenizer.padding_side = "left"
        processor = Qwen2VLProcessor.from_pretrained(
            model_name_or_path,
            image_processor=image_processor,
            video_processor=video_processor,
            tokenizer=tokenizer,
        )

    elif model_backbone == QWEN2_5_VL:
        processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            padding_side="left",
        )

    elif model_backbone == QWEN3_VL:
        processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            padding_side="left",
        )

    elif model_backbone == FAST_VLM:
        processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            padding_side="left",
        )
        processor = normalize_fast_vlm_processor(processor)

    else:
        raise NotImplementedError(f"Unsupported VLM backbone: {model_backbone}")

    return _apply_resize_config(processor, data_args)


def get_backbone_name(hf_config, model_type=None):
    if model_type is not None:
        setattr(hf_config, "model_type", model_type)
    model_type = hf_config.model_type
    assert model_type in SUPPORTED_MODELS, f"Unknown backbone name {model_type}. Supported models are {SUPPORTED_MODELS}"
    return MODEL2BACKBONE[model_type]


def _is_empty_visual(visual):
    if visual is None:
        return True
    if isinstance(visual, str):
        return visual == ""
    if isinstance(visual, (list, tuple)):
        return len(visual) == 0 or all(_is_empty_visual(item) for item in visual)
    return False


def _as_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _resize_tiny_images(images, min_size=28, resize_to=56):
    resized = []
    for image in images:
        if isinstance(image, PIL.Image.Image) and (image.size[0] < min_size or image.size[1] < min_size):
            image = image.resize((resize_to, resize_to))
        resized.append(image)
    return resized


def _to_list_input_ids(input_ids):
    if torch.is_tensor(input_ids):
        input_ids = input_ids.squeeze().tolist()
    elif isinstance(input_ids, np.ndarray):
        input_ids = input_ids.squeeze().tolist()
    if isinstance(input_ids, int):
        input_ids = [input_ids]
    return input_ids


def _pad_input_ids(processor, input_ids):
    batch_encoding = processor.tokenizer.pad({"input_ids": input_ids}, return_tensors="pt")
    return batch_encoding["input_ids"].long(), batch_encoding["attention_mask"].long()


def _safe_tensor_list(values):
    tensor_values = [value for value in values if value is not None]
    if not tensor_values:
        return None
    if all(torch.is_tensor(value) for value in tensor_values):
        try:
            return torch.cat(tensor_values, dim=0)
        except RuntimeError:
            return values
    return values


def Llava_NEXT_process_fn(model_inputs: dict, processor, max_length=None):
    input_ids, pixel_values, image_sizes = [], [], []
    texts, visual_inputs = model_inputs["text"], model_inputs["images"]

    for text, images in zip(texts, visual_inputs):
        if _is_empty_visual(images):
            inputs = processor(images=None, text=text, return_tensors="pt", max_length=max_length, truncation=True)
            input_ids.append(_to_list_input_ids(inputs["input_ids"]))
            pixel_values.append(None)
            image_sizes.append(None)
            continue

        images = _as_list(images)
        inputs = processor(
            images=images,
            text=text,
            return_tensors="pt",
            max_length=max_length,
            truncation=max_length is not None,
            input_data_format=ChannelDimension.LAST,
        )
        input_ids.append(_to_list_input_ids(inputs["input_ids"]))
        pixel_values.append(inputs.get("pixel_values"))
        image_sizes.append(inputs.get("image_sizes"))

    padded_input_ids, attention_mask = _pad_input_ids(processor, input_ids)
    return {
        "input_ids": padded_input_ids,
        "attention_mask": attention_mask,
        "pixel_values": _safe_tensor_list(pixel_values),
        "image_sizes": _safe_tensor_list(image_sizes),
        "texts": texts,
        "images": visual_inputs,
    }


def Llava_ONEVISION_process_fn(model_inputs: dict, processor, max_length=None):
    texts, visual_inputs = model_inputs["text"], model_inputs["images"]
    has_visual = not all(_is_empty_visual(images) for images in visual_inputs)

    if not has_visual:
        inputs = processor(
            images=None,
            text=texts,
            return_tensors="pt",
            max_length=max_length,
            truncation=max_length is not None,
            padding=True,
        )
    else:
        inputs = processor(
            images=visual_inputs,
            text=texts,
            return_tensors="pt",
            max_length=max_length,
            truncation=max_length is not None,
            padding=True,
            input_data_format=ChannelDimension.LAST,
        )

    inputs["input_ids"] = inputs["input_ids"].long()
    inputs["attention_mask"] = inputs["attention_mask"].long()
    inputs["texts"] = texts
    inputs["images"] = visual_inputs
    return inputs


def FastVLM_process_fn(model_inputs: dict, processor, max_length=None):
    texts, visual_inputs = model_inputs["text"], model_inputs["images"]
    kwargs = {
        "images": visual_inputs,
        "text": texts,
        "return_tensors": "pt",
        "padding": True,
    }
    if max_length is not None:
        kwargs.update({"max_length": max_length, "truncation": True})
    inputs = processor(**kwargs)
    if "input_ids" in inputs:
        inputs["input_ids"] = inputs["input_ids"].long()
    if "attention_mask" in inputs:
        inputs["attention_mask"] = inputs["attention_mask"].long()
    inputs["texts"] = texts
    inputs["images"] = visual_inputs
    return inputs


def Qwen_VL_process_fn(model_inputs: dict, processor, max_length=None):
    input_ids = []
    pixel_values, image_grid_thw = [], []
    pixel_values_videos, video_grid_thw = [], []
    second_per_grid_ts = []
    texts, visual_inputs = model_inputs["text"], model_inputs["images"]

    image_token = getattr(processor, "image_token", VLM_IMAGE_TOKENS[QWEN2_VL])
    video_token = getattr(processor, "video_token", VLM_VIDEO_TOKENS[QWEN2_VL])

    for text, visual in zip(texts, visual_inputs):
        if _is_empty_visual(visual):
            inputs = processor(text=[text], images=None, videos=None, return_tensors="pt", max_length=max_length, truncation=True)
            input_ids.append(_to_list_input_ids(inputs["input_ids"]))
            pixel_values.append(None)
            image_grid_thw.append(None)
            pixel_values_videos.append(None)
            video_grid_thw.append(None)
            second_per_grid_ts.append(None)
            continue

        visual_list = _as_list(visual)
        common_kwargs = {
            "text": [text],
            "return_tensors": "pt",
            "max_length": None,
            "truncation": False,
            "input_data_format": ChannelDimension.LAST,
        }
        if image_token in text:
            images = _resize_tiny_images(visual_list)
            inputs = processor(images=images, videos=None, **common_kwargs)
        elif video_token and video_token in text:
            inputs = processor(images=None, videos=[visual_list], **common_kwargs)
        else:
            raise NotImplementedError(f"Text does not contain an expected visual token: {text}")

        input_ids.append(_to_list_input_ids(inputs["input_ids"]))
        pixel_values.append(inputs.get("pixel_values"))
        image_grid_thw.append(inputs.get("image_grid_thw"))
        pixel_values_videos.append(inputs.get("pixel_values_videos"))
        video_grid_thw.append(inputs.get("video_grid_thw"))
        second_per_grid_ts.append(inputs.get("second_per_grid_ts"))

    padded_input_ids, attention_mask = _pad_input_ids(processor, input_ids)
    return {
        "input_ids": padded_input_ids,
        "attention_mask": attention_mask,
        "pixel_values": _safe_tensor_list(pixel_values),
        "image_grid_thw": _safe_tensor_list(image_grid_thw),
        "pixel_values_videos": _safe_tensor_list(pixel_values_videos),
        "video_grid_thw": _safe_tensor_list(video_grid_thw),
        "second_per_grid_ts": _safe_tensor_list(second_per_grid_ts),
        "texts": texts,
        "images": visual_inputs,
    }


def process_input_text(instruction, model_backbone, text=None, add_video_token=False, add_image_token=False):
    prompt = instruction
    if text:
        prompt = f"{prompt} {text}"
    if add_video_token:
        prompt = f"{VLM_VIDEO_TOKENS[model_backbone]} {prompt}"
    if add_image_token:
        prompt = f"{VLM_IMAGE_TOKENS[model_backbone]} {prompt}"
    return prompt


process_vlm_inputs_fns = {
    FAST_VLM: FastVLM_process_fn,
    LLAVA_NEXT: Llava_NEXT_process_fn,
    LLAVA_ONEVISION: Llava_ONEVISION_process_fn,
    QWEN2_VL: Qwen_VL_process_fn,
    QWEN2_5_VL: Qwen_VL_process_fn,
    QWEN3_VL: Qwen_VL_process_fn,
}
