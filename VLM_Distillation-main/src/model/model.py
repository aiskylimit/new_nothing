import os

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

from src.arguments import ModelArguments
from src.model.processor import (
    FAST_VLM,
    LLAVA_NEXT,
    LLAVA_ONEVISION,
    QWEN2_5_VL,
    QWEN2_VL,
    QWEN3_VL,
    backbone2model,
    get_backbone_name,
)
from src.model.vlm_backbone.llava_next import LlavaNextForConditionalGeneration
from src.model.vlm_backbone.llava_onevision import LlavaOnevisionForConditionalGeneration

try:
    from src.utils import print_master
except Exception:
    def print_master(*args, **kwargs):
        print(*args, **kwargs)


class VLMModel(nn.Module):
    """
    Common wrapper around a VLM generation model.

    The forward pass returns the raw generation output fields from the underlying
    backbone: loss, logits, hidden_states, attentions, vision_feature_mask,
    text_feature_mask, and any model-specific extras.
    """

    TRANSFORMER_CLS = AutoModelForCausalLM

    def __init__(self, encoder: PreTrainedModel, output_attentions: bool = True):
        super().__init__()
        self.encoder = encoder
        self.config = encoder.config
        self.output_attentions = output_attentions

    @staticmethod
    def _distributed_context():
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        distributed = world_size > 1

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            distributed = True
            world_size = torch.distributed.get_world_size()
            if local_rank < 0:
                local_rank = torch.distributed.get_rank()

        return distributed, local_rank, world_size

    @staticmethod
    def _set_config_attr_if_present(config, name, value):
        if config is not None and hasattr(config, name):
            setattr(config, name, value)

    @classmethod
    def _force_eager_attention(cls, config, vision_output_attentions=True, output_attentions=True):
        config.use_cache = False
        if output_attentions:
            # Transformers generally needs eager attention when callers request
            # the full attention matrices.  Keep that behaviour for criteria
            # such as SCVA/SRE, but leave the configured efficient backend alone
            # when a criterion (for example DWA-KD) does not consume attentions.
            config._attn_implementation = "eager"
            config.attn_implementation = "eager"
        config.output_attentions = output_attentions
        config.output_hidden_states = True

        for sub_config_name in ("text_config", "vision_config", "vision_config_2", "audio_config"):
            sub_config = getattr(config, sub_config_name, None)
            if sub_config is None:
                continue
            if output_attentions:
                sub_config._attn_implementation = "eager"
                sub_config.attn_implementation = "eager"
            sub_output_attentions = (
                vision_output_attentions if sub_config_name.startswith("vision_config") else output_attentions
            )
            cls._set_config_attr_if_present(
                sub_config,
                "output_attentions",
                sub_output_attentions,
            )
            cls._set_config_attr_if_present(sub_config, "output_hidden_states", True)
            cls._set_config_attr_if_present(sub_config, "use_cache", False)

        return config

    @staticmethod
    def _model_path(model_args: ModelArguments):
        return model_args.checkpoint_path if model_args.checkpoint_path else model_args.model_name

    @staticmethod
    def _clean_model_inputs(model_inputs):
        ignore_keys = {
            "texts",
            "text",
            "images",
            "global_dataset_name",
            "dataset_name",
            "pooler_safe_idx",
            "pooler_mask",
            "offset_mapping",
        }
        cleaned = {}
        for key, value in model_inputs.items():
            if key in ignore_keys or value is None:
                continue
            cleaned[key] = value
        return cleaned

    @staticmethod
    def _move_tensor_lists_to_device(model_inputs, device):
        for key, value in list(model_inputs.items()):
            if isinstance(value, list) and any(torch.is_tensor(item) for item in value if item is not None):
                tensors = [item.to(device) for item in value if item is not None]
                if not tensors:
                    model_inputs[key] = None
                    continue
                try:
                    model_inputs[key] = torch.cat(tensors, dim=0)
                except RuntimeError:
                    model_inputs[key] = [item.to(device) if torch.is_tensor(item) else item for item in value]
        return model_inputs

    def _model_dtype(self):
        try:
            return next(self.encoder.parameters()).dtype
        except StopIteration:
            return torch.float32

    @classmethod
    def _cast_floating_tensors(cls, value, dtype):
        if torch.is_tensor(value):
            return value.to(dtype=dtype) if torch.is_floating_point(value) else value
        if isinstance(value, list):
            return [cls._cast_floating_tensors(item, dtype) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._cast_floating_tensors(item, dtype) for item in value)
        if isinstance(value, dict):
            return {key: cls._cast_floating_tensors(item, dtype) for key, item in value.items()}
        return value

    def _cast_inputs_to_model_dtype(self, model_inputs):
        dtype = self._model_dtype()
        for key, value in list(model_inputs.items()):
            model_inputs[key] = self._cast_floating_tensors(value, dtype)
        return model_inputs

    def _forward_defaults(self):
        return {
            "return_dict": True,
            "output_hidden_states": True,
            "output_attentions": self.output_attentions,
            "use_cache": False,
        }

    def forward(self, **model_inputs):
        model_inputs = self._clean_model_inputs(model_inputs)
        if "input_ids" in model_inputs and torch.is_tensor(model_inputs["input_ids"]):
            model_inputs = self._move_tensor_lists_to_device(model_inputs, model_inputs["input_ids"].device)
        model_inputs = self._cast_inputs_to_model_dtype(model_inputs)

        forward_kwargs = self._forward_defaults()
        forward_kwargs.update(model_inputs)
        forward_kwargs["return_dict"] = True
        forward_kwargs["output_hidden_states"] = True
        forward_kwargs["output_attentions"] = self.output_attentions
        forward_kwargs["use_cache"] = False

        return self.encoder(**forward_kwargs)

    def generate(self, **model_inputs):
        generation_kwargs = self._clean_model_inputs(model_inputs)
        if "input_ids" in model_inputs and torch.is_tensor(model_inputs["input_ids"]):
            generation_kwargs = self._move_tensor_lists_to_device(generation_kwargs, model_inputs["input_ids"].device)
        generation_kwargs = self._cast_inputs_to_model_dtype(generation_kwargs)
        return self.encoder.generate(**generation_kwargs)

    @classmethod
    def _prepare_load_kwargs(cls, kwargs):
        load_kwargs = dict(kwargs)
        disable_device_map = load_kwargs.pop("disable_device_map", False)
        use_device_map = load_kwargs.pop("use_device_map", False)
        distributed, local_rank, _world_size = cls._distributed_context()
        using_engine_parallelism = (
            os.environ.get("ACCELERATE_USE_DEEPSPEED", "").lower() == "true"
            or os.environ.get("ACCELERATE_USE_FSDP", "").lower() == "true"
        )

        if distributed and torch.cuda.is_available() and local_rank >= 0:
            torch.cuda.set_device(local_rank)

        if (
            use_device_map
            and distributed
            and torch.cuda.is_available()
            and local_rank >= 0
            and not using_engine_parallelism
        ):
            load_kwargs.setdefault("device_map", {"": local_rank})
        if disable_device_map:
            load_kwargs.pop("device_map", None)

        return load_kwargs

    @classmethod
    def _place_model_for_distributed(cls, model):
        distributed, local_rank, _world_size = cls._distributed_context()
        using_engine_parallelism = (
            os.environ.get("ACCELERATE_USE_DEEPSPEED", "").lower() == "true"
            or os.environ.get("ACCELERATE_USE_FSDP", "").lower() == "true"
        )
        has_device_map = hasattr(model, "hf_device_map") or hasattr(getattr(model, "encoder", None), "hf_device_map")

        if distributed and torch.cuda.is_available() and local_rank >= 0 and not using_engine_parallelism and not has_device_map:
            model.to(torch.device("cuda", local_rank))

        return model

    @classmethod
    def _load_base_model(
        cls,
        model_args: ModelArguments,
        model_name_or_path: str,
        output_attentions: bool = True,
        **kwargs,
    ):
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        model_backbone = get_backbone_name(
            hf_config=config,
            model_type=model_args.model_backbone or model_args.model_type,
        )
        setattr(model_args, "model_backbone", model_backbone)

        config = cls._force_eager_attention(
            config,
            vision_output_attentions=output_attentions and model_backbone != FAST_VLM,
            output_attentions=output_attentions,
        )
        config.padding_side = "left"

        print_master(f"Loading generation backbone [{model_backbone}] from {model_name_or_path}")
        kwargs = cls._prepare_load_kwargs(kwargs)

        load_kwargs = {
            "config": config,
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            **kwargs,
        }
        
        if model_backbone == LLAVA_NEXT:
            return LlavaNextForConditionalGeneration.from_pretrained(model_name_or_path, **load_kwargs), model_backbone

        if model_backbone == LLAVA_ONEVISION:
            return LlavaOnevisionForConditionalGeneration.from_pretrained(model_name_or_path, **load_kwargs), model_backbone

        if model_backbone in {FAST_VLM, QWEN2_VL, QWEN2_5_VL, QWEN3_VL}:
            # print(f"Using custom loading for backbone {model_backbone} with config {config}")
            return backbone2model[model_backbone].from_pretrained(model_name_or_path, **load_kwargs), model_backbone

        return cls.TRANSFORMER_CLS.from_pretrained(
            model_name_or_path,
            attn_implementation="eager",
            trust_remote_code=True,
            **load_kwargs,
        ), model_backbone

    @staticmethod
    def _find_lora_targets(base_model, model_args):
        base_targets = [target.strip() for target in model_args.lora_target_modules.split(",") if target.strip()]
        if model_args.model_backbone not in {LLAVA_ONEVISION, LLAVA_NEXT}:
            return base_targets

        explicit_targets = []
        for name, _module in base_model.named_modules():
            if "language_model" in name and name.split(".")[-1] in base_targets:
                explicit_targets.append(name)
        if not explicit_targets:
            raise ValueError(f"No LoRA target modules found for {base_targets} inside language_model")
        return explicit_targets

    @classmethod
    def _wrap_lora_if_needed(cls, base_model, model_args: ModelArguments, model_name_or_path: str, is_trainable=True):
        if model_args.load_pretrained_lora or (model_args.lora and model_args.checkpoint_path):
            print_master(f"Loading LoRA adapter from {model_name_or_path}")
            lora_config = LoraConfig.from_pretrained(model_name_or_path)
            return PeftModel.from_pretrained(
                base_model,
                model_name_or_path,
                config=lora_config,
                is_trainable=is_trainable,
            )

        if model_args.lora:
            print_master("Initializing LoRA adapter")
            lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=cls._find_lora_targets(base_model, model_args),
                lora_dropout=model_args.lora_dropout,
                init_lora_weights="gaussian",
                use_dora=True,
                inference_mode=False,
            )
            return get_peft_model(base_model, lora_config)

        return base_model

    @classmethod
    def build(cls, model_args: ModelArguments, output_attentions: bool = True, **kwargs):
        base_model, model_backbone = cls._load_base_model(
            model_args,
            model_args.model_name,
            output_attentions=output_attentions,
            **kwargs,
        )
        encoder = cls._wrap_lora_if_needed(base_model, model_args, cls._model_path(model_args), is_trainable=True)
        model = cls(encoder=encoder, output_attentions=output_attentions)
        model.model_backbone = model_backbone
        return cls._place_model_for_distributed(model)

    @classmethod
    def load(cls, model_args: ModelArguments, is_trainable=True, output_attentions: bool = True, **kwargs):
        model_name_or_path = cls._model_path(model_args)

        is_adapter_load = bool(model_args.load_pretrained_lora or (model_args.lora and model_args.checkpoint_path))
        base_load_path = model_args.model_name if is_adapter_load else model_name_or_path
        base_model, model_backbone = cls._load_base_model(
            model_args,
            base_load_path,
            output_attentions=output_attentions,
            **kwargs,
        )
        encoder = cls._wrap_lora_if_needed(base_model, model_args, model_name_or_path, is_trainable=is_trainable)

        model = cls(encoder=encoder, output_attentions=output_attentions)
        model.model_backbone = model_backbone
        return cls._place_model_for_distributed(model)

    def save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.encoder.save_pretrained(output_dir)
