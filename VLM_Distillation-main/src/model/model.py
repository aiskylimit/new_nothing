import os

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

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
    normalize_fast_vlm_config,
    normalize_fast_vlm_model,
    normalize_fast_vlm_tokenizer,
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
        self._scva_module_cache = None
        self._scva_capture_active = False

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
    def _configure_attention_backend(
        cls,
        config,
        vision_output_attentions=True,
        output_attentions=True,
        force_eager=False,
    ):
        sub_configs = {
            name: getattr(config, name, None)
            for name in ("text_config", "vision_config", "vision_config_2", "audio_config")
        }
        config.use_cache = False
        if force_eager or output_attentions:
            # Transformers generally needs eager attention when callers request
            # the full attention matrices. FastVLM also stays on eager because
            # its current implementation does not support the SDPA backend.
            config._attn_implementation = "eager"
            config.attn_implementation = "eager"
        else:
            # Attention-free criteria use SDPA for every non-FastVLM backbone.
            config._attn_implementation = "sdpa"
            config.attn_implementation = "sdpa"
        config.output_attentions = output_attentions
        config.output_hidden_states = True

        for sub_config_name, sub_config in sub_configs.items():
            if sub_config is None:
                continue
            sub_output_attentions = (
                vision_output_attentions if sub_config_name.startswith("vision_config") else output_attentions
            )
            if force_eager or sub_output_attentions:
                sub_config._attn_implementation = "eager"
                sub_config.attn_implementation = "eager"
            else:
                sub_config._attn_implementation = "sdpa"
                sub_config.attn_implementation = "sdpa"
            cls._set_config_attr_if_present(
                sub_config,
                "output_attentions",
                sub_output_attentions,
            )
            sub_hidden_states = not (
                sub_config_name.startswith("vision_config") and not vision_output_attentions
            )
            cls._set_config_attr_if_present(sub_config, "output_hidden_states", sub_hidden_states)
            cls._set_config_attr_if_present(sub_config, "use_cache", False)

        return config

    @staticmethod
    def _model_path(model_args: ModelArguments):
        return model_args.checkpoint_path if model_args.checkpoint_path else model_args.model_name

    @staticmethod
    def _verify_attention_backend(model, expected_backend: str, model_backbone: str):
        config = model.config
        configs = {"config": config}
        configs.update(
            {
                name: sub_config
                for name in ("text_config", "vision_config", "vision_config_2", "audio_config")
                if (sub_config := getattr(config, name, None)) is not None
            }
        )
        mismatches = {
            name: getattr(candidate, "_attn_implementation", None)
            for name, candidate in configs.items()
            if getattr(candidate, "_attn_implementation", None) != expected_backend
        }
        if mismatches:
            raise RuntimeError(
                f"Attention backend verification failed for {model_backbone}: "
                f"expected {expected_backend}, got {mismatches}."
            )
        print_master(
            f"Verified attention backend [{expected_backend}] for [{model_backbone}] "
            f"across {', '.join(configs)}."
        )

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

        # Baseline criteria do not use SCVA. Skip all capture bookkeeping and
        # module-tree walks on their hot forward path.
        if not self._scva_capture_active:
            return self.encoder(**forward_kwargs)

        self._prepare_scva_capture(model_inputs)
        try:
            outputs = self.encoder(**forward_kwargs)
            scva_attentions = self._collect_scva_captures()
        finally:
            self._clear_scva_capture_context()

        if scva_attentions:
            setattr(outputs, "scva_attentions", scva_attentions)
        return outputs

    def _scva_attention_modules(self):
        if self._scva_module_cache is None:
            self._scva_module_cache = tuple(
                module for module in self.encoder.modules()
                if getattr(module, "supports_scva_sparse_capture", False)
                and getattr(module, "layer_idx", None) is not None
            )
        return self._scva_module_cache

    def attention_layer_indices(self) -> list[int]:
        return sorted({int(module.layer_idx) for module in self._scva_attention_modules()})

    def configure_scva_attention(self, layer_indices) -> None:
        selected = {int(index) for index in layer_indices}
        available = set(self.attention_layer_indices())
        missing = selected - available
        if missing:
            raise ValueError(f"SCVA requested unavailable decoder layers: {sorted(missing)}")
        for module in self._scva_attention_modules():
            module._scva_capture_enabled = int(module.layer_idx) in selected
        self._scva_capture_active = bool(selected)

    def _prepare_scva_capture(self, model_inputs) -> None:
        selected = [
            module for module in self._scva_attention_modules()
            if getattr(module, "_scva_capture_enabled", False)
        ]
        if not selected:
            return

        input_ids = model_inputs.get("input_ids")
        labels = model_inputs.get("labels")
        if not torch.is_tensor(input_ids) or not torch.is_tensor(labels):
            return

        vision_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_attr in ("image_token_id", "video_token_id"):
            token_id = getattr(self.config, token_attr, None)
            if token_id is not None:
                vision_mask |= input_ids.eq(int(token_id))
        response_mask = labels.ne(-100)
        attention_mask = model_inputs.get("attention_mask")
        if torch.is_tensor(attention_mask) and attention_mask.ndim == 2:
            valid = attention_mask.to(dtype=torch.bool, device=response_mask.device)
            response_mask &= valid
            vision_mask &= valid.to(vision_mask.device)

        for module in selected:
            module._scva_response_mask = response_mask
            module._scva_vision_mask = vision_mask
            module._scva_attention = None

    def _collect_scva_captures(self):
        captures = {}
        for module in self._scva_attention_modules():
            if not getattr(module, "_scva_capture_enabled", False):
                continue
            attention = getattr(module, "_scva_attention", None)
            if attention is not None:
                captures[int(module.layer_idx)] = attention
        return captures

    def _clear_scva_capture_context(self) -> None:
        for module in self._scva_attention_modules():
            module._scva_response_mask = None
            module._scva_vision_mask = None
            module._scva_attention = None

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
        vision_output_attentions: bool | None = None,
        force_sdpa: bool = False,
        **kwargs,
    ):
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        model_backbone = get_backbone_name(
            hf_config=config,
            model_type=model_args.model_backbone or model_args.model_type,
        )
        setattr(model_args, "model_backbone", model_backbone)

        fast_vlm_tokenizer = None
        if model_backbone == FAST_VLM:
            fast_vlm_tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
            )
            normalize_fast_vlm_tokenizer(fast_vlm_tokenizer)
            normalize_fast_vlm_config(
                config,
                fast_vlm_tokenizer.eos_token_id,
                fast_vlm_tokenizer.pad_token_id,
            )

        if vision_output_attentions is None:
            vision_output_attentions = output_attentions and model_backbone != FAST_VLM
        config = cls._configure_attention_backend(
            config,
            vision_output_attentions=vision_output_attentions,
            output_attentions=output_attentions,
            force_eager=model_backbone == FAST_VLM,
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
        
        expected_attention_backend = "eager" if model_backbone == FAST_VLM or output_attentions else "sdpa"
        if model_backbone == LLAVA_NEXT:
            model = LlavaNextForConditionalGeneration.from_pretrained(model_name_or_path, **load_kwargs)
        elif model_backbone == LLAVA_ONEVISION:
            model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_name_or_path, **load_kwargs)
        elif model_backbone in {FAST_VLM, QWEN2_VL, QWEN2_5_VL, QWEN3_VL}:
            # print(f"Using custom loading for backbone {model_backbone} with config {config}")
            model = backbone2model[model_backbone].from_pretrained(model_name_or_path, **load_kwargs)
            if model_backbone == FAST_VLM:
                normalize_fast_vlm_model(model, fast_vlm_tokenizer)
        else:
            model = cls.TRANSFORMER_CLS.from_pretrained(
                model_name_or_path,
                attn_implementation=expected_attention_backend,
                trust_remote_code=True,
                **load_kwargs,
            )

        cls._verify_attention_backend(model, expected_attention_backend, model_backbone)
        return model, model_backbone

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
    def build(
        cls,
        model_args: ModelArguments,
        output_attentions: bool = True,
        vision_output_attentions: bool | None = None,
        force_sdpa: bool = False,
        **kwargs,
    ):
        base_model, model_backbone = cls._load_base_model(
            model_args,
            model_args.model_name,
            output_attentions=output_attentions,
            vision_output_attentions=vision_output_attentions,
            force_sdpa=force_sdpa,
            **kwargs,
        )
        encoder = cls._wrap_lora_if_needed(base_model, model_args, cls._model_path(model_args), is_trainable=True)
        model = cls(encoder=encoder, output_attentions=output_attentions)
        model.model_backbone = model_backbone
        return cls._place_model_for_distributed(model)

    @classmethod
    def load(
        cls,
        model_args: ModelArguments,
        is_trainable=True,
        output_attentions: bool = True,
        vision_output_attentions: bool | None = None,
        force_sdpa: bool = False,
        **kwargs,
    ):
        model_name_or_path = cls._model_path(model_args)

        is_adapter_load = bool(model_args.load_pretrained_lora or (model_args.lora and model_args.checkpoint_path))
        base_load_path = model_args.model_name if is_adapter_load else model_name_or_path
        base_model, model_backbone = cls._load_base_model(
            model_args,
            base_load_path,
            output_attentions=output_attentions,
            vision_output_attentions=vision_output_attentions,
            force_sdpa=force_sdpa,
            **kwargs,
        )
        encoder = cls._wrap_lora_if_needed(base_model, model_args, model_name_or_path, is_trainable=is_trainable)

        model = cls(encoder=encoder, output_attentions=output_attentions)
        model.model_backbone = model_backbone
        return cls._place_model_for_distributed(model)

    def save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.encoder.save_pretrained(output_dir)
