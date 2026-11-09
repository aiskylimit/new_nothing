"""Tiện ích nạp base model đóng băng + M LoRA expert (spec §2, §2b.2).

Toàn bộ repo dùng **một** base 7B đóng băng, mang nhiều adapter cùng lúc
(``PeftModel.load_adapter`` / ``set_adapter``) — đúng hạ tầng mà spec đề xuất
mượn từ PEFT thay vì fork X-LoRA.
"""

from __future__ import annotations

import contextlib
import os
from typing import Iterable, List, Optional, Sequence

import torch

DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

def expert_name(index: int) -> str:
    return f"expert_{index}"


def expert_names(n_experts: int) -> List[str]:
    return [expert_name(i) for i in range(n_experts)]


def _require_model(model_name):
    if not model_name:
        raise ValueError("model_name là bắt buộc (không còn model mặc định): truyền --model_name hoặc đặt trong YAML.")
    return model_name


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    _require_model(model_name)

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if not tok.is_fast:
        raise RuntimeError(
            "Cần fast tokenizer để lấy offset_mapping cho core/segmentation.py"
        )
    return tok


def load_base_model(
    model_name: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device_map: Optional[str] = None,
    gradient_checkpointing: bool = False,
    attn_implementation: Optional[str] = "sdpa",
):
    """Nạp causal LM ở bf16, đóng băng toàn bộ tham số gốc."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    for param in model.parameters():
        param.requires_grad_(False)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    return model


def make_lora_config(
    r: int = 16,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: Optional[Sequence[str]] = None,
):
    from peft import LoraConfig

    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules or DEFAULT_TARGET_MODULES),
        bias="none",
        task_type="CAUSAL_LM",
    )


def attach_experts(base_model, n_experts: int, lora_config, prefix: str = "expert"):
    """Gắn ``n_experts`` LoRA adapter mới lên base -> trả ``(PeftModel, names)``."""
    from peft import get_peft_model

    names = [f"{prefix}_{i}" for i in range(n_experts)]
    model = get_peft_model(base_model, lora_config, adapter_name=names[0])
    for name in names[1:]:
        model.add_adapter(name, lora_config)
    return model, names


def attach_single_lora(base_model, lora_config):
    """Gắn đúng một adapter tên ``default`` (Vanilla / Entropy baseline).

    Tên ``default`` để ``save_pretrained`` ghi thẳng file adapter vào ``out_dir``
    thay vì lồng thêm một cấp thư mục — eval nạp bằng ``PeftModel.from_pretrained
    (base, out_dir)``.
    """
    from peft import get_peft_model

    return get_peft_model(base_model, lora_config, adapter_name="default")


def load_experts(base_model, expert_dirs: Sequence[str], names: Optional[Sequence[str]] = None):
    """Nạp M adapter đã train từ đĩa lên base đóng băng."""
    from peft import PeftModel

    names = list(names or expert_names(len(expert_dirs)))
    if len(names) != len(expert_dirs):
        raise ValueError("số thư mục expert != số tên adapter")
    model = PeftModel.from_pretrained(base_model, expert_dirs[0], adapter_name=names[0])
    for path, name in zip(expert_dirs[1:], names[1:]):
        model.load_adapter(path, adapter_name=name)
    return model, names


def lora_parameters(model, adapter_names: Optional[Iterable[str]] = None):
    """Yield tham số LoRA (lọc theo adapter nếu truyền ``adapter_names``)."""
    wanted = set(adapter_names) if adapter_names is not None else None
    for name, param in model.named_parameters():
        if "lora_" not in name:
            continue
        if wanted is None or any(f".{a}." in name or name.endswith(f".{a}") for a in wanted):
            yield name, param


def set_lora_trainable(model, adapter_names: Sequence[str], trainable: bool = True) -> int:
    """Bật/tắt ``requires_grad`` cho LoRA của các adapter chỉ định.

    Cần thiết vì ``PeftModel.set_adapter`` chỉ để adapter đang active là
    trainable; Phase 1 train cả M expert nên phải bật lại sau mỗi lần switch.
    """
    count = 0
    for _, param in lora_parameters(model, adapter_names):
        param.requires_grad_(trainable)
        count += param.numel()
    return count


def activate_adapter(model, name: str, trainable_names: Optional[Sequence[str]] = None) -> None:
    """``set_adapter(name)`` + khôi phục requires_grad cho toàn bộ expert."""
    model.set_adapter(name)
    if trainable_names:
        set_lora_trainable(model, trainable_names, True)


def set_active_adapters(model, names: Sequence[str]) -> List[str]:
    """Bật ĐỒNG THỜI nhiều adapter — tương thích cả peft cũ lẫn >= 0.20.

    Từ ``peft 0.20``, ``PeftModel.set_adapter`` chỉ nhận **một** string (truyền
    list vào sẽ ném ``TypeError: unhashable type: 'list'``); việc bật nhiều
    adapter cùng lúc chuyển xuống ``LoraModel`` tức ``model.base_model``.
    ``LoraModel.set_adapter`` nhận list ở cả bản cũ lẫn bản mới, nên ta thử
    đường đó trước rồi mới fallback về ``PeftModel``.

    Forward của LoRA đọc ``_active_adapter`` do chính ``LoraModel.set_adapter``
    đặt trên từng layer, nên đường này đủ để cộng dồn delta của M adapter.
    """
    names = list(names)
    if len(names) == 1:
        model.set_adapter(names[0])
        return names
    inner = getattr(model, "base_model", None)
    if inner is not None and hasattr(inner, "set_adapter"):
        try:
            inner.set_adapter(names)
            return names
        except (TypeError, ValueError):
            pass
    model.set_adapter(names)  # peft cũ: PeftModel nhận thẳng list
    return names


@contextlib.contextmanager
def base_only(model):
    """Context manager: tắt toàn bộ adapter -> forward bằng base thuần."""
    with model.disable_adapter():
        yield model


def freeze_module(module) -> None:
    for param in module.parameters():
        param.requires_grad_(False)


def save_adapter(model, adapter_name: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    model.set_adapter(adapter_name)
    model.save_pretrained(out_dir, selected_adapters=[adapter_name])
    return os.path.join(out_dir, adapter_name)


def count_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
