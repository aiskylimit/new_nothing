"""Declared reference recipe; ordinary OPD is independent of DSKDv2 losses."""
from pathlib import Path
import hashlib

TEACHER = "meta-llama/Llama-2-7b-hf"
STUDENT = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
UPSTREAM_COMMIT = "ae4993e8b857ca001742de651498dc627138dd32"
RECIPES = {
    "dskd-tiny": dict(epochs=10, grad_accum=32, lr=1e-3, projector_lr=1e-3,
        lora_rank=256, lora_alpha=8, lora_dropout=.1, lora_targets="q_proj,v_proj",
        max_length=512, max_prompt_length=256, max_new_tokens=256,
        bi_max_length=256, kd_temperature=2., kd_rate=.5, kd_objective="forward_kl",
        scheduler="cosine", temperature_squared=False, truncate_prompts=True),
    "legacy": dict(epochs=0, grad_accum=8, lr=1e-5, projector_lr=5e-5,
        lora_rank=32, lora_alpha=64, lora_dropout=0., lora_targets="all-linear",
        max_length=2048, max_prompt_length=512, max_new_tokens=512,
        bi_max_length=1024, kd_temperature=1., kd_rate=1., kd_objective="reverse_kl",
        scheduler="constant", temperature_squared=True, truncate_prompts=False),
}


def apply_recipe(args):
    for key, value in RECIPES[args.preset].items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    return args


def adapter_signature(path):
    if not path:
        return None
    files = sorted(Path(path).glob("adapter*"))
    if not any(p.suffix == ".safetensors" for p in files):
        raise FileNotFoundError(f"Missing safetensors teacher adapter: {path}")
    h = hashlib.sha256()
    for p in files:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def load_teacher_adapter(model, path):
    if path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, path).merge_and_unload()
    return model.eval().requires_grad_(False)
