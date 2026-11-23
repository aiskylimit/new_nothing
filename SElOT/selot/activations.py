"""Model loading, chat formatting and activation extraction (residual stream at the input of layer l).

Generation (scripts/generate.py) uses the same `format_prompts` / `tokenize` so fitted and steered
activations come from identical token sequences.
"""

import logging
from typing import Dict, List

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from selot.steering import capture_layer_inputs, decoder_layers

logger = logging.getLogger(__name__)
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def load_model(path: str, dtype: str = "bfloat16", device: str = "cuda"):
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, dtype=DTYPES[dtype]) if _new_tf() else \
        AutoModelForCausalLM.from_pretrained(path, torch_dtype=DTYPES[dtype])
    model.to(device).eval()
    model.generation_config.pad_token_id = tok.pad_token_id
    return model, tok


def _new_tf() -> bool:
    import transformers
    return int(transformers.__version__.split(".")[0]) >= 5


def num_layers(model) -> int:
    return len(decoder_layers(model))


def resolve_layers(model_cfg: dict, n_layers: int) -> List[int]:
    """Explicit `layers`, else every layer in `layer_range` (fractions of depth, inclusive)."""
    if model_cfg.get("layers"):
        return sorted(int(l) for l in model_cfg["layers"])
    lo, hi = model_cfg.get("layer_range", [0.3, 0.7])
    import math
    return list(range(math.ceil(lo * n_layers), math.floor(hi * n_layers) + 1))


def format_prompts(tok, prompts: List[str]) -> List[str]:
    """One user turn, chat template, generation prompt appended (no system prompt)."""
    return [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            for p in prompts]


def tokenize(tok, texts: List[str], device):
    # The chat template already contains BOS where the model wants one. No truncation: a cut prompt loses the end
    # of the chat template, i.e. the last token the activations are read at and the model answers from.
    return tok(texts, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False).to(device)


@torch.no_grad()
def extract_last_token(model, tok, prompts: List[str], layers: List[int],
                       batch_size: int = 16) -> Dict[int, torch.Tensor]:
    """{layer: n x d float32}: input of decoder layer `layer` at the last prompt token."""
    device = next(model.parameters()).device
    texts = format_prompts(tok, prompts)
    out = {l: [] for l in layers}
    for i in tqdm(range(0, len(texts), batch_size), desc="extract", leave=False):
        batch = tokenize(tok, texts[i:i + batch_size], device)
        with capture_layer_inputs(model, layers) as store:
            model(**batch, use_cache=False, logits_to_keep=1)   # logits are not used: skip the full-vocab
            # lm_head over every position (Qwen: 152k vocab x batch x length, ~14 GB at batch 32)
        for l in layers:
            out[l].append(store[l][:, -1, :].float().cpu())          # left padding: last position is real
    return {l: torch.cat(v) for l, v in out.items()}


@torch.no_grad()
def extract_all_tokens(model, tok, prompts: List[str], layers: List[int], batch_size: int = 8):
    """For RQ3 at every prompt position: ({layer: N_tokens x d}, is_last: N_tokens bool)."""
    device = next(model.parameters()).device
    texts = format_prompts(tok, prompts)
    out, is_last = {l: [] for l in layers}, []
    for i in tqdm(range(0, len(texts), batch_size), desc="extract-all", leave=False):
        batch = tokenize(tok, texts[i:i + batch_size], device)
        mask = batch["attention_mask"].bool()
        with capture_layer_inputs(model, layers) as store:
            model(**batch, use_cache=False, logits_to_keep=1)   # logits are not used: skip the full-vocab
            # lm_head over every position (Qwen: 152k vocab x batch x length, ~14 GB at batch 32)
        last = torch.zeros_like(mask)
        last[:, -1] = True
        is_last.append(last[mask].cpu())
        for l in layers:
            out[l].append(store[l][mask].float().cpu())
    return {l: torch.cat(v) for l, v in out.items()}, torch.cat(is_last)
