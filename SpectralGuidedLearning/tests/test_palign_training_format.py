"""P-ALIGN trains with LLaMA-Factory; the `palign` family reproduces its token ids and labels.

Checked against LLaMA-Factory's own template code when it is importable (pip install llamafactory,
or LLAMAFACTORY_SRC=<checkout>/src); skipped otherwise. Three templates, as P-ALIGN's configs use
them: deepseekr1 (R1-Distill, enable_thinking false), qwen3 (enable_thinking false), qwen (Qwen2.5).
"""
import copy
import json
import os
import sys
import types
from pathlib import Path

import pytest

from sgl.data.prepare import build_record

ASSETS = Path(__file__).resolve().parent / "regression" / "assets"
if os.environ.get("LLAMAFACTORY_SRC"):
    sys.path.insert(0, os.environ["LLAMAFACTORY_SRC"])
lf_template = pytest.importorskip("llamafactory.data.template")

R1_TEMPLATE = (
    "{{ bos_token }}{% for m in messages %}{% if m['role'] == 'user' %}<｜User｜>{{ m['content'] }}"
    "{% endif %}{% endfor %}{% if add_generation_prompt %}<｜Assistant｜><think>\n{% endif %}"
)
QWEN25_TEMPLATE = (
    "{% if messages[0]['role'] != 'system' %}<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. "
    "You are a helpful assistant.<|im_end|>\n{% endif %}{% for m in messages %}<|im_start|>{{ m['role'] }}\n"
    "{{ m['content'] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
QWEN3_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% if enable_thinking is defined and "
    "enable_thinking is false %}<think>\n\n</think>\n\n{% endif %}{% endif %}"
)
CASES = {  # LLaMA-Factory template -> (HF chat template of the real model, eos, stop suffix the config sets)
    "deepseekr1": (R1_TEMPLATE, "<｜end▁of▁sentence｜>", ""),
    "qwen3": (QWEN3_TEMPLATE, "<|im_end|>", "\n"),
    "qwen": (QWEN25_TEMPLATE, "<|im_end|>", "\n"),
}


def tokenizer_for(chat_template: str, eos: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(ASSETS / "tokenizer"))
    tokenizer.add_special_tokens({
        "bos_token": "<｜begin▁of▁sentence｜>", "eos_token": eos,
        "additional_special_tokens": ["<｜User｜>", "<｜Assistant｜>", "<｜end▁of▁sentence｜>", "<think>", "</think>"],
    })
    tokenizer.chat_template = chat_template
    return tokenizer


@pytest.mark.parametrize("name", sorted(CASES))
def test_prepare_matches_llamafactory_encoding(name):
    chat_template, eos, stop_suffix = CASES[name]
    ours_tokenizer = tokenizer_for(chat_template, eos)
    lf_tokenizer = copy.deepcopy(ours_tokenizer)
    data_args = types.SimpleNamespace(template=name, train_on_prompt=False, tool_format=None, default_system=None,
                                      enable_thinking=False, preserve_thinking=False)
    template = lf_template.get_template_and_fix_tokenizer(lf_tokenizer, data_args)
    rows = [json.loads(line) for line in (ASSETS / "palign_sample.jsonl").open()]
    for row in rows:
        prompt_ids, response_ids = template.encode_oneturn(lf_tokenizer, [
            {"role": "user", "content": row["instruction"] + "\n" + row["input"]},  # AlpacaDatasetConverter
            {"role": "assistant", "content": row["output"]},
        ])
        record = build_record(ours_tokenizer, row["input"], row["output"].strip(), 10**6, chat_template=True,
                              enable_thinking=False, palign_prompt=True, instruction_separator="\n",
                              stop_suffix=stop_suffix)
        assert record["input_ids"] == prompt_ids + response_ids
        # what build_masks --vanilla-only supervises: the response span and everything after it
        assert record["response_token_span"][0] == len(prompt_ids)
