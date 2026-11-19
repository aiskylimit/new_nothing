"""Benchmark metrics that do not need a judge model: GSM8K exact match, MMLU letter scoring, set -> metric mapping.

Every eval set in the config has a `metric`:
    asr          harmful prompts (attacks, AdvBench): ASR by the HarmBench classifier (or keyword), StrongREJECT score,
                 refusal rate
    refusal      over-refusal sets (XSTest safe, OR-Bench-Hard, OKTest): refusal and compliance rate;
                 with `kind: harmful` (XSTest unsafe) the refusal rate is the number to keep high
    gsm8k        generation with GSM8K_TEMPLATE, exact match of the final number
    mmlu         no generation: the next-token logits of A/B/C/D after the question, argmax vs the key
    alpaca_eval  generation; quality by a local reward model (selot/reward.py, judge stage): win rate vs the
                 unsteered (alpha 0) response and vs the text-davinci-003 reference
"""

import math
import re
from typing import Dict, List, Optional

import torch

METRICS = ("asr", "refusal", "gsm8k", "mmlu", "alpaca_eval")

# AlphaSteer's GSM8K / MATH prompt (src/generate_response.py), so both methods see the same input.
GSM8K_TEMPLATE = ("Please solve this problem, and put your final answer within \\boxed{{}}\n"
                  "This is the problem:\n{prompt}\n"
                  "Please remember to put your final answer within \\boxed{{}}\n")
TEMPLATES = {"gsm8k": GSM8K_TEMPLATE}

MMLU_TEMPLATE = ("The following is a multiple choice question about {subject}. Answer with the letter of the correct "
                 "option only.\n\n{question}\n{options}\nAnswer:")
LETTERS = "ABCD"


def set_metric(spec: dict) -> str:
    m = spec.get("metric") or ("asr" if spec.get("kind", "harmful") == "harmful" else "refusal")
    if m not in METRICS:
        raise ValueError(f"eval set {spec.get('name')}: unknown metric {m!r} (one of {METRICS})")
    return m


def set_kind(spec: dict) -> str:
    return spec.get("kind") or ("harmful" if set_metric(spec) == "asr" else "benign")


def apply_template(spec: dict, prompt: str) -> str:
    t = spec.get("template")
    if not t:
        return prompt
    return TEMPLATES.get(t, t).format(prompt=prompt)


# ---------------------------------------------------------------- GSM8K
def _boxed(text: str) -> Optional[str]:
    """content of the last \\boxed{...}, braces balanced"""
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth, k = 0, j
    while k < len(text):
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        if depth == 0:
            return text[j + 1:k]
        k += 1
    return None


_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+")


def _to_number(s: str) -> Optional[float]:
    s = s.replace("\\!", "").replace("$", "").replace("\\%", "").replace("%", "")
    s = re.sub(r"\\(text|mathrm)\{[^}]*\}", "", s)
    frac = re.search(r"\\d?frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}", s)
    if frac:
        try:
            return float(frac.group(1)) / float(frac.group(2))
        except ZeroDivisionError:
            return None
    nums = _NUM.findall(s)
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", ""))
    except ValueError:
        return None


def gsm8k_extract(response: str) -> Optional[float]:
    """last \\boxed{} answer, else the last number in the response"""
    b = _boxed(response)
    if b is not None:
        v = _to_number(b)
        if v is not None:
            return v
    return _to_number(response)


def gsm8k_correct(response: str, answer: str) -> bool:
    pred = gsm8k_extract(response)
    gold = _to_number(str(answer))
    return pred is not None and gold is not None and math.isclose(pred, gold, rel_tol=0, abs_tol=1e-6)


# ---------------------------------------------------------------- MMLU
def mmlu_prompt(rec: dict) -> str:
    options = "\n".join(f"{l}. {c}" for l, c in zip(LETTERS, rec["choices"]))
    subject = rec.get("subject", "").replace("_", " ")
    return MMLU_TEMPLATE.format(subject=subject, question=rec["prompt"], options=options)


def letter_token_ids(tok) -> Dict[str, List[int]]:
    """token ids that spell each letter as the first answer token ("A" and " A" where they are single tokens)"""
    out = {}
    for l in LETTERS:
        ids = set()
        for s in (l, " " + l):
            enc = tok.encode(s, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(enc[0])
        if not ids:
            raise ValueError(f"tokenizer has no single token for {l!r}")
        out[l] = sorted(ids)
    return out


@torch.no_grad()
def mmlu_letter_scores(logits: torch.Tensor, ids: Dict[str, List[int]]) -> torch.Tensor:
    """logits (batch x vocab) at the first answer position -> (batch x 4) log-probs over A-D (max over spellings)"""
    lp = logits.float().log_softmax(-1)
    cols = [lp[:, v].max(-1).values for v in ids.values()]
    return torch.stack(cols, -1).log_softmax(-1)
