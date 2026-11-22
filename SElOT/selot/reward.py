"""Response quality with a local reward model (replaces AlpacaEval's API judge).

RewardModel scores (prompt, response) with Skywork-Reward-V2-Llama-3.1-8B (Skywork/Skywork-Reward-V2-Llama-3.1-8B),
as its model card does: AutoModelForSequenceClassification (num_labels=1), the conversation [user, assistant] through
the tokenizer's chat template with no system turn, a duplicate leading BOS removed, score = the single logit.
Batches are right-padded with the model's own pad token (<|finetune_right_pad_id|>, never in a conversation), so the
classifier reads the last real token, as in an unbatched call.

win_rates compares two responses per prompt by reward (tie = half a win), as AlpacaEval does with a judge;
the length-controlled rate follows AlpacaEval 2's idea: a logistic fit of win ~ tanh(length difference / its std),
read at length difference 0.
"""

import math
from typing import List, Optional, Sequence

import torch


class RewardModel:
    name = "skywork-reward-v2"

    def __init__(self, path: str, dtype: str = "bfloat16", device: str = "cuda", batch_size: int = 8,
                 max_length: int = 4096):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from selot.activations import DTYPES
        self.tok = AutoTokenizer.from_pretrained(path)
        self.tok.padding_side = "right"
        if self.tok.pad_token is None:
            self.tok.pad_token = "<|finetune_right_pad_id|>"
        kw = {"dtype": DTYPES[dtype]} if int(__import__("transformers").__version__.split(".")[0]) >= 5 else \
            {"torch_dtype": DTYPES[dtype]}
        self.model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=1, **kw).to(device).eval()
        self.model.config.pad_token_id = self.tok.pad_token_id
        self.batch_size, self.max_length = batch_size, max_length

    def text(self, prompt: str, response: str) -> str:
        t = self.tok.apply_chat_template([{"role": "user", "content": prompt},
                                          {"role": "assistant", "content": response}], tokenize=False)
        bos = self.tok.bos_token
        return t[len(bos):] if bos and t.startswith(bos) else t

    @torch.no_grad()
    def __call__(self, prompts: Sequence[str], responses: Sequence[str]) -> List[float]:
        texts = [self.text(p, r) for p, r in zip(prompts, responses)]
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        out = [0.0] * len(texts)
        for i in range(0, len(order), self.batch_size):
            idx = order[i:i + self.batch_size]
            enc = self.tok([texts[j] for j in idx], return_tensors="pt", padding=True, truncation=True,
                           max_length=self.max_length).to(self.model.device)
            scores = self.model(**enc).logits[:, 0].float().cpu().tolist()
            for j, s in zip(idx, scores):
                out[j] = s
        return out


def win_rates(r_a: Sequence[float], r_b: Sequence[float], len_a: Sequence[int], len_b: Sequence[int]):
    """(win rate, length-controlled win rate) of responses a over b, paired per prompt; tie = 0.5"""
    wins = [1.0 if x > y else 0.5 if x == y else 0.0 for x, y in zip(r_a, r_b)]
    if not wins:
        return None, None
    return sum(wins) / len(wins), _lc(wins, [a - b for a, b in zip(len_a, len_b)])


def _lc(wins: List[float], dlen: List[int]) -> Optional[float]:
    sd = math.sqrt(sum(d * d for d in dlen) / len(dlen) - (sum(dlen) / len(dlen)) ** 2)
    if sd == 0 or len(set(wins)) < 2 and 0.5 not in wins:
        return sum(wins) / len(wins)       # no length effect to remove, or a single class
    from sklearn.linear_model import LogisticRegression
    x = [[math.tanh(d / sd)] for d in dlen]
    # a tie is half a win and half a loss
    X = x + x
    y = [1] * len(wins) + [0] * len(wins)
    w = wins + [1 - v for v in wins]
    fit = LogisticRegression(C=1e6, max_iter=1000).fit(X, y, sample_weight=w)
    return float(1 / (1 + math.exp(-fit.intercept_[0])))
