"""Metric đa dạng cho bảng ablation loss Phase 1 (teacher-forcing trên long-CoT proxy).

Với mỗi mẫu proxy (question + long-CoT DSR1), ép cả M expert đọc CÙNG chuỗi CoT và đo, trên từng token
của phần thinking, Jensen-Shannon giữa phân phối next-token (full vocab, fp32) của từng cặp expert:
  js_key    = JS trung bình trên token thuộc KEY step (top-frac entropy dưới base, đã tính bằng keystep_select)
  js_nonkey = JS trung bình trên token thuộc step còn lại
Đồng thời ghi NLL teacher-forced của từng expert (key / non-key) làm proxy chất lượng rẻ tiền.

Chỉ tính trên các step thinking (không tính segment đáp án sau </think>). Model ở eval mode (tắt lora_dropout).
Số liệu: (i) token-mean (tổng JS / số token, có trọng số theo độ dài step), (ii) step-mean (trung bình JS mỗi step),
kèm khoảng tin cậy bootstrap theo mẫu.

    python -m proposal_v2.ablation_metrics --model_name models/Qwen2.5-7B-Instruct \
        --phase1_dir outputs/proposal_v2/phase1_svgd_func_keystep_k1_noDropout \
        --keystep_dir outputs/proposal_v2/keystep_proxy500_10pct --out results/ablation_metrics/7b_noDropout.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time

import numpy as np
import torch

from core.data_utils import StepDataset
from core.losses import token_logprobs
from core.proctitle import set_proc_title
from core.model_utils import load_base_model, load_experts, load_tokenizer
from train.sft_engine import build_segments


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="js-key / js-nonkey / NLL cho các expert Phase 1")
    p.add_argument("--model_name", required=True)
    p.add_argument("--phase1_dir", required=True, help="thư mục chứa expert_0..expert_{M-1}")
    p.add_argument("--proxy_file", default="data/ablation_proxy/proxy500.jsonl")
    p.add_argument("--keystep_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--M", type=int, default=3)
    p.add_argument("--cutoff_len", type=int, default=12288)
    p.add_argument("--segmentation_mode", default="paragraph")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--chunk", type=int, default=512, help="số vị trí mỗi chunk khi tính JS full-vocab")
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--mc_dropout", type=int, default=0,
                   help="chỉ dùng khi M=1: bật lora_dropout, chạy N lượt forward ngẫu nhiên của CÙNG một expert "
                        "và tính JS giữa các lượt (nền nhiễu dropout, KHÔNG so sánh trực tiếp với JS giữa expert độc lập)")
    p.add_argument("--sanity", action="store_true", help="đối chiếu NLL với core.losses.token_logprobs ở mẫu đầu")
    return p.parse_args()


def _js_pair(lp_i: torch.Tensor, lp_j: torch.Tensor) -> torch.Tensor:
    """JS(P_i, P_j) theo từng vị trí, log-prob fp32 ``(c, V)`` -> ``(c,)`` (nats)."""
    log_m = torch.logaddexp(lp_i, lp_j) - math.log(2.0)
    kl_i = (lp_i.exp() * (lp_i - log_m)).sum(-1)
    kl_j = (lp_j.exp() * (lp_j - log_m)).sum(-1)
    return 0.5 * (kl_i + kl_j)


@torch.no_grad()
def per_token_stats(model, names, input_ids, lo: int, hi: int, chunk: int):
    """Trả (js ``(P, L)``, nll ``(M, L)``) cho các token ở vị trí [lo, hi) — L = hi-lo."""
    logits = []
    for nm in names:
        model.set_adapter(nm)
        logits.append(model(input_ids=input_ids, use_cache=False).logits[0])  # (T, V) bf16
    pairs = list(itertools.combinations(range(len(names)), 2))
    L = hi - lo
    js = torch.empty(len(pairs), L, device=input_ids.device)
    nll = torch.empty(len(names), L, device=input_ids.device)
    for a in range(0, L, chunk):
        b = min(a + chunk, L)
        pos = torch.arange(lo + a, lo + b, device=input_ids.device)
        tgt = input_ids[0, pos]
        lps = [torch.log_softmax(lg[pos - 1].float(), dim=-1) for lg in logits]  # token p <- logits[p-1]
        for k, lp in enumerate(lps):
            nll[k, a:b] = -lp.gather(-1, tgt[:, None])[:, 0]
        for pi, (i, j) in enumerate(pairs):
            js[pi, a:b] = _js_pair(lps[i], lps[j])
        del lps
    del logits
    return js, nll, pairs


def main() -> None:
    args = parse_args()
    set_proc_title(os.path.splitext(os.path.basename(args.out))[0])
    device = torch.device(args.device)
    tok = load_tokenizer(args.model_name)
    base = load_base_model(args.model_name).to(device)
    dirs = [os.path.join(args.phase1_dir, f"expert_{i}") for i in range(args.M)]
    model, names = load_experts(base, dirs)
    model.to(device).eval()
    if args.mc_dropout:
        if args.M != 1:
            raise ValueError("--mc_dropout chỉ dùng với --M 1")
        names = list(names) * args.mc_dropout  # cùng một adapter, N lượt forward với mask dropout khác nhau
        n_on = 0
        for n_, m_ in model.named_modules():
            if "lora_dropout" in n_ and isinstance(m_, torch.nn.Dropout):
                m_.train(); n_on += 1
        print(f"[metrics] mc_dropout: {args.mc_dropout} lượt, bật {n_on} module lora_dropout", flush=True)
    M_eff = len(names)

    ds = StepDataset(args.proxy_file, tok, cutoff_len=args.cutoff_len,
                     segmentation_mode=args.segmentation_mode, limit=args.max_samples)
    P = M_eff * (M_eff - 1) // 2
    rec = []
    t0 = time.time()
    for idx in range(len(ds)):
        ex = ds[idx]
        z = np.load(os.path.join(args.keystep_dir, f"{ex.example_id}.npz"))
        key_all = z["key_mask"].astype(bool)
        n_all = len(build_segments(ex))
        if len(key_all) != n_all:
            raise RuntimeError(f"{ex.example_id}: key_mask {len(key_all)} != {n_all} segment (lệch cutoff/tokenizer?)")
        spans = list(ex.step_spans)
        key = key_all[: len(spans)]
        lo, hi = spans[0][0], spans[-1][1]
        ids = ex.input_ids.unsqueeze(0).to(device)
        js, nll, pairs = per_token_stats(model, names, ids, lo, hi, args.chunk)

        if args.sanity and idx == 0:
            model.set_adapter(names[0])
            with torch.no_grad():
                ref = token_logprobs(model(input_ids=ids, use_cache=False).logits, ids)[lo:hi]
            print(f"[sanity] |nll - ref| max = {(nll[0] + ref).abs().max().item():.4f}", flush=True)

        js_c, nll_c = js.cpu().numpy(), nll.cpu().numpy()
        r = {"id": ex.example_id, "n_steps": len(spans), "n_key_steps": int(key.sum()),
             "tok": [0, 0], "js": np.zeros((P, 2)), "nll": np.zeros((M_eff, 2)),
             "step_js_sum": np.zeros((P, 2)), "step_n": [0, 0]}
        for (s, e), k in zip(spans, key):
            g = 0 if k else 1
            a, b = s - lo, e - lo
            if b <= a:
                continue
            r["tok"][g] += b - a
            r["js"][:, g] += js_c[:, a:b].sum(1)
            r["nll"][:, g] += nll_c[:, a:b].sum(1)
            r["step_js_sum"][:, g] += js_c[:, a:b].mean(1)
            r["step_n"][g] += 1
        rec.append(r)
        if (idx + 1) % 25 == 0 or idx == 0:
            print(f"[metrics] {idx + 1}/{len(ds)} | {time.time() - t0:.0f}s", flush=True)

    def agg(sel, kind):
        """kind: 'tok' (token-mean) hoặc 'step' (step-mean); trả js theo (pair, group)."""
        num = np.sum([r["js" if kind == "tok" else "step_js_sum"][:, :] for r in sel], axis=0)  # (P,2)
        den = np.sum([r["tok" if kind == "tok" else "step_n"] for r in sel], axis=0)  # (2,)
        return num / np.maximum(den, 1)

    def summarize(sel):
        out = {}
        for kind in ("tok", "step"):
            m = agg(sel, kind).mean(0)  # trung bình 3 cặp -> (2,)
            out[f"js_key_{kind}"], out[f"js_nonkey_{kind}"] = float(m[0]), float(m[1])
            out[f"ratio_{kind}"] = float(m[0] / max(m[1], 1e-12))
        den = np.sum([r["tok"] for r in sel], axis=0)
        nl = np.sum([r["nll"] for r in sel], axis=0) / np.maximum(den, 1)  # (M,2)
        out["nll_key_mean_over_experts"], out["nll_nonkey_mean_over_experts"] = float(nl[:, 0].mean()), float(nl[:, 1].mean())
        return out

    result = summarize(rec)
    rng = np.random.default_rng(0)
    boots = [summarize([rec[i] for i in rng.integers(0, len(rec), len(rec))]) for _ in range(args.n_boot)]
    ci = {k: [float(np.percentile([b[k] for b in boots], 2.5)), float(np.percentile([b[k] for b in boots], 97.5))]
          for k in result}
    pair_tok = agg(rec, "tok")
    out = {"model_name": args.model_name, "phase1_dir": args.phase1_dir, "n_samples": len(rec),
           "key_steps_frac_of_steps": float(np.sum([r["n_key_steps"] for r in rec]) / np.sum([r["n_steps"] for r in rec])),
           "tokens": {"key": int(np.sum([r["tok"][0] for r in rec])), "nonkey": int(np.sum([r["tok"][1] for r in rec]))},
           "metrics": result, "ci95_bootstrap_over_samples": ci,
           "js_by_pair_tok": {f"e{i}-e{j}": {"key": float(pair_tok[p, 0]), "nonkey": float(pair_tok[p, 1])}
                              for p, (i, j) in enumerate(pairs)},
           "nll_per_expert": {(names[m] if not args.mc_dropout else f"pass{m}"): {"key": float(np.sum([r["nll"][m, 0] for r in rec]) / np.sum([r["tok"][0] for r in rec])),
                                          "nonkey": float(np.sum([r["nll"][m, 1] for r in rec]) / np.sum([r["tok"][1] for r in rec]))}
                              for m in range(M_eff)},
           "runtime_sec": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    m = result
    print(f"[metrics] xong -> {args.out} | js_key={m['js_key_tok']:.4f} js_nonkey={m['js_nonkey_tok']:.4f} "
          f"ratio={m['ratio_tok']:.2f} | nll key={m['nll_key_mean_over_experts']:.3f} nonkey={m['nll_nonkey_mean_over_experts']:.3f}",
          flush=True)


if __name__ == "__main__":
    main()
