#!/usr/bin/env python3
"""Regen results/SUMMARY_vs_palign.md — bảng P@1/P@3 mỗi benchmark, so P-ALIGN.

    python scripts/gen_summary_table.py > results/SUMMARY_vs_palign.md
"""
import json, os, sys

BENCH = ["aime25", "aime24", "amc", "math500"]

GROUPS = [
    ("Bản chính — Qwen2.5-7B-Instruct, expert non-DPP (JS diversity, cổng V FAIL)", [
        ("SGL baseline", "sgl_s1k"),
        ("Ours – gate (rho_bal)", "ours_s1k_rho_bal"),
        ("**Ours – merge TIES**", "ours_v2_phi_merge_ties"),
        ("Ours – merge DARE-TIES", "phi_merge_dare_ties"),
        ("Ours – merge linear", "phi_merge_linear"),
        ("Ours – merge cat (r48)", "phi_merge_cat"),
        ("Ours – distill `_nokl` (λ_kl=0)", "ours_v2_distill_nokl"),
        ("Ours – distill `_distill` (λ_kl=1)", "ours_v2_distill"),
    ]),
    ("proposal-rebuild — DPP+Kneedle experts (§4.3, cổng V PASS: V p90 0.091), merge sweep chưa distill", [
        ("iso_c", "proposal_iso_c_merge"),
        ("ta", "proposal_ta_merge"),
        ("dare_ties", "proposal_dare_ties_merge"),
        ("ties", "proposal_ties_merge"),
        ("tsv (bug cũ)", "proposal_tsv_merge"),
        ("**ctm** (Compress-then-Merge)", "proposal_ctm_merge"),
        ("ctm (ties base)", "proposal_ctm_ties_merge"),
        ("distill `_nokl` trên merge tốt nhất", "proposal_ties_distill_nokl"),
    ]),
    ("Nhánh v2 — phân vùng dữ liệu (frac 0.65, λ_div 0.15) — NEGATIVE (cổng V FAIL)", [
        ("v2p_phi_merge", "v2p_phi_merge"),
        ("v2p_distill_nokl", "v2p_distill_nokl"),
    ]),
]

HDR = ("| Method | AIME25 | AIME24 | AMC | MATH500 | **Avg** P@1/P@3 | tok |\n"
       "|---|--:|--:|--:|--:|--:|--:|")


def fmt_row(name, jf):
    p = f"results/{jf}.json"
    if not os.path.exists(p):
        return f"| {name} | *chưa có* | | | | | |"
    d = json.load(open(p)); b = d["benchmarks"]
    cells = []
    for k in BENCH:
        if k in b:
            cells.append(f"{b[k]['pass@1']*100:.1f}/{b[k]['pass@3']*100:.1f}")
        else:
            cells.append("–")
    return (f"| {name} | " + " | ".join(cells)
            + f" | **{d['avg']*100:.2f}**/{d.get('avg_passk', 0)*100:.1f} | {d['overall_tokens']['mean']:.0f} |")


def main():
    out = []
    out.append("# Kết quả s1K vs P-ALIGN\n")
    out.append("Số mỗi ô = **Pass@1 / Pass@3** (%). Eval: 4 benchmark, k=3, temp 0.6, "
               "top_p 0.9, rep_penalty 1.05, max_tokens 4096 (khớp P-ALIGN). `tok` = token/đáp án.\n")
    out.append("## P-ALIGN (paper, Table 1 — Qwen2.5-7B-Instruct)\n")
    out.append("""| Method | AIME25 | AIME24 | AMC12 | MATH500 | Avg |
|---|--:|--:|--:|--:|--:|
| Zero-Shot | 6.7/10.0 | 10.0/20.0 | 40.8/49.7 | 69.8/84.8 | 31.81/41.13 |
| SFT (Label) | 3.3/6.7 | 3.3/10.0 | 32.9/34.6 | 62.2/71.0 | 25.45/30.57 |
| SFT (Long-CoT) | 10.0/20.0 | 10.0/26.7 | 48.2/56.6 | 75.6/86.6 | 35.95/47.68 |
| UPFT | 13.3/20.0 | 13.3/23.3 | 47.6/59.0 | 74.8/84.6 | 37.26/46.74 |
| **P-ALIGN** | 16.7/26.7 | 16.7/26.7 | 49.4/63.9 | 75.8/85.2 | **39.64**/50.60 |
""")
    for title, rows in GROUPS:
        out.append(f"## {title}\n")
        out.append(HDR)
        for name, jf in rows:
            out.append(fmt_row(name, jf))
        out.append("")
    out.append("""## Nhận xét chính

- **Best hiện tại = merge TIES non-DPP (36.32)** ≈ SFT (Long-CoT) baseline (35.95); thua UPFT (37.26), **thua P-ALIGN 3.3đ** (39.64).
- **Distill (w_i-NLL ± KL) không đóng góp** — cả 3 bản (35.98–36.32) ≈ plain merge; `_nokl` còn gấp đôi token.
- **Nhánh v2 (partition)**: tệ hơn (33.7) — cổng V vẫn FAIL, expert yếu hơn do thấy ít data.
- **proposal DPP (§4.3)**: cổng V **PASS lần đầu** (V p90 0.091 vs JS 0.0005) — DPP+Kneedle làm council bất đồng được. NHƯNG `λ_div=0.1` over-diversified → mọi merge 34–35, thua non-DPP. Đang: sweep λ_div thấp + thử merge CtM (compress-then-merge).
- Điểm yếu xuyên suốt: **AIME** (10–16 vs 16.7) và **AMC** (41–45 vs 49.4). MATH500 luôn ngang paper.
""")
    print("\n".join(out))


if __name__ == "__main__":
    main()
