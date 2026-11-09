"""Gộp ``results/*.json`` thành ``results/table.md`` (spec §10).

Mọi con số đọc từ file kết quả của run thật — script này không nhận số nhập tay
và sẽ để trống ô nếu file tương ứng chưa tồn tại.

    python -m eval.make_table
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional

MAIN_ROWS = [
    ("Base (no adapter)", "base_no_adapter"),
    ("Vanilla SFT", "vanilla_sft"),
    ("Entropy Selection", "entropy_sft"),
    ("Ours (U+V)", "ours"),
    ("Ours (U + rho)", "ours_rho"),
    ("Ours (U + rho + Balanced)", "ours_rho_bal"),
    ("Ours (Dual-Ortho + JS + rho, no bal)", "ours_div_rho_nobal"),
    ("Ours (Dual-Ortho + JS + rho + bal)", "ours_div_rho_bal"),
    ("Ours (Dual-Ortho + JS + rho)", "ours_div_rho"),
]

ABLATION_ROWS = [
    ("Ours (U+V)", "ours"),
    ("U-only (lambda_D=0)", "abl_u_only"),
    ("V-only (lambda_U=0)", "abl_v_only"),
    ("uniform (w=1)", "abl_uniform"),
    ("M=1 (no council)", "abl_m1"),
    ("segmentation=sentence", "abl_sentence"),
    ("Ours (dynamic routing)", "ours_dynamic"),
]

COLUMNS = [("GSM8K", "gsm8k"), ("MATH500", "math500"),
           ("AIME24", "aime24"), ("AIME25", "aime25")]

# --- Kịch bản CurioSFT (--preset curiosft) ---
CS_ROWS = [
    ("Base (no adapter)", "base_no_adapter"),
    ("Vanilla SFT (Old)", "vanilla_sft"),
    ("Vanilla SFT (Fixed)", "vanilla_sft_fixed"),
    ("CurioSFT (Old)", "curiosft"),
    ("CurioSFT (Fixed)", "curiosft_fixed"),
    ("Entropy Selection", "entropy_sft"),
    ("Ours + CurioSFT", "ours_curio"),
    ("Ours (U+rho, no SED)", "ours_rho"),
]
CS_ID_COLS = [("AIME24", "cs_aime24"), ("AIME25", "cs_aime25"), ("AMC", "cs_amc"),
              ("MATH500", "cs_math500"), ("Minerva", "cs_minerva"), ("Olymp", "cs_olympiad")]
CS_OOD_COLS = [("GPQA", "cs_gpqa"), ("MMLU-Pro", "cs_mmlu_pro"), ("ARC-C", "cs_arc_c")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build results/table.md")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--out", default="results/table.md")
    parser.add_argument("--preset", default="default", choices=["default", "curiosft"])
    return parser.parse_args()


def load_results(results_dir: str) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for path in glob.glob(os.path.join(results_dir, "*.json")):
        with open(path, "r", encoding="utf-8") as handle:
            try:
                data = json.load(handle)
            except json.JSONDecodeError:
                continue
        if isinstance(data, dict) and "benchmarks" in data:
            out[data.get("method", os.path.splitext(os.path.basename(path))[0])] = data
    return out


def _cell(data: Optional[dict], key: str) -> str:
    if not data:
        return ""
    entry = data.get("benchmarks", {}).get(key)
    if not entry:
        return ""
    return f"{entry['pass@1'] * 100:.2f}"


def _avg_cell(data: Optional[dict], columns: List[str]) -> str:
    """Avg chỉ tính khi có đủ mọi cột — tránh so Avg trên tập benchmark khác nhau."""
    if not data:
        return ""
    values = []
    for key in columns:
        entry = data.get("benchmarks", {}).get(key)
        if not entry:
            return ""
        values.append(entry["pass@1"])
    return f"{sum(values) / len(values) * 100:.2f}"


def render_table(results: Dict[str, dict], rows) -> str:
    header = "| Method | " + " | ".join(name for name, _ in COLUMNS) + " | Avg |"
    sep = "|---" * (len(COLUMNS) + 2) + "|"
    lines = [header, sep]
    for label, method in rows:
        data = results.get(method)
        cells = [_cell(data, key) for _, key in COLUMNS]
        lines.append(
            f"| {label} | " + " | ".join(cells) + " | "
            + _avg_cell(data, [key for _, key in COLUMNS]) + " |"
        )
    return "\n".join(lines)


def render_curiosft_table(results: Dict[str, dict]) -> str:
    id_keys = [k for _, k in CS_ID_COLS]
    ood_keys = [k for _, k in CS_OOD_COLS]
    head = (["Method"] + [n for n, _ in CS_ID_COLS] + ["**ID Avg**"]
            + [n for n, _ in CS_OOD_COLS] + ["**OOD Avg**"])
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for label, method in CS_ROWS:
        data = results.get(method)
        cells = ([label]
                 + [_cell(data, k) for k in id_keys] + [_avg_cell(data, id_keys)]
                 + [_cell(data, k) for k in ood_keys] + [_avg_cell(data, ood_keys)])
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_curiosft_token_table(results: Dict[str, dict]) -> str:
    id_keys = [k for _, k in CS_ID_COLS]
    head = ["Method", "**Overall Mean**", "**Min**", "**Max**"] + [n for n, _ in CS_ID_COLS]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for label, method in CS_ROWS:
        data = results.get(method)
        if not data:
            continue
        overall = data.get("overall_tokens") or {}
        o_mean = f"{overall.get('mean', 0):.1f}" if "mean" in overall else "—"
        o_min = f"{overall.get('min', 0)}" if "min" in overall else "—"
        o_max = f"{overall.get('max', 0)}" if "max" in overall else "—"

        bench_cells = []
        for k in id_keys:
            b_info = data.get("benchmarks", {}).get(k, {})
            t_stats = b_info.get("token_stats")
            if t_stats and "mean" in t_stats:
                bench_cells.append(f"{t_stats['mean']:.1f}")
            else:
                bench_cells.append("—")
        cells = [label, o_mean, o_min, o_max] + bench_cells
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    results = load_results(args.results_dir)

    if args.preset == "curiosft":
        out = args.out if args.out != "results/table.md" else "results/table_curiosft.md"
        body = [
            "# Kết quả — kịch bản CurioSFT (Avg@8, %)",
            "",
            "Bench = `Hao0oWang/CurioSFT_Data/test.*.parquet` (đúng tập bài của bài báo).",
            "MMLU-Pro n=1, còn lại k=8. Số từ `eval/run_eval.py`.",
            "",
            "## 1. Độ chính xác (Accuracy / Pass@1)",
            "",
            render_curiosft_table(results),
            "",
            "## 2. Thống kê Output Tokens (Mean / Min / Max)",
            "",
            render_curiosft_token_table(results),
            "",
            f"> Chưa có: {', '.join(m for _, m in CS_ROWS if m not in results) or '—'}",
        ]
        v, c = results.get("vanilla_sft"), results.get("curiosft")
        if v and c:
            dv = (c.get("avg", 0) - v.get("avg", 0)) * 100
            body += ["", f"**curiosft − vanilla (avg): {dv:+.2f}**"]
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w", encoding="utf-8") as h:
            h.write("\n".join(body) + "\n")
        print(f"[table] -> {out}")
        print("\n".join(body))
        return

    parts = [
        "# Kết quả (Pass@1, %)",
        "",
        "Số sinh từ `eval/run_eval.py` — không nhập tay (spec §14).",
        "",
        "## Bảng chính",
        "",
        render_table(results, MAIN_ROWS),
        "",
        "## Ablation (spec §9)",
        "",
        render_table(results, ABLATION_ROWS),
        "",
        "## Cấu hình của từng run",
        "",
        "| Method | engine | ours_mode | adapter | temp | top_p | max_tokens |",
        "|---|---|---|---|---|---|---|",
    ]
    for method, data in sorted(results.items()):
        sampling = data.get("sampling", {})
        parts.append(
            f"| {method} | {data.get('engine', '')} | {data.get('ours_mode') or ''} | "
            f"`{data.get('adapter') or ''}` | {sampling.get('temperature', '')} | "
            f"{sampling.get('top_p', '')} | {sampling.get('max_tokens', '')} |"
        )

    missing = [m for _, m in MAIN_ROWS if m not in results]
    if missing:
        parts += ["", f"> Chưa có kết quả cho: {', '.join(missing)}"]

    # Sanity check §14: ablation uniform phải trùng Vanilla SFT.
    vanilla, uniform = results.get("vanilla_sft"), results.get("abl_uniform")
    if vanilla and uniform:
        parts += ["", "## Sanity check §14 — `uniform` phải trùng `Vanilla SFT`", ""]
        for label, key in COLUMNS:
            a = vanilla.get("benchmarks", {}).get(key)
            b = uniform.get("benchmarks", {}).get(key)
            if not a or not b:
                continue
            delta = (b["pass@1"] - a["pass@1"]) * 100
            flag = "OK" if abs(delta) < 1.0 else "**LỆCH — kiểm tra weighted loss**"
            parts.append(f"- {label}: Vanilla {a['pass@1']*100:.2f} vs uniform "
                         f"{b['pass@1']*100:.2f} (Δ = {delta:+.2f}) {flag}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write("\n".join(parts) + "\n")
    print(f"[table] -> {args.out}")
    print("\n".join(parts[:12]))


if __name__ == "__main__":
    main()
