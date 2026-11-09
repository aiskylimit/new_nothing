"""Gom bảng ablation Phase 1: avg pass@1 (trung bình 3 expert), js-key, js-nonkey, tỉ số, NLL. In markdown."""
import json, os, sys
SC = ["full", "noDropout", "noSVGD", "noOrtho", "onlySVGD", "onlyOrtho", "onlyDropout", "allOff"]
BM = ["math500", "aime24", "aime25", "amc"]
for model in (sys.argv[1:] or ["7b", "8b"]):
    print(f"\n### {model}\n")
    print("| kịch bản | avg P@1 (3 expert) | math500 | aime24 | aime25 | amc | js-key | js-nonkey | key/nonkey | NLL key | NLL nonkey |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for sc in SC:
        T = f"abl_{model}_{sc}"
        ev = [json.load(open(p)) for i in range(3) if os.path.exists(p := f"results/{T}_expert{i}_exact.json")]
        # allOff 8B chỉ train M=1 (1 expert); js đo bằng MC-dropout trên expert đó
        js = next((json.load(open(p)) for p in (f"results/ablation_metrics/{T}_js.json", f"results/ablation_metrics/{T}_mcdropout_js.json") if os.path.exists(p)), None)
        m1 = len(ev) == 1 and sc == "allOff"
        cells = ["-"] * 5
        if len(ev) == 3:
            per = [100 * sum(e["benchmarks"][b]["pass@1"] for b in BM) / 4 for e in ev]
            cells = [f"{sum(per)/3:.2f}"] + [f"{100*sum(e['benchmarks'][b]['pass@1'] for e in ev)/3:.2f}" for b in BM]
        elif m1:
            per = 100 * sum(ev[0]["benchmarks"][b]["pass@1"] for b in BM) / 4
            cells = [f"{per:.2f} (M=1)"] + [f"{100*ev[0]['benchmarks'][b]['pass@1']:.2f}" for b in BM]
        elif ev:
            cells[0] = f"({len(ev)}/3 expert)"
        j = ["-"] * 5
        if js:
            m, ci = js["metrics"], js["ci95_bootstrap_over_samples"]
            j = [f"{m['js_key_tok']:.4f} [{ci['js_key_tok'][0]:.4f},{ci['js_key_tok'][1]:.4f}]",
                 f"{m['js_nonkey_tok']:.4f} [{ci['js_nonkey_tok'][0]:.4f},{ci['js_nonkey_tok'][1]:.4f}]",
                 f"{m['ratio_tok']:.2f}", f"{m['nll_key_mean_over_experts']:.3f}", f"{m['nll_nonkey_mean_over_experts']:.3f}"]
        print(f"| {sc} | " + " | ".join(cells + j) + " |")
