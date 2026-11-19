"""Download every raw source for the fitting sets and the eval sets into data/raw/ (parquet).

    python scripts/fetch_data.py [--force]                       # online: Hugging Face Hub + GitHub raw files
    python scripts/fetch_data.py --src DIR [--force]             # offline: the same files, already downloaded
    python scripts/fetch_data.py --download-lines /mnt/local/@PROJECT@/selot/raw_src   # download.txt lines for --src

Offline mode (servers without outbound access): every file this script reads is listed by --download-lines as one
`--url` line (`--hf-dataset` for gated repos), each into its own folder <root>/<key>; with --src DIR the file is found
as DIR/<key>/**/<file name>, and AlphaSteer's prompt files as DIR/AlphaLab-USTC__AlphaSteer/**/<file name>.

Fit sources:  alpaca, coconot (original + pref), MATH train, WildGuardMix train (gated: needs `hf auth login`),
              AdvBench, MaliciousInstruct, TDC23-RedTeaming, JBB harmful, Do-Not-Answer, JailbreakV-28K.
Eval sources: AdvBench-100 test split + its attack prompts (GCG, AutoDAN, PAIR, ReNeLLM, Jailbroken, Cipher, AIM) taken
              from the AlphaSteer repo (--alphasteer), XSTest, OR-Bench-Hard-1K, OKTest, AlpacaEval, MMLU, GSM8K.
"""

import argparse
import io
import json
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MATH_SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory",
                 "prealgebra", "precalculus"]
ATTACKS = ["gcg", "autodan", "pair", "renellm", "jailbroken", "cipher", "aim"]   # TAP: no prompt file is public


SRC = None                       # --src: folder of the downloaded files (offline mode)
GATED = {"allenai/wildguardmix", "allenai/coconot"}    # fetched whole with --hf-dataset (the account accepted the terms)
ALPHASTEER_GH = "AlphaLab-USTC/AlphaSteer"
NEEDED = []                      # (kind, source, key) of every file read, for --download-lines


def hf_key(repo, filename):
    if repo in GATED:
        return repo.replace("/", "__")
    parent = str(Path(filename).parent)
    return repo.replace("/", "__") + ("" if parent == "." else "__" + parent.replace("/", "__"))


def find(key, name):
    """the one file DIR/<key>/**/<name>; name may hold folders (a whole-repo download keeps the repo's layout)"""
    hits = sorted((SRC / key).glob(f"**/{name}")) if (SRC / key).exists() else []
    if len(hits) != 1:
        raise SystemExit(f"offline: expected one {name} under {SRC / key}, found {len(hits)} (run the download job)")
    return str(hits[0])


def hf(repo, filename):
    NEEDED.append(("hf", repo, filename))
    if SRC is not None:
        return find(hf_key(repo, filename), filename if repo in GATED else Path(filename).name)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, filename, repo_type="dataset")


def hf_parquet(repo, filename):
    return pd.read_parquet(hf(repo, filename))


def gh_key(url):
    owner, repo = url.split("raw.githubusercontent.com/")[1].split("/")[:2]
    return f"{owner}__{repo}"


def url_text(url):
    NEEDED.append(("url", url, None))
    if SRC is not None:
        return Path(find(gh_key(url), url.rsplit("/", 1)[1])).read_text(encoding="utf-8")
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode("utf-8")


def alphasteer_files():
    return (["data/instructions/test/advbench_test.json"] +
            [f"data/responses/llama3.1/{a}_llama3.1_results.json" for a in ATTACKS])


def alphasteer_file(alpha_dir: Path, rel: str) -> Path:
    if SRC is not None:
        return Path(find(ALPHASTEER_GH.replace("/", "__"), Path(rel).name))
    return alpha_dir / rel


def alphasteer_eval(alpha_dir: Path) -> pd.DataFrame:
    """AdvBench-100 test questions + the attack prompts AlphaSteer evaluates on (Llama-3.1 versions)."""
    rows = [dict(eval_set="advbench_test", prompt=d["query"], base=d["query"])
            for d in json.load(open(alphasteer_file(alpha_dir, "data/instructions/test/advbench_test.json")))]
    for a in ATTACKS:
        for d in json.load(open(alphasteer_file(alpha_dir, f"data/responses/llama3.1/{a}_llama3.1_results.json"))):
            if a == "gcg":                       # 'query' has the Llama chat template baked in
                text, base = d["prompt"] + " " + d["adv_suffix"], d["prompt"]
            elif a == "cipher":                  # CipherChat (Caesar / SelfDefine / ASCII / Morse); plain question kept too
                text, base = d["query"], d["original_query"]
                rows.append(dict(eval_set=f"attack_{a}_plain", prompt=base, base=base))
            else:
                text, base = d["query"], None
            rows.append(dict(eval_set=f"attack_{a}", prompt=text, base=base))
    return pd.DataFrame(rows)


def sources(alpha_dir: Path):
    return {
        # ---------------- fit
        "alpaca": lambda: hf_parquet("tatsu-lab/alpaca", "data/train-00000-of-00001-a09b74b3ef9c3b56.parquet"),
        "coconot_original": lambda: hf_parquet("allenai/coconot", "original/train-00000-of-00001.parquet"),
        "coconot_pref": lambda: hf_parquet("allenai/coconot", "pref/train-00000-of-00001.parquet"),
        "math_train": lambda: pd.concat([hf_parquet("EleutherAI/hendrycks_math", f"{s}/train-00000-of-00001.parquet")
                                         for s in MATH_SUBJECTS], ignore_index=True),
        "wgm_train": lambda: hf_parquet("allenai/wildguardmix", "train/wildguard_train.parquet"),
        "advbench": lambda: hf_parquet("walledai/AdvBench", "data/train-00000-of-00001.parquet"),
        "mi": lambda: hf_parquet("walledai/MaliciousInstruct", "data/train-00000-of-00001.parquet"),
        "tdc23": lambda: hf_parquet("walledai/TDC23-RedTeaming", "data/train-00000-of-00001.parquet"),
        "jbb_harmful": lambda: pd.read_csv(hf("JailbreakBench/JBB-Behaviors", "data/harmful-behaviors.csv")),
        "dna": lambda: pd.read_csv(hf("LibrAI/do-not-answer", "data_en.csv")),
        "jailbreakv": lambda: pd.read_csv(hf("JailbreakV-28K/JailBreakV-28k", "JailBreakV_28K/JailBreakV_28K.csv")),
        # ---------------- eval
        "eval_alphasteer": lambda: alphasteer_eval(alpha_dir),
        "xstest": lambda: pd.read_csv(io.StringIO(url_text("https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv"))),
        "orbench_hard": lambda: pd.read_csv(hf("bench-llm/or-bench", "or-bench-hard-1k.csv")),
        "oktest": lambda: pd.concat([
            pd.read_csv(io.StringIO(url_text(f"https://raw.githubusercontent.com/InvokerStark/OverKill/main/data/{f}")))
            .assign(split=f) for f in ["OKTest.csv", "OKTest_heldout.csv"]], ignore_index=True),
        "alpaca_eval": lambda: pd.DataFrame(json.load(open(hf("tatsu-lab/alpaca_eval", "alpaca_eval.json")))),
        "mmlu": lambda: hf_parquet("cais/mmlu", "all/test-00000-of-00001.parquet"),
        "gsm8k": lambda: hf_parquet("openai/gsm8k", "main/test-00000-of-00001.parquet"),
    }


def download_lines(root: str):
    """download.txt lines (new_nothing format) for every file the sources read, into root/<key>"""
    import unittest.mock as um
    global SRC
    NEEDED.clear()
    SRC = None
    # record the files without fetching: run every source with stub readers
    fake = um.MagicMock()
    with um.patch(f"{__name__}.hf", side_effect=lambda repo, fn: NEEDED.append(("hf", repo, fn)) or "x"), \
         um.patch(f"{__name__}.url_text", side_effect=lambda url: NEEDED.append(("url", url, None)) or "a\n1\n"), \
         um.patch("pandas.read_parquet", return_value=fake), um.patch("pandas.read_csv", return_value=fake), \
         um.patch("json.load", return_value=[]), um.patch("builtins.open", um.mock_open(read_data="[]")):
        for name, fn in sources(Path("/nonexistent")).items():
            if name != "eval_alphasteer":
                try:
                    fn()
                except Exception:
                    pass
    lines, seen = [], set()
    for kind, a, b in NEEDED:
        if kind == "hf" and a in GATED:
            line = f"--hf-dataset {a} {root}/{hf_key(a, b)}"
        elif kind == "hf":
            line = f"--url https://huggingface.co/datasets/{a}/resolve/main/{b} {root}/{hf_key(a, b)}"
        else:
            line = f"--url {a} {root}/{gh_key(a)}"
        if line not in seen:
            seen.add(line)
            lines.append(line)
    for rel in alphasteer_files():
        lines.append(f"--url https://raw.githubusercontent.com/{ALPHASTEER_GH}/main/{rel} "
                     f"{root}/{ALPHASTEER_GH.replace('/', '__')}/{Path(rel).parent}")
    return lines


def main():
    global SRC
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--alphasteer", default=str(ROOT.parent / "AlphaSteer"), help="local clone of AlphaSteer")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--src", default=None, help="offline: folder of the files listed by --download-lines")
    ap.add_argument("--download-lines", default=None, metavar="ROOT", help="print the download.txt lines and exit")
    ap.add_argument("--raw", default=None, help="output folder (default data/raw)")
    a = ap.parse_args()
    global RAW
    if a.raw:
        RAW = Path(a.raw)
    if a.download_lines:
        print("\n".join(download_lines(a.download_lines)))
        return
    if a.src:
        SRC = Path(a.src)
    RAW.mkdir(parents=True, exist_ok=True)
    for name, fn in sources(Path(a.alphasteer)).items():
        if a.only and name not in a.only:
            continue
        out = RAW / f"{name}.parquet"
        if out.exists() and not a.force:
            print(f"skip {name} (exists)")
            continue
        df = fn()
        df.columns = [str(c) for c in df.columns]
        for c in df.columns:                      # nested objects (lists, dicts) -> json strings for parquet
            if df[c].map(lambda v: isinstance(v, (list, dict))).any():
                df[c] = df[c].map(lambda v: json.dumps(v) if isinstance(v, (list, dict)) else v)
        df.to_parquet(out, index=False)
        print(f"{name}: {len(df)} rows, columns {list(df.columns)[:10]}")


if __name__ == "__main__":
    main()
