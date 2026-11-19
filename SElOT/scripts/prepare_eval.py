"""Write every eval set to data/eval/<set>.jsonl with the fields its metric needs (from data/raw/, fetch_data.py).

    python scripts/prepare_eval.py

Sets and fields (all rows have id, prompt):
  advbench_test            behavior (= prompt), target
  attack_{gcg,autodan,pair,renellm,jailbroken,cipher,aim}
                           behavior = the AdvBench-100 question the attack was built on, behavior_id,
                           behavior_match (how it was found, see below), behavior_score;
                           attack_cipher also has cipher (CipherChat mutation_type: CaesarExpert, SelfDefineCipher,
                           AsciiExpert, MorseExpert), used to decode the responses before judging (selot/cipher.py)
  xstest_safe / xstest_unsafe   type, focus (XSTest: 250 safe prompts measure over-refusal, 200 unsafe contrast prompts)
  orbench_hard             category
  oktest                   (prompt only)
  alpaca_eval              dataset, reference (AlpacaEval 805 instructions; reference = text-davinci-003 output, the
                           opponent in AlphaSteer's protocol; scored by the reward model)
  mmlu                     subject, choices (4), answer (A-D)
  gsm8k                    answer (the number after ####)

Behavior of an attack prompt, first that works: `field` (GCG prompt, Cipher original_query), `contain` (the question
or its AdvBench target "Sure, here is ..." is in the prompt after decoding base64 / ROT13 / leetspeak, or its consonant
skeleton is in the prompt's: disemvowel), `index` (ReNeLLM keeps the
AdvBench-100 order), `assign` (PAIR, 100 prompts for 100 questions: optimal one-to-one assignment on MiniLM
cosine for the rest), `nearest` (Jailbroken, 5 mutations per question but some are paraphrases: nearest question by
MiniLM cosine). The HarmBench and StrongREJECT judges grade the response against this behavior.
"""

import base64
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, EVAL = ROOT / "data" / "raw", ROOT / "data" / "eval"
# AlphaSteer's prompt files: its clone next to SElOT, or SELOT_ALPHASTEER_DIR (offline: the downloaded files,
# found by name under that folder)
ALPHASTEER = Path(os.environ.get("SELOT_ALPHASTEER_DIR") or ROOT.parent / "AlphaSteer")


def alphasteer_file(rel: str) -> Path:
    p = ALPHASTEER / rel
    if p.exists():
        return p
    hits = sorted(ALPHASTEER.glob(f"**/{Path(rel).name}"))
    if len(hits) != 1:
        raise SystemExit(f"expected one {Path(rel).name} under {ALPHASTEER}, found {len(hits)}")
    return hits[0]
ATTACKS = ["gcg", "autodan", "pair", "renellm", "jailbroken", "cipher", "aim"]
INDEX_ALIGNED = {"autodan", "renellm"}   # checked: same order as advbench_test.json


def norm(s):
    return " ".join(re.findall(r"[a-z0-9]+", str(s).lower()))


def decode_b64(s):
    out = []
    for m in re.findall(r"[A-Za-z0-9+/=]{24,}", s):
        try:
            out.append(base64.b64decode(m + "=" * (-len(m) % 4)).decode("utf-8", "ignore"))
        except ValueError:
            pass
    return " ".join(out)


LEET = str.maketrans({"@": "a", "4": "a", "3": "e", "1": "i", "!": "i", "0": "o", "5": "s", "$": "s", "7": "t"})


def skeleton(s):
    """consonants only: matches a disemvowelled prompt against its question"""
    return re.sub(r"[aeiou ]", "", norm(s))


def variants(prompt):
    """normalised texts the question can appear in: plain, base64 segments, ROT13, leetspeak"""
    import codecs
    return [norm(prompt + " " + decode_b64(prompt)), norm(codecs.decode(prompt, "rot13")),
            norm(prompt.lower().translate(LEET))]


def write(name, rows):
    EVAL.mkdir(parents=True, exist_ok=True)
    with open(EVAL / f"{name}.jsonl", "w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"id": f"{name}_{i:05d}", **r}, ensure_ascii=False) + "\n")
    print(f"{name}: {len(rows)}")


def attack_behaviors(base, targets):
    """{attack: list of rows}"""
    from build_fit_sets import embed
    from scipy.optimize import linear_sum_assignment
    B, T = [norm(b) for b in base], [norm(t) for t in targets]
    SK = [skeleton(b) for b in base]
    Eb = embed(base, "advbench_test_base", "cpu")
    out = {}
    for a in ATTACKS:
        recs = json.load(open(alphasteer_file(f"data/responses/llama3.1/{a}_llama3.1_results.json")))
        rows, beh = [], [None] * len(recs)
        for i, d in enumerate(recs):
            if a == "gcg":
                prompt, field = d["prompt"] + " " + d["adv_suffix"], d["prompt"]
            elif a == "cipher":
                prompt, field = d["query"], d["original_query"]
            else:
                prompt, field = d["query"], None
            rows.append({"prompt": prompt, "cipher": d["mutation_type"]} if a == "cipher" else {"prompt": prompt})
            if field is not None:
                j = B.index(norm(field))
                beh[i] = (j, "field", 1.0)
                continue
            vs = variants(prompt)
            hit = sorted({j for n in vs for j in range(100) if B[j] in n}) or \
                sorted({j for n in vs for j in range(100) if T[j] and T[j] in n}) or \
                [j for j in range(100) if len(SK[j]) >= 15 and SK[j] in skeleton(prompt)]
            if len(hit) == 1:
                beh[i] = (hit[0], "contain", 1.0)
            elif a in INDEX_ALIGNED:
                beh[i] = (i % 100, "index", 1.0)
        todo = [i for i, b in enumerate(beh) if b is None]
        if todo:
            reps = len(recs) // 100
            used = np.bincount([b[0] for b in beh if b is not None], minlength=100)
            slots = [j for j in range(100) for _ in range(max(reps - used[j], 0))]
            Ep = embed([rows[i]["prompt"] for i in todo], f"attack_{a}_todo", "cpu")
            S = Ep @ Eb.T
            if reps == 1:      # one prompt per question (PAIR): one-to-one assignment
                r, c = linear_sum_assignment(-S[:, slots])
                for ri, ci in zip(r, c):
                    j = slots[ci]
                    beh[todo[ri]] = (j, "assign", float(S[ri, j]))
            else:              # several mutations per question, not all exactly reps times: nearest question
                for ri, i in enumerate(todo):
                    j = int(S[ri].argmax())
                    beh[i] = (j, "nearest", float(S[ri, j]))
        for row, (j, how, score) in zip(rows, beh):
            row.update(behavior=base[j], behavior_id=j, behavior_match=how, behavior_score=round(score, 3))
        out[a] = rows
        print(f"attack_{a}: behavior match {pd.Series([b[1] for b in beh]).value_counts().to_dict()}")
    return out


def main():
    base = [d["query"] for d in json.load(open(alphasteer_file("data/instructions/test/advbench_test.json")))]
    adv = pd.read_parquet(RAW / "advbench.parquet")
    tmap = dict(zip(adv.prompt.str.strip(), adv.target.str.strip()))
    targets = [tmap[b.strip()] for b in base]
    write("advbench_test", [{"prompt": b, "behavior": b, "target": t} for b, t in zip(base, targets)])
    for a, rows in attack_behaviors(base, targets).items():
        write(f"attack_{a}", rows)

    xs = pd.read_parquet(RAW / "xstest.parquet")
    for lab in ["safe", "unsafe"]:
        g = xs[xs.label == lab]
        write(f"xstest_{lab}", [{"prompt": p, "type": t, "focus": f} for p, t, f in zip(g.prompt, g.type, g.focus)])
    orb = pd.read_parquet(RAW / "orbench_hard.parquet")
    write("orbench_hard", [{"prompt": p, "category": c} for p, c in zip(orb.prompt, orb.category)])
    write("oktest", [{"prompt": p.strip()} for p in pd.read_parquet(RAW / "oktest.parquet").prompt])
    ae = pd.read_parquet(RAW / "alpaca_eval.parquet")
    write("alpaca_eval", [{"prompt": p, "dataset": d, "reference": o, "reference_generator": g}
                          for p, d, o, g in zip(ae.instruction, ae.dataset, ae.output, ae.generator)])
    mm = pd.read_parquet(RAW / "mmlu.parquet")
    write("mmlu", [{"prompt": q, "subject": s, "choices": list(json.loads(c) if isinstance(c, str) else c),
                    "answer": "ABCD"[int(a)]} for q, s, c, a in zip(mm.question, mm.subject, mm.choices, mm.answer)])
    gs = pd.read_parquet(RAW / "gsm8k.parquet")
    write("gsm8k", [{"prompt": q, "answer": a.split("####")[-1].strip().replace(",", "")}
                    for q, a in zip(gs.question, gs.answer)])


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
