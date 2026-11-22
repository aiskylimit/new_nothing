"""Steps (i)-(iv) of the fitting data, model-independent: candidate pools -> dedup against every eval set -> harm
topics -> behaviour groups and the fixed train/val split. Steps (v)-(vi) (behaviour labelling with the model,
per-topic cap within each split, topic weights) run per model in the `label` stage (selot/fit_data.py).

    python scripts/build_fit_sets.py [--emb-th 0.85] [--device cpu] [--skip-bleu]
    python scripts/build_fit_sets.py --no-dedup --out data/ablation_nodedup   # same pools without step (ii), to
                                                                              # measure the train-test overlap

(i) Candidate pools.
     benign (~16.5k)   alpaca 10000, coconot 4000 (all of pref + original minus "safety concerns"), math_train 900,
                       WildGuardMix benign 150 per topic (vanilla and adversarial halves)
     refusal (~3.6k)   vanilla harmful: every prompt of advbench (520 minus AlphaSteer's 100 test prompts),
                       malicious_instruct, tdc23, jbb_harmful and do_not_answer; WildGuardMix vanilla harmful fills
                       up to POOL_SIZE, given to the topics with the fewest prompts first
     jailbreak (~3k)   JailbreakV-28K text-only, 1000 (spread evenly over topics, <= 2 variants per redteam_query,
                       formats interleaved); WildGuardMix adversarial harmful fills up to POOL_SIZE, fewest-first
   Harmful prompts outside the 11 topics (out_of_scope, uncertain) are not pooled.
(ii) Dedup on the whole raw pool, before any sampling: a prompt is dropped when it overlaps any eval prompt
   (AdvBench-100 test and AlphaSteer's attacks on it, XSTest, OR-Bench-Hard-1K, OKTest, AlpacaEval, MMLU, GSM8K):
     exact    equal after normalisation (lowercase, alphanumerics only); JailbreakV also on its redteam_query
     contain  one normalised text inside the other (shorter side >= 6 words)
     emb      all-MiniLM-L6-v2 cosine >= emb_th; JailbreakV also on its redteam_query
     bleu     sentence-BLEU >= 0.5 with an eval prompt (each eval prompt's 50 TF-IDF nearest pool prompts, repeated
              with flagged prompts masked out until nothing new is found)
(iii) Topics (scripts/topics.py).
(iv) Behaviour groups and split, one pass over the three pools together, so the vanilla and the jailbreak version of
   one behaviour always land on the same side. Text = redteam_query when there is one, else the prompt. Leader
   clustering (no chaining): the prompt with the most MiniLM neighbours at cosine >= split_th leads a group of every
   ungrouped neighbour, repeat; variants of one redteam_query are always merged. Then 1 / val_folds of every stratum
   (harmful: set x topic, benign: source) goes to val, whole groups at a time, fixed seed. Every model uses this split.
   Workflow 1 (fit_data.workflow=1, the pipeline of the first Colab run) does not use it: it reads `group_v1`, written
   here too, = connected components of MiniLM cosine >= emb_th within each pool (prompt embedding, the redteam_query's
   when there is one) plus equal redteam_query, and splits per model in the label stage after the topic cap.

Outputs: data/fit/{benign,refusal,jailbreak}.jsonl (id, prompt, source, source_label, topic, topic_method, group,
split, group_v1 [, redteam_query]); data/reports/dedup_report.md, composition.md, removed.jsonl, bleu.json.
"""

import argparse
import os
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from topics import TOPICS, label_topics  # noqa: E402
from selot.fit_data import split_val  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW, FIT, EVAL, REP, EMB = (ROOT / "data" / d for d in ("raw", "fit", "eval", "reports", "emb"))
SEED = 0
# SELOT_EMB_MODEL: a local folder of the same model (offline servers); the cache key keeps the hub name so that the
# cached embeddings are shared
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMB_PATH = os.environ.get("SELOT_EMB_MODEL") or EMB_MODEL

POOL_SIZE = {"refusal": 3600, "jailbreak": 3000}
CURATED = {"refusal": ["advbench", "malicious_instruct", "tdc23", "jbb_harmful", "do_not_answer"],
           "jailbreak": ["jailbreakv"]}
CURATED_TAKE = {"jailbreakv": 1000}            # None / absent = every clean in-topic prompt
FILLER = {"refusal": "wgm_harmful_vanilla", "jailbreak": "wgm_harmful_adv"}
BENIGN_FIXED = {"alpaca": 10000, "coconot": 4000, "math_train": 900}
WGM_BENIGN_PER_TOPIC = 150
RAW_CAP = {"alpaca": 30000, "math_train": 2700, "wgm_benign_vanilla": 8000, "wgm_benign_adv": 8000}
JBV_POOL_VARIANTS, JBV_MAX_VARIANTS = 3, 2
BLEU_TH, BLEU_K = 0.5, 50
SPLIT_TH, VAL_FOLDS = 0.7, 7       # behaviour groups (step iv); 1/7 of every stratum is held out
EVAL_GROUPS = {
    "safety": ["advbench_test", "attack_gcg", "attack_autodan", "attack_pair", "attack_renellm", "attack_jailbroken",
               "attack_cipher", "attack_cipher_plain", "attack_aim"],
    "over_refusal": ["xstest", "orbench_hard", "oktest"],
    "utility": ["alpaca_eval", "mmlu", "gsm8k"],
}
SET_TITLE = {"refusal": "D_r (refusal)", "jailbreak": "D_j (jailbreak)", "benign": "D_b (benign)"}


def norm(s):
    return " ".join(re.findall(r"[a-z0-9]+", str(s).lower()))


def fix_mojibake(s):
    """JailbreakV has UTF-8 text decoded as GBK in places ('鈥檛' for '’t')."""
    if not isinstance(s, str) or "鈥" not in s:
        return s
    try:
        return s.encode("gb18030").decode("utf-8")
    except UnicodeError:
        return s.replace("鈥檛", "'t").replace("鈥檚", "'s").replace("鈥", "'")


def raw(name):
    return pd.read_parquet(RAW / f"{name}.parquet")


# ---------------------------------------------------------------- eval prompts
def load_eval():
    ev = raw("eval_alphasteer")[["eval_set", "prompt"]]
    xs = raw("xstest").assign(eval_set="xstest")[["eval_set", "prompt"]]
    orb = raw("orbench_hard").assign(eval_set="orbench_hard")[["eval_set", "prompt"]]
    ok = raw("oktest").assign(eval_set="oktest")[["eval_set", "prompt"]]
    ae = raw("alpaca_eval").rename(columns={"instruction": "prompt"}).assign(eval_set="alpaca_eval")[["eval_set", "prompt"]]
    mm = raw("mmlu").rename(columns={"question": "prompt"}).assign(eval_set="mmlu")[["eval_set", "prompt"]]
    gs = raw("gsm8k").rename(columns={"question": "prompt"}).assign(eval_set="gsm8k")[["eval_set", "prompt"]]
    E = pd.concat([ev, xs, orb, ok, ae, mm, gs], ignore_index=True)
    E["prompt"] = E.prompt.astype(str).str.strip()
    return E[E.prompt.str.len() > 0].reset_index(drop=True)


# ---------------------------------------------------------------- pools
def load_pools(rng):
    P = []

    def add(set_, src, df, prompt, label=None, base=None):
        d = pd.DataFrame({"prompt": df[prompt].astype(str).str.strip(),
                          "source_label": df[label].astype(str) if label else "",
                          "base": df[base].astype(str).str.strip() if base else None})
        d = d[d.prompt.str.len() > 0]
        cap = RAW_CAP.get(src)
        if cap and len(d) > cap:
            d = d.sample(n=cap, random_state=rng)
        P.append(d.assign(set=set_, source=src))

    # curated harmful benchmarks first: they keep their copy of an exact duplicate
    test = set(raw("eval_alphasteer").query("eval_set == 'advbench_test'").prompt.map(norm))
    adv = raw("advbench")
    adv = adv[~adv.prompt.map(norm).isin(test)]
    assert len(adv) == 420, len(adv)
    add("refusal", "advbench", adv, "prompt")
    add("refusal", "malicious_instruct", raw("mi"), "prompt")
    add("refusal", "tdc23", raw("tdc23"), "prompt")
    add("refusal", "jbb_harmful", raw("jbb_harmful"), "Goal", "Category")
    dna = raw("dna").assign(lab=lambda d: d.risk_area + "|" + d.types_of_harm + "|" + d.specific_harms)
    add("refusal", "do_not_answer", dna, "question", "lab")

    jv = raw("jailbreakv")
    jv = jv[jv.format.isin(["Template", "Persuade", "Logic"])].copy()       # text-only (no image needed)
    jv["jailbreak_query"] = jv.jailbreak_query.map(fix_mojibake)
    jv["redteam_query"] = jv.redteam_query.map(fix_mojibake)
    jv["lab"] = jv.policy + "|" + jv.format
    jv = jv.drop_duplicates("jailbreak_query").sample(frac=1, random_state=rng)
    jv["_f"] = jv.groupby(["redteam_query", "format"]).cumcount()            # spread the variants over formats
    jv = jv.sort_values("_f", kind="stable")
    jv = jv[jv.groupby("redteam_query").cumcount() < JBV_POOL_VARIANTS]
    add("jailbreak", "jailbreakv", jv, "jailbreak_query", "lab", base="redteam_query")

    w = raw("wgm_train").dropna(subset=["prompt", "prompt_harm_label"])
    w["subcategory"] = w.subcategory.fillna("")
    for harm, adv_, src, set_ in [("harmful", False, "wgm_harmful_vanilla", "refusal"),
                                  ("harmful", True, "wgm_harmful_adv", "jailbreak"),
                                  ("unharmful", False, "wgm_benign_vanilla", "benign"),
                                  ("unharmful", True, "wgm_benign_adv", "benign")]:
        add(set_, src, w[(w.prompt_harm_label == harm) & (w.adversarial == adv_)], "prompt", "subcategory")

    al = raw("alpaca")
    al["p"] = [(i + ("\n\n" + x if x else "")).strip() for i, x in zip(al.instruction, al.input)]
    add("benign", "alpaca", al, "p")
    co_pref = raw("coconot_pref").assign(lab=lambda d: d.category + "|" + d.subcategory)
    co_orig = raw("coconot_original")
    co_orig = co_orig[co_orig.category != "Requests with safety concerns"].assign(lab=lambda d: d.category + "|" + d.subcategory)
    add("benign", "coconot", pd.concat([co_pref, co_orig]), "prompt", "lab")
    add("benign", "math_train", raw("math_train"), "problem", "type")

    C = pd.concat(P, ignore_index=True)
    C["n"] = C.prompt.map(norm)
    before = len(C)
    C = C[C.n.str.len() > 0].drop_duplicates("n").reset_index(drop=True)
    return C, before - len(C)


# ---------------------------------------------------------------- overlap detection
def embed(texts, tag, device):
    import hashlib
    EMB.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(("\x00".join(texts) + EMB_MODEL).encode()).hexdigest()[:12]
    p = EMB / f"{tag}_{key}.npy"
    if p.exists():
        return np.load(p)
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(EMB_PATH, device=device)
    E = m.encode(list(texts), batch_size=128, normalize_embeddings=True, show_progress_bar=True,
                 convert_to_numpy=True).astype(np.float32)
    np.save(p, E)
    return E


def overlaps(C, E, emb_th, device):
    """Per pool prompt: flags exact / contain / emb / bleu and the eval sets it matched. Also returns the topic-text
    embedding of every pool prompt (redteam_query when there is one, else the prompt)."""
    n = len(C)
    flags = {k: np.zeros(n, bool) for k in ("exact", "contain", "emb")}
    hits = [set() for _ in range(n)]
    emb_best = np.zeros(n, np.float32)
    emb_match = [""] * n
    en = E.prompt.map(norm).tolist()
    names = E.eval_set.tolist()

    # exact (also on JailbreakV's underlying redteam question)
    idx = defaultdict(set)
    for s, nm in zip(en, names):
        idx[s].add(nm)
    bases = C.base.fillna("").map(norm).tolist()
    for i, (s, b) in enumerate(zip(C.n, bases)):
        m = idx.get(s, set()) | (idx.get(b, set()) if b else set())
        if m:
            flags["exact"][i] = True
            hits[i] |= m

    # containment, both directions, via a 6-word prefix index of the shorter side
    def prefix_index(texts):
        ix = defaultdict(list)
        for j, s in enumerate(texts):
            t = s.split()
            if len(t) >= 6:
                ix[" ".join(t[:6])].append(j)
        return ix

    def scan(long_texts, short_texts, short_ix, on_hit):
        for i, s in enumerate(long_texts):
            t = s.split()
            padded = " " + s + " "
            for k in range(len(t) - 5):
                for j in short_ix.get(" ".join(t[k:k + 6]), ()):
                    if " " + short_texts[j] + " " in padded:
                        on_hit(i, j)

    cn = C.n.tolist()

    def hit_eval_in_train(i, j):
        flags["contain"][i] = True
        hits[i].add(names[j])

    def hit_train_in_eval(j, i):
        flags["contain"][i] = True
        hits[i].add(names[j])

    scan(cn, en, prefix_index(en), hit_eval_in_train)
    scan(en, cn, prefix_index(cn), hit_train_in_eval)

    # embedding near-duplicates
    Ev = embed(E.prompt.tolist(), "eval", device)
    Cv = embed(C.prompt.tolist(), "fit_pool", device)
    has_base = (C.base.fillna("").str.len() > 0).values
    Bv = embed(C.base[has_base].tolist(), "fit_pool_base", device) if has_base.any() else None
    base_row = {r: k for k, r in enumerate(np.where(has_base)[0])}
    names_arr = np.array(names)
    for a in range(0, n, 2048):
        S = Cv[a:a + 2048] @ Ev.T
        for r, i in enumerate(range(a, min(a + 2048, n))):
            s = S[r]
            if i in base_row:
                s = np.maximum(s, Bv[base_row[i]] @ Ev.T)
            j = int(s.argmax())
            emb_best[i], emb_match[i] = s[j], f"{names[j]}: {E.prompt[j][:160]}"
            if s[j] >= emb_th:
                flags["emb"][i] = True
                hits[i] |= set(names_arr[s >= emb_th])

    # BLEU near-copies (shared jailbreak templates, AdvBench-style rephrasings the embedding misses)
    import sacrebleu
    from sklearn.feature_extraction.text import TfidfVectorizer
    flags["bleu"] = np.zeros(n, bool)
    texts = C.prompt.tolist()
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1).fit(texts + E.prompt.tolist())
    T = vec.transform(texts).T.tocsc()
    Q = vec.transform(E.prompt)
    active = np.arange(len(E))
    # repeat on the eval prompts that hit something, flagged prompts masked out: a template copied hundreds of times
    # can fill the whole top-k of one eval prompt
    while len(active):
        found = set()
        for a in range(0, len(active), 500):
            rows = active[a:a + 500]
            S = (Q[rows] @ T).toarray()
            S[:, flags["bleu"]] = -1
            top = np.argpartition(-S, BLEU_K, axis=1)[:, :BLEU_K]
            for r, e in enumerate(rows):
                hyp = E.prompt.iloc[e]
                for i in top[r]:
                    if S[r, i] > 0 and sacrebleu.sentence_bleu(hyp, [texts[i]]).score / 100 >= BLEU_TH:
                        flags["bleu"][i] = True
                        hits[i].add(names[e])
                        found.add(e)
        active = np.array(sorted(found), dtype=int)

    out = pd.DataFrame(flags)
    out["any"] = out.any(axis=1)
    out["eval_sets"] = [",".join(sorted(h)) for h in hits]
    out["emb_max"] = emb_best
    out["emb_nearest"] = emb_match
    X = Cv.copy()
    for r, k in base_row.items():
        X[r] = Bv[k]
    return out, X


# ---------------------------------------------------------------- BLEU (AlphaSteer, App. D.1)
def bleu_report(train_texts, E, k=20):
    """For every eval prompt: max sentence-BLEU (sacrebleu, 0-1) against its k nearest train prompts by word TF-IDF.
    Per eval set: mean, share of eval prompts with BLEU < 0.3, max."""
    import sacrebleu
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1).fit(list(train_texts) + E.prompt.tolist())
    T = vec.transform(train_texts)
    per = np.zeros(len(E))
    for a in range(0, len(E), 2000):
        S = (vec.transform(E.prompt[a:a + 2000]) @ T.T).toarray()
        top = np.argsort(-S, axis=1)[:, :k]
        for r in range(S.shape[0]):
            hyp = E.prompt.iloc[a + r]
            per[a + r] = max(sacrebleu.sentence_bleu(hyp, [train_texts[j]]).score for j in top[r]) / 100
    E = E.assign(bleu=per)

    def stats(g):
        return dict(n=len(g), mean=round(float(g.bleu.mean()), 4), frac_lt_0_3=round(float((g.bleu < 0.3).mean()), 4),
                    frac_ge_0_5=round(float((g.bleu >= 0.5).mean()), 4), max=round(float(g.bleu.max()), 4))
    res = {name: stats(g) for name, g in E.groupby("eval_set")}
    for grp, sets in EVAL_GROUPS.items():
        res[f"[{grp}]"] = stats(E[E.eval_set.isin(sets)])
    res["[all]"] = stats(E)
    return res


# ---------------------------------------------------------------- quotas and split
def waterfill(avail, target):
    """Share `target` equally over sources; a source short of its share passes the rest on."""
    take = {s: 0 for s in avail}
    left = target
    while left > 0:
        act = sorted((s for s in avail if avail[s] > take[s]), key=lambda s: avail[s] - take[s])
        if not act:
            break
        share = max(1, left // len(act))
        for s in act:
            g = min(share, avail[s] - take[s], left)
            take[s] += g
            left -= g
            if left == 0:
                break
    return take


def spread(g, rng):
    """Order a (source, topic) cell so that a head of any length is spread over the source's sub-labels; JailbreakV
    takes first variants of every question before second ones and keeps <= JBV_MAX_VARIANTS."""
    g = g.sample(frac=1, random_state=rng)
    if g.empty:
        return g
    src = g.source.iat[0]
    if src == "jailbreakv":
        g = g.assign(_v=g.groupby("base").cumcount(), _f=g.groupby(g.source_label.str.split("|").str[-1]).cumcount())
        g = g[g._v < JBV_MAX_VARIANTS].sort_values(["_v", "_f"], kind="stable")
    elif src == "do_not_answer" or src.startswith("wgm_"):
        key = g.source_label.str.split("|").str[-1]
        g = g.assign(_f=g.groupby(key).cumcount()).sort_values("_f", kind="stable")
    return g.drop(columns=[c for c in ("_v", "_f") if c in g])


def level_fill(avail, have, budget):
    """Give `budget` prompts to topics so that the totals have + take are as even as possible (lowest first)."""
    take = {t: 0 for t in avail}
    for _ in range(budget):
        open_ = [t for t in avail if take[t] < avail[t]]
        if not open_:
            break
        t = min(open_, key=lambda t: (have.get(t, 0) + take[t], t))
        take[t] += 1
    return take


def draw(pool, src, take, rng):
    """take[topic] prompts of `src`, spread over its sub-labels"""
    parts = [spread(pool[(pool.source == src) & (pool.topic == t)], rng).head(k) for t, k in take.items() if k]
    return pd.concat(parts) if parts else pool.iloc[:0]


def harmful_pool(clean, set_, rng):
    """Curated sources first (all, or CURATED_TAKE spread evenly over topics), the filler evens out the topics."""
    pool = clean[(clean.set == set_) & clean.topic.isin(TOPICS)]
    parts = []
    for src in CURATED[set_]:
        cell = pool[pool.source == src]
        n = CURATED_TAKE.get(src)
        if n is None:
            parts.append(cell if src != "jailbreakv" else spread(cell, rng))
        else:
            avail = {t: len(spread(cell[cell.topic == t], rng)) for t in TOPICS}
            parts.append(draw(pool, src, level_fill(avail, {}, n), rng))
    have = pd.concat(parts).topic.value_counts().to_dict()
    budget = POOL_SIZE[set_] - sum(have.values())
    filler = pool[pool.source == FILLER[set_]]
    avail = filler.topic.value_counts().reindex(TOPICS, fill_value=0).to_dict()
    parts.append(draw(pool, FILLER[set_], level_fill(avail, have, budget), rng))
    return pd.concat(parts)


def benign_pool(clean, rng):
    parts = []
    wgm = clean[(clean.set == "benign") & clean.source.str.startswith("wgm_") & clean.topic.isin(TOPICS)]
    for t in TOPICS:
        cells = {s: spread(wgm[(wgm.source == s) & (wgm.topic == t)], rng) for s in ("wgm_benign_vanilla", "wgm_benign_adv")}
        take = waterfill({s: len(c) for s, c in cells.items()}, WGM_BENIGN_PER_TOPIC)
        parts += [cells[s].head(k) for s, k in take.items() if k]
    for src, n in BENIGN_FIXED.items():
        g = clean[(clean.set == "benign") & (clean.source == src)]
        if src == "coconot":            # AlphaSteer: all of pref + the rest from original
            pref = g[g.prompt.isin(set(raw("coconot_pref").prompt.str.strip()))]
            rest = g.drop(pref.index)
            g = pd.concat([pref, rest.sample(n=min(len(rest), n - len(pref)), random_state=rng)])
        else:
            g = g.sample(n=min(len(g), n), random_state=rng)
        parts.append(g)
    return pd.concat(parts)


def pool_groups(X, base, th):
    """Workflow 1 groups (the first Colab run): connected components of cosine >= th, plus equal non-empty base text.
    Group id = the smallest member index."""
    n = len(X)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for a in range(0, n, 2048):
        S = X[a:a + 2048] @ X.T
        for r, j in zip(*np.nonzero(S >= th)):
            if a + r < j:
                union(a + r, j)
    first = {}
    for i, b in enumerate(base):
        if isinstance(b, str) and b:
            union(i, first.setdefault(b, i))
    return np.array([find(i) for i in range(n)])


def behaviour_groups(X, base, th):
    """Leader clustering: the prompt with the most neighbours at cosine >= th leads a group of all its ungrouped
    neighbours, then the next, ... Every member is within th of its leader, so groups do not chain (connected
    components at 0.7 merge unrelated prompts into groups of 1000+). Equal non-empty base texts are then merged."""
    n = len(X)
    nb = []
    for a in range(0, n, 2048):
        nb += [np.nonzero(row >= th)[0] for row in X[a:a + 2048] @ X.T]
    lead = np.full(n, -1)
    for i in sorted(range(n), key=lambda i: (-len(nb[i]), i)):
        if lead[i] < 0:
            m = nb[i][lead[nb[i]] < 0]
            lead[m] = i
            lead[i] = i
    parent = lead.copy()

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)

    first = {}
    for i, b in enumerate(base):
        if isinstance(b, str) and b:
            union(i, first.setdefault(b, i))
    return np.array([find(i) for i in range(n)])


def md_table(rows, cols):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def composition_md(final, topic_info):
    order = TOPICS + ["none"]
    md = ["# SElOT candidate pools: composition", "",
          "Model-independent pools after dedup (steps i-iii). The label stage keeps, per model, what the model refuses "
          "(D_r) or answers (D_j, D_b) and caps D_r / D_j per topic within each split. The train/val split is fixed "
          "here, by behaviour group, for every model.", "",
          f"Topic classifier: LogReg on MiniLM, {topic_info['clf_train']} native/rule labels, 5-fold CV accuracy "
          f"{topic_info['clf_cv_acc']}; WildGuardMix subcategory vs rule agreement {topic_info['wgm_vs_rule_agree']}.", ""]
    for s in ["refusal", "jailbreak", "benign"]:
        g = final[final.set == s]
        ct = pd.crosstab(g.topic, g.source)
        ct = ct.reindex([t for t in order if t in ct.index])
        ct["total"] = ct.sum(axis=1)
        tot = ct.sum().to_frame().T
        tot.index = ["**total**"]
        share = (100 * tot / len(g)).round(0).astype(int).astype(str) + "%"
        share.index = ["**share**"]
        md += [f"## {SET_TITLE[s]} pool: {len(g)} prompts, {g.group.nunique()} behaviour groups, "
               f"val {int((g.split == 'val').sum())} ({100 * (g.split == 'val').mean():.1f}%)", "",
               pd.concat([ct.astype(str), tot.astype(str), share]).to_markdown(), "",
               "topic_method: " + ", ".join(f"{k} {v}" for k, v in g.topic_method.value_counts().items()), ""]
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-th", type=float, default=0.85)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--min-p", type=float, default=0.5, help="topic classifier threshold")
    ap.add_argument("--split-th", type=float, default=SPLIT_TH, help="behaviour-group cosine (step iv)")
    ap.add_argument("--val-folds", type=int, default=VAL_FOLDS)
    ap.add_argument("--skip-bleu", action="store_true")
    ap.add_argument("--no-dedup", action="store_true", help="skip step (ii): keep pool prompts that overlap eval")
    ap.add_argument("--out", default=None, help="write <out>/fit and <out>/reports instead of data/fit, data/reports")
    a = ap.parse_args()
    global FIT, REP
    if a.out:
        FIT, REP = Path(a.out) / "fit", Path(a.out) / "reports"
    elif a.no_dedup:
        raise SystemExit("--no-dedup needs --out (data/fit must stay deduplicated)")
    rng = np.random.RandomState(SEED)
    for d in (FIT, REP):
        d.mkdir(parents=True, exist_ok=True)

    # (i) raw pools, (ii) dedup against eval
    E = load_eval()      # the eval files with their metric fields are written by prepare_eval.py
    C, n_intra = load_pools(rng)
    print(f"eval prompts {len(E)}, pool prompts {len(C)} ({n_intra} exact duplicates across pools dropped)", flush=True)
    F, X = overlaps(C, E, a.emb_th, a.device)
    C = pd.concat([C, F], axis=1)

    stat = []
    for (s, src), g in C.groupby(["set", "source"], sort=False):
        stat.append(dict(set=s, source=src, pool=len(g), exact=int(g.exact.sum()), contain=int(g.contain.sum()),
                         emb=int(g.emb.sum()), bleu=int(g.bleu.sum()), removed=int(g["any"].sum()),
                         removed_pct=f"{100 * g['any'].mean():.2f}%", clean=int((~g["any"]).sum())))
    by_eval = defaultdict(lambda: defaultdict(int))
    for s, sets in zip(C.set, C.eval_sets):
        for e in filter(None, sets.split(",")):
            by_eval[e][s] += 1
    with open(REP / "removed.jsonl", "w") as f:
        for r in C[C["any"]].itertuples():
            f.write(json.dumps(dict(set=r.set, source=r.source, prompt=r.prompt, eval_sets=r.eval_sets,
                                    exact=bool(r.exact), contain=bool(r.contain), emb=bool(r.emb), bleu=bool(r.bleu),
                                    emb_max=round(float(r.emb_max), 3), nearest=r.emb_nearest), ensure_ascii=False) + "\n")

    # (iii) topics on the clean harmful pool and the clean WildGuardMix benign pool
    keep = np.ones(len(C), bool) if a.no_dedup else ~C["any"].values
    C, X = C[keep].reset_index(drop=True), X[keep]
    C["text"] = [b if isinstance(b, str) and b else p for p, b in zip(C.prompt, C.base)]
    C["topic"], C["topic_method"] = None, None
    tm = ((C.set != "benign") | C.source.str.startswith("wgm_")).values
    topic, method, topic_info = label_topics(C[tm].reset_index(drop=True), X[tm], RAW, a.min_p)
    C.loc[tm, "topic"], C.loc[tm, "topic_method"] = topic, method
    pool_topics = pd.crosstab(C[tm].topic, C[tm].source)
    print("clean pool, topic x source:\n" + pool_topics.to_string(), flush=True)

    # pools: curated first, WildGuardMix evens out the topics
    final = pd.concat([harmful_pool(C, "refusal", rng), harmful_pool(C, "jailbreak", rng), benign_pool(C, rng)])
    final["topic"] = final.topic.fillna("none")
    final["topic_method"] = final.topic_method.fillna("none")
    Xf = X[final.index.values]
    final = final.reset_index(drop=True)

    # workflow 1 groups: within each pool, on the dedup embeddings (as the first Colab run)
    final["group_v1"] = ""
    for s in ("refusal", "jailbreak", "benign"):
        m = (final.set == s).values
        final.loc[m, "group_v1"] = [f"{s}_g{int(k)}" for k in pool_groups(Xf[m], final[m].base.tolist(), a.emb_th)]

    # (iv) behaviour groups over the three pools together, then the fixed split
    Xt = embed(final.text.tolist(), "pooltext", a.device)
    final["group"] = behaviour_groups(Xt, final.base.tolist(), a.split_th)
    recs = [dict(group=f"g{int(g)}", stratum=f"benign:{src}" if s == "benign" else f"{s}:{t}")
            for g, s, src, t in zip(final.group, final.set, final.source, final.topic)]
    split_val(recs, a.val_folds, SEED, lambda r: r["stratum"])
    final["split"] = [r["split"] for r in recs]
    print("val share per set:", final.groupby("set").split.apply(lambda x: round((x == "val").mean(), 3)).to_dict(),
          flush=True)

    for s, g in final.groupby("set"):
        with open(FIT / f"{s}.jsonl", "w") as f:
            for i, r in enumerate(g.itertuples()):
                row = dict(id=f"{s}_{i:05d}", prompt=r.prompt, source=r.source, source_label=r.source_label,
                           topic=r.topic, topic_method=r.topic_method, group=f"g{int(r.group)}", split=r.split,
                           group_v1=r.group_v1)
                if isinstance(r.base, str) and r.base:
                    row["redteam_query"] = r.base
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    bleu = {}
    if not a.skip_bleu:
        mal = final[final.set != "benign"].prompt.tolist()
        bleu["malicious"] = bleu_report(mal, E)
        bleu["benign"] = bleu_report(final[final.set == "benign"].prompt.tolist(), E)
        bleu["overlap_in_pools"] = {s: {k: int(g[k].sum()) for k in ("exact", "contain", "emb", "bleu", "any")}
                                    | {"n": len(g)} for s, g in final.groupby("set")}
        bleu["dedup"] = not a.no_dedup
        (REP / "bleu.json").write_text(json.dumps(bleu, indent=1))

    md = ["# SElOT fitting sets: dedup report", "",
          f"Eval prompts: {len(E)} in {E.eval_set.nunique()} sets. Overlap = exact | contain (substring, >= 6 words) | "
          f"MiniLM cosine >= {a.emb_th} (JailbreakV also on its redteam_query) | sentence-BLEU >= {BLEU_TH}. Exact "
          f"duplicates across pools dropped first: {n_intra}. Dedup runs on the whole pool, before topics and quotas.",
          "", "## Removed per source", "",
          md_table(stat, ["set", "source", "pool", "exact", "contain", "emb", "bleu", "removed", "removed_pct", "clean"]),
          "", "## Removed pool prompts per eval set (a prompt can match several)", "",
          md_table([dict(eval_set=e, n_eval=int((E.eval_set == e).sum()), benign=v.get("benign", 0),
                         refusal=v.get("refusal", 0), jailbreak=v.get("jailbreak", 0))
                    for e, v in sorted(by_eval.items())], ["eval_set", "n_eval", "benign", "refusal", "jailbreak"]),
          "", "## Clean pool: topic x source", "", pool_topics.to_markdown()]
    for key in ("malicious", "benign"):
        if key not in bleu:
            continue
        md += ["", f"## BLEU: {key} pooled prompts vs eval prompts ({'after' if bleu['dedup'] else 'WITHOUT'} dedup)", "",
               md_table([dict(eval_set=k, **v) for k, v in bleu[key].items()],
                        ["eval_set", "n", "mean", "frac_lt_0_3", "frac_ge_0_5", "max"])]
    (REP / "dedup_report.md").write_text("\n".join(md) + "\n")
    comp = composition_md(final, topic_info)
    (REP / "composition.md").write_text(comp + "\n")
    print(comp)


if __name__ == "__main__":
    main()
