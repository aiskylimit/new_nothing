"""Harm topics (taxonomy.yaml) for the fitting pools: one topic per prompt. Used by build_fit_sets.py.

The topic text is the prompt, or for JailbreakV its redteam_query (the plain question inside the jailbreak).
Order of evidence, first that applies wins (column `topic_method`):

  plain prompts (vanilla, benchmarks, JailbreakV via redteam_query)
    native   curated source label (Do-Not-Answer, MaliciousInstruct, JBB, JailbreakV policy), not `refine`
    rule     keyword rules (RULES); the topic with the most distinct hits wins, ties go to the earlier rule
    wgm      WildGuardMix subcategory, not `refine`
    clf      logistic regression on MiniLM embeddings of the topic text, trained on native + rule labels, p >= min_p
    default  the `default` of a refine label
  WildGuardMix adversarial prompts (a long wrapper: role play and story words fire the rules)
    wgm > clf > rule > default
Harmful prompts with no evidence get `uncertain`, benign ones `none`.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
TAX = yaml.safe_load(open(ROOT / "taxonomy.yaml"))
TOPICS = list(TAX["topics"])
MAPS = TAX["mappings"]
WRAPPED = {"wgm_harmful_adv", "wgm_benign_adv"}   # sources whose prompt text is a jailbreak-style wrapper

RULES = [  # (topic, regex); keep them specific
    ("weapons_cbrn", r"\b(bombs?|explosives?|explosive device|detonat\w*|grenades?|ied|tnt|c-?4|napalm|molotov|pipe bomb|"
                     r"(fire)?arms? (dealer|trafficking)|guns?|firearms?|rifles?|pistols?|handguns?|ammunition|ammo|"
                     r"silencer|suppressor|ghost gun|weapons?|weaponi[sz]\w*|nerve agent|sarin|vx|mustard gas|"
                     r"chlorine gas|anthrax|ricin|botulinum|bioweapons?|biological weapons?|chemical weapons?|"
                     r"nuclear (weapon|bomb|device)s?|enriched uranium|plutonium|dirty bomb|"
                     r"(engineer|create|synthesi[sz]e|spread|release) (a |the )?(deadly |lethal )?(virus|pathogen|toxin)\w*|"
                     r"(deadly|lethal) (virus|pathogen|toxin)\w*)\b"),
    ("cyber", r"\b(malware|ransomware|(computer|software) virus(es)?|trojan horse|trojans?|keylogger|spyware|botnet|ddos|"
              r"denial[- ]of[- ]service|phishing|hack(s|ed|ing|er|ers)?|exploit(s|ing)? (a |the )?vulnerabilit\w*|"
              r"sql injection|xss|backdoor|rootkit|brute[- ]force|crack (a |the )?password|password cracking|"
              r"zero[- ]day|bypass (the )?(firewall|antivirus|authentication)|cyber ?attacks?)\b"),
    ("self_harm_unsafe", r"\b(suicid\w*|kill (myself|yourself|themselves)|self[- ]harm\w*|cut(ting)? myself|"
                         r"anorexi\w*|bulimi\w*|eating disorder|starv(e|ing) (myself|yourself)|overdose)\b"),
    ("sexual", r"\b(porn\w*|sexually explicit|explicit sex\w*|erotic\w*|nude|nudes|sex scene|sexual (content|story|"
               r"fantas\w*|acts?)|child sexual|csam|prostitut\w*|escort service|sex trafficking)\b"),
    ("fraud_economic", r"\b(scam\w*|ponzi|pyramid scheme|counterfeit (money|currency|bills)|launder\w*|tax evasion|"
                       r"evade taxes|insider trading|pump[- ]and[- ]dump|embezzl\w*|credit card fraud|"
                       r"identity theft|steal\w* (someone's |somebody's |a person's |their |people's )?identit\w*|"
                       r"fraud\w*|extort\w*|blackmail\w*|fake (charity|invoices?|reviews))\b"),
    ("illegal_crime", r"\b(meth(amphetamine)?|cocaine|heroin|fentanyl|lsd|mdma|ecstasy|opioids?|narcotics?|"
                      r"illegal drugs?|drug (deal\w*|traffick\w*|smuggl\w*)|smuggl\w*|human trafficking|traffick\w*|"
                      r"shoplift\w*|burglar\w*|break into (a |someone's )?(house|home|car)|steal(ing)? (a )?car|"
                      r"pick a lock|lockpick\w*|forge(d|ry)? (documents?|signatures?)|fake (id|passport)s?)\b"),
    ("copyright", r"\b(lyrics|copyrighted|pirat(e|ed|ing)|torrent\w*|drm|full text of|entire (text|script|book))\b"),
    ("misinfo", r"\b(fake news|disinformation|misinformation|propaganda|conspiracy theor\w*|false (information|"
                r"claims?|rumou?rs?)|spread (false|rumou?rs)|hoax|defam\w*|manipulate (public )?opinion)\b"),
    ("privacy", r"\b(dox\w*|home address|social security number|ssn|personal (information|data|details) of|"
                r"medical records? of|track (someone|my (wife|husband|partner|ex))|spy on|stalk\w*|surveil\w*|"
                r"phone number of|private (information|messages) of|leak (the )?confidential)\b"),
    ("hate_harassment", r"\b(racis\w*|slurs?|stereotyp\w*|discriminat\w*|hate speech|inferior|bully\w*|harass\w*|"
                        r"insult\w*|derogatory|bigot\w*|xenophob\w*|homophob\w*|sexis\w*|antisemit\w*|white supremac\w*)\b"),
    ("violence", r"\b(kill(ing)?|murder\w*|assault\w*|kidnap\w*|tortur\w*|terroris\w*|terror attack|mass shooting|"
                 r"shoot(ing)? (someone|people|up)|stab(s|bed|bing)?|beat (someone|him|her) up|hurt (someone|people)|"
                 r"poison(ing)? (someone|my|a person)|animal cruelty|dog ?fight\w*|massacre|genocide)\b"),
]
RULES = [(t, re.compile(p, re.I)) for t, p in RULES]


def ok(x):
    """a usable label: pandas turns None into NaN, which is truthy"""
    return isinstance(x, str) and bool(x)


def mapped(mapping, label):
    """taxonomy entry -> (topic, refine)"""
    v = mapping.get(label)
    if v is None:
        return None, False
    return (v["default"], bool(v.get("refine"))) if isinstance(v, dict) else (v, False)


def rule_topic(text):
    """topic with the most distinct keyword hits; ties go to the earlier rule"""
    best, n_best = None, 0
    for t, rx in RULES:
        n = len({m.group(0).lower() for m in rx.finditer(text)})
        if n > n_best:
            best, n_best = t, n
    return best


class NativeLabels:
    def __init__(self, raw_dir):
        self.dna_keys = sorted(MAPS["do_not_answer"], key=len, reverse=True)
        self.mi = {p.strip(): i for i, p in enumerate(pd.read_parquet(Path(raw_dir) / "mi.parquet").prompt)}

    def __call__(self, source, label, prompt):
        """(topic, refine, kind); kind = native for curated sources, wgm for WildGuardMix"""
        label = label or ""
        if source == "do_not_answer":
            spec = label.split("|")[-1].strip().lower()
            k = next((k for k in self.dna_keys if spec.startswith(k.lower())), None)
            return (*mapped(MAPS["do_not_answer"], k), "native") if k else (None, False, "native")
        if source == "malicious_instruct":
            i = self.mi.get(prompt.strip())
            return (MAPS["malicious_instruct"][i // 10], False, "native") if i is not None else (None, False, "native")
        if source == "jbb_harmful":
            return (*mapped(MAPS["jbb"], label), "native")
        if source == "jailbreakv":
            return (*mapped(MAPS["jailbreakv"], label.split("|")[0]), "native")
        if source.startswith("wgm_harmful"):
            return (*mapped(MAPS["wildguardmix"], label), "wgm")
        return None, False, None


def label_topics(D, X, raw_dir, min_p=0.5):
    """D: rows with set, source, source_label, prompt, text (topic text); X: embeddings of `text` (normalised).
    Returns (topic, topic_method, info)."""
    native = NativeLabels(raw_dir)
    nat = [native(s, l, p) for s, l, p in zip(D.source, D.source_label, D.prompt)]
    nat_topic, nat_refine, nat_kind = map(np.array, zip(*nat))
    rule = np.array([rule_topic(t) for t in D.text], dtype=object)
    wrapped = D.source.isin(WRAPPED).values

    # classifier: curated native labels + rule hits on plain text
    y = np.array([t if (k == "native" and not rf and t in TOPICS) else (r if ok(r) and not w else None)
                  for t, rf, k, r, w in zip(nat_topic, nat_refine, nat_kind, rule, wrapped)], dtype=object)
    lab = np.array([v is not None for v in y])
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    clf = LogisticRegression(C=2.0, max_iter=3000, class_weight="balanced")
    yl = y[lab].astype(str)
    cvp = cross_val_predict(clf, X[lab], yl, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    clf.fit(X[lab], yl)
    P = clf.predict_proba(X)
    clf_t, clf_p = clf.classes_[P.argmax(1)], P.max(1)

    topic, method = [], []
    for i in range(len(D)):
        harmful = D.set.iat[i] != "benign"
        cands = {
            "native": nat_topic[i] if nat_kind[i] == "native" and ok(nat_topic[i]) and not nat_refine[i] else None,
            "rule": rule[i],
            "wgm": nat_topic[i] if nat_kind[i] == "wgm" and ok(nat_topic[i]) and not nat_refine[i] else None,
            "clf": clf_t[i] if clf_p[i] >= min_p else None,
            "default": nat_topic[i] if ok(nat_topic[i]) else None,
        }
        order = ["native", "wgm", "clf", "rule", "default"] if wrapped[i] else ["native", "rule", "wgm", "clf", "default"]
        m = next((m for m in order if ok(cands[m])), None)
        topic.append(cands[m] if m else ("uncertain" if harmful else "none"))
        method.append(m or "none")

    plain_wgm = (nat_kind == "wgm") & np.array([ok(t) for t in nat_topic]) & ~nat_refine.astype(bool) & ~wrapped
    both = plain_wgm & np.array([ok(r) for r in rule])
    info = {"clf_train": int(lab.sum()), "clf_cv_acc": round(float((cvp == yl).mean()), 3),
            "wgm_vs_rule_agree": round(float((nat_topic[both] == rule[both]).mean()), 3) if both.any() else None}
    return np.array(topic), np.array(method), info
