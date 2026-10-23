"""Build the NER data in data/general_domain from CoNLL-2003 and OntoNotes 5.0 (tner/conll2003, tner/ontonotes5).

Labels are merged only where both datasets mean the same thing, checked on the entity strings the two
datasets share (how OntoNotes labels CoNLL's strings and the other way round):
    PER <-> PERSON   93% / 93%   merged
    LOC  -> GPE 95% (CoNLL LOC covers countries and cities; OntoNotes LOC is natural locations only)
    ORG  -> ORG 49%, GPE 44% (CoNLL tags countries playing sport as ORG)
    MISC -> NORP 83%, EVENT 6%, ORG 5%, LANGUAGE 4%
so 4 + 18 - 1 = 21 labels. Labels that are not merged get names that say what they cover, so no two
labels share a name.

    ner_v1: original codes (CoNLL's broader LOC/ORG written LOCATION/ORGANIZATION)
    ner_v2: lowercase words

Rows follow data/general_domain/ee_*: system_prompt / user_prompt / response plus "dataset", "task" ("ner")
and "gold_eval" ([[span, label in the dataset's own scheme], ...]). The prompt names neither the dataset nor
the labels. Sentences without entities are kept: they are 20% (CoNLL) and 44% (OntoNotes) of the test sets.
The text is the tokens joined by spaces, so every span is an exact substring. An I- tag without a B- before it
(40 in OntoNotes) starts an entity. dev.jsonl keeps at most DEV_PER_DATASET rows per dataset, dev_full.jsonl all.

Usage (from the unified-ie dir):
    python tools/build_ner_data.py --raw-dir ../OpenED/data/ner_raw
"""
import argparse
import csv
import json
import os
import random

SHUFFLE_SEED = 2021
DEV_PER_DATASET = 300
FILES = {
    "conll03": {"train": ["train.json"], "dev": ["valid.json"], "test": ["test.json"]},
    "ontonotes": {"train": ["train00.json", "train01.json", "train02.json", "train03.json"],
                  "dev": ["valid.json"], "test": ["test.json"]},
}
RAW_DIR = {"conll03": "conll03", "ontonotes": "ontonotes5"}
# (dataset, original label) -> (v1, v2)
LABELS = {
    ("conll03", "PER"): ("PERSON", "person"),
    ("conll03", "LOC"): ("LOCATION", "location"),
    ("conll03", "ORG"): ("ORGANIZATION", "organization or team"),
    ("conll03", "MISC"): ("MISC", "miscellaneous"),
    ("ontonotes", "PERSON"): ("PERSON", "person"),
    ("ontonotes", "NORP"): ("NORP", "nationality or religious or political group"),
    ("ontonotes", "FAC"): ("FAC", "facility"),
    ("ontonotes", "ORG"): ("ORG", "organization"),
    ("ontonotes", "GPE"): ("GPE", "country or city or state"),
    ("ontonotes", "LOC"): ("LOC", "natural location"),
    ("ontonotes", "PRODUCT"): ("PRODUCT", "product"),
    ("ontonotes", "EVENT"): ("EVENT", "event"),
    ("ontonotes", "WORK_OF_ART"): ("WORK_OF_ART", "work of art"),
    ("ontonotes", "LAW"): ("LAW", "law"),
    ("ontonotes", "LANGUAGE"): ("LANGUAGE", "language"),
    ("ontonotes", "DATE"): ("DATE", "date"),
    ("ontonotes", "TIME"): ("TIME", "time"),
    ("ontonotes", "PERCENT"): ("PERCENT", "percent"),
    ("ontonotes", "MONEY"): ("MONEY", "money"),
    ("ontonotes", "QUANTITY"): ("QUANTITY", "quantity"),
    ("ontonotes", "ORDINAL"): ("ORDINAL", "ordinal"),
    ("ontonotes", "CARDINAL"): ("CARDINAL", "cardinal number"),
}

NER_SYSTEM = (
    "You are a named entity recognition system. Your task is to identify named entities mentioned "
    "in a given text.\n"
    "IMPORTANT:Output ONLY valid JSON. No explanations, no markdown, no extra text.\n"
    "Output Format (JSON only, no markdown):\n\n"
    '{"entities": [[<entity span>, <entity type>], [<entity span 2>, <entity type 2>]]}\n\n'
    '- If no entities are detected, return: {"entities": []}'
)
NER_USER = (
    "Given an input text: \n<input>\n{text}\n</input>\n\n"
    "Your task is to extract all named entities present in the text. "
    "The text may contain zero, one, or multiple entities.\n\n"
    "For each entity:\n"
    "- Identify the entity span: the exact words that name it.\n"
    "- Identify the entity type.\n\n"
    "Constraints and Guidelines\n"
    "- Do not invent information not supported by the text.\n"
    "- Do not paraphrase entity spans.\n"
    "- Each entity span must exactly match a span in the original input (from <input>...</input>).\n"
)


def read_split(raw_dir, ds, split):
    id2label = {v: k for k, v in json.load(open(f"{raw_dir}/{RAW_DIR[ds]}/dataset/label.json")).items()}
    rows = []
    for fn in FILES[ds][split]:
        with open(f"{raw_dir}/{RAW_DIR[ds]}/dataset/{fn}", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                tokens, ents, cur = r["tokens"], [], None
                for i, t in enumerate(r["tags"]):
                    lab = id2label[t]
                    if lab.startswith("B-") or (lab.startswith("I-") and (cur is None or cur[0] != lab[2:])):
                        if cur:
                            ents.append(cur)
                        cur = [lab[2:], i, i + 1]
                    elif lab.startswith("I-"):
                        cur[2] = i + 1
                    else:
                        if cur:
                            ents.append(cur)
                        cur = None
                if cur:
                    ents.append(cur)
                rows.append({"text": " ".join(tokens), "entities": [[" ".join(tokens[s:e]), t] for t, s, e in ents]})
    return rows


def render(ds, row, version):
    ents = [[span, LABELS[(ds, t)][0 if version == "v1" else 1]] for span, t in row["entities"]]
    return {"system_prompt": NER_SYSTEM, "user_prompt": NER_USER.format(text=row["text"]),
            "response": json.dumps({"entities": ents}, ensure_ascii=False),
            "dataset": ds, "task": "ner", "gold_eval": row["entities"]}


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, help="holds conll03/dataset and ontonotes5/dataset from tner")
    ap.add_argument("--out-root", default="data/general_domain")
    args = ap.parse_args()

    data = {(ds, s): read_split(args.raw_dir, ds, s) for ds in FILES for s in ("train", "dev", "test")}
    for (ds, s), rows in data.items():
        print(f"{ds:9s} {s:5s} {len(rows):6d} sentences, {sum(len(r['entities']) for r in rows):6d} entities, "
              f"{sum(not r['entities'] for r in rows):5d} without entities")
    for version in ("v1", "v2"):
        out_dir = f"{args.out_root}/ner_{version}"
        os.makedirs(out_dir, exist_ok=True)
        for s in ("train", "dev", "test"):
            rows = [render(ds, r, version) for ds in FILES for r in data[(ds, s)]]
            random.Random(SHUFFLE_SEED).shuffle(rows)
            if s == "dev":
                write_jsonl(f"{out_dir}/dev_full.jsonl", rows)
                seen, kept = {}, []
                for r in rows:
                    seen[r["dataset"]] = seen.get(r["dataset"], 0) + 1
                    if seen[r["dataset"]] <= DEV_PER_DATASET:
                        kept.append(r)
                rows = kept
            print(f"{out_dir}/{s}.jsonl: {write_jsonl(f'{out_dir}/{s}.jsonl', rows)} rows")

    with open(f"{args.out_root}/ner_label_map.tsv", "w", encoding="utf-8", newline="\n") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["dataset", "original_label", "eval_label", "label_v1", "label_v2"])
        for (ds, lab), (v1, v2) in LABELS.items():
            w.writerow([ds, lab, lab, v1, v2])
    v1s = {v[0] for v in LABELS.values()}
    v2s = {v[1] for v in LABELS.values()}
    assert len(v1s) == len(v2s) == len(LABELS) - 1, "two labels share a name"
    print(f"=== {len(v2s)} NER labels -> {args.out_root}/ner_label_map.tsv")


if __name__ == "__main__":
    main()
