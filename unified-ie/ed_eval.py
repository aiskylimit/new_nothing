import csv
import json
from collections import Counter, defaultdict

def normalize(text):
    return text.strip().lower()

def safe_load(x):
    if isinstance(x, str):
        try:
            x = json.loads(x)
        except:
            return {"events": []}
    # valid JSON that is not an object (an answer like "[]" or "3") has nothing to score
    return x if isinstance(x, dict) else {}

def extract_triggers(data):
    triggers = []
    trigger_texts = []
    try:
        for event in data.get("events", []):
            # triggers.append((normalize(event["trigger_text"]), normalize(event["type"])))
            # trigger_texts.append(normalize(event["trigger_text"]))
            triggers.append((normalize(event[0]), normalize(event[1])))
            trigger_texts.append(normalize(event[0]))
    except:
        pass
    
    return list(set(triggers)), list(set(trigger_texts))

def extract_arguments(data):
    args = []
    try:
        for event in data.get("events", []):
            # trigger = normalize(event["trigger_text"])
            # e_type = normalize(event["type"])
            # for arg in event.get("arguments", []):
            #     arg_text = normalize(arg["text"])
            #     role = normalize(arg["role"])
            #     args.append((trigger, e_type, arg_text, role))
            trigger = normalize(event[0])
            e_type = normalize(event[1])
            for arg in event[2]:
                arg_text = arg[0]
                role = normalize(arg[1])
                args.append((trigger, e_type, arg_text, role))
    except:
        pass

    return list(set(args))

def update_counts(pred_items, gt_items, counts):
    pred_counter = Counter(pred_items)
    gt_counter = Counter(gt_items)

    tp = sum((pred_counter & gt_counter).values())
    fp = sum((pred_counter - gt_counter).values())
    fn = sum((gt_counter - pred_counter).values())

    counts["tp"] += tp
    counts["fp"] += fp
    counts["fn"] += fn

def compute_f1(counts):
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision > 0 and recall > 0 else 0

    return precision, recall, f1


# def ed_evaluate(pred_list, gt_list):
#     trigger_counts = {"tp": 0, "fp": 0, "fn": 0}
#     argument_counts = {"tp": 0, "fp": 0, "fn": 0}
#     trigger_text_counts = {"tp": 0, "fp": 0, "fn": 0}


#     for pred_json, gt_json in zip(pred_list, gt_list):
#         pred = safe_load(pred_json)
#         gt = safe_load(gt_json[0])

#         pred_triggers, pred_trigger_texts = extract_triggers(pred)
#         gt_triggers, gt_trigger_texts = extract_triggers(gt)

#         pred_args = extract_arguments(pred)
#         gt_args = extract_arguments(gt)

#         update_counts(pred_triggers, gt_triggers, trigger_counts)
#         update_counts(pred_args, gt_args, argument_counts)
#         update_counts(pred_trigger_texts, gt_trigger_texts, trigger_text_counts)


#     trigger_metrics = compute_f1(trigger_counts)
#     argument_metrics = compute_f1(argument_counts)
#     trigger_text_metrics = compute_f1(trigger_text_counts)

#     return {
#         "trigger_counts": trigger_counts,
#         "argument_counts": argument_counts,
#         "trigger_text": {
#             "precision": trigger_text_metrics[0],
#             "recall": trigger_text_metrics[1],
#             "f1": trigger_text_metrics[2],
#         },
#         "trigger": {
#             "precision": trigger_metrics[0],
#             "recall": trigger_metrics[1],
#             "f1": trigger_metrics[2],
#         },
#         "argument": {
#             "precision": argument_metrics[0],
#             "recall": argument_metrics[1],
#             "f1": argument_metrics[2],
#         }
#     }
def ed_evaluate(pred_list, gt_list):
    trigger_counts = {"tp": 0, "fp": 0, "fn": 0}
    argument_counts = {"tp": 0, "fp": 0, "fn": 0}
    trigger_text_counts = {"tp": 0, "fp": 0, "fn": 0}
    
    trigger_type_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    
    global_gt_types = set()

    for pred_json, gt_json in zip(pred_list, gt_list):
        pred = safe_load(pred_json)
        gt = safe_load(gt_json[0])

        pred_triggers, pred_trigger_texts = extract_triggers(pred)
        gt_triggers, gt_trigger_texts = extract_triggers(gt)

        for t in gt_triggers:
            global_gt_types.add(t[1])

        pred_args = extract_arguments(pred)
        gt_args = extract_arguments(gt)

        update_counts(pred_triggers, gt_triggers, trigger_counts)
        update_counts(pred_args, gt_args, argument_counts)
        update_counts(pred_trigger_texts, gt_trigger_texts, trigger_text_counts)

        all_types_in_doc = set([t[1] for t in pred_triggers] + [t[1] for t in gt_triggers])
        
        for e_type in all_types_in_doc:
            pred_triggers_by_type = [t for t in pred_triggers if t[1] == e_type]
            gt_triggers_by_type = [t for t in gt_triggers if t[1] == e_type]
            
            update_counts(pred_triggers_by_type, gt_triggers_by_type, trigger_type_counts[e_type])

    trigger_metrics = compute_f1(trigger_counts)
    argument_metrics = compute_f1(argument_counts)
    trigger_text_metrics = compute_f1(trigger_text_counts)

    trigger_per_type_metrics = {}
    
    for e_type, counts in trigger_type_counts.items():
        if e_type in global_gt_types:
            precision, recall, f1 = compute_f1(counts)
            trigger_per_type_metrics[e_type] = {
                "counts": dict(counts),
                "precision": precision,
                "recall": recall,
                "f1": f1
            }

    return {
        "trigger_counts": trigger_counts,
        "argument_counts": argument_counts,
        "trigger_text": {
            "precision": trigger_text_metrics[0],
            "recall": trigger_text_metrics[1],
            "f1": trigger_text_metrics[2],
        },
        "trigger": {
            "precision": trigger_metrics[0],
            "recall": trigger_metrics[1],
            "f1": trigger_metrics[2],
        },        
        "argument": {
            "precision": argument_metrics[0],
            "recall": argument_metrics[1],
            "f1": argument_metrics[2],
        },
        "trigger_per_type": trigger_per_type_metrics,
    }

# ---- per-dataset scoring for the merged label set (data/general_domain) ----

def load_label_map(path):
    """dataset -> {normalized merged label (v1 or v2 spelling): label in the dataset's own scheme}.
    `path` may list several maps, comma-separated (mt_* folders: ee_label_map.tsv,ner_label_map.tsv);
    they are keyed by dataset, and no dataset is in two of them."""
    label_map = defaultdict(dict)
    for p in path.split(","):
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                for col in ("label_v1", "label_v2"):
                    label_map[r["dataset"]][normalize(r[col])] = r["eval_label"]
    return label_map

def scheme_label(dataset, label):
    # RAMS is scored on its 38 level-2 types
    return ".".join(label.split(".")[:2]) if dataset == "rams" else label

def role_key(role):
    # v1 keeps the dataset's role names, v2 writes them in lowercase words
    return normalize(role).replace("_", " ")

def ed_evaluate_by_dataset(pred_list, rows, label_map):
    """Score every dataset in its own label scheme. A predicted label is mapped back through label_map;
    a label the row's dataset does not use is dropped, since that dataset never annotates it."""
    counts = defaultdict(lambda: {k: {"tp": 0, "fp": 0, "fn": 0} for k in ("trigger", "trigger_id", "argument")})
    has_args = set()
    for pred_json, row in zip(pred_list, rows):
        ds = row["dataset"]
        m = label_map[ds]
        gold_t, gold_a = set(), set()
        for g in row["gold_eval"]:
            t, label = normalize(g[0]), scheme_label(ds, g[1])
            gold_t.add((t, label))
            if len(g) > 2:
                has_args.add(ds)
                gold_a.update((t, label, normalize(a[0]), role_key(a[1])) for a in g[2])
        pred_t, pred_a = set(), set()
        for event in safe_load(pred_json).get("events", []):
            try:
                label = m.get(normalize(event[1]))
                if label is None:
                    continue
                t, label = normalize(event[0]), scheme_label(ds, label)
                pred_t.add((t, label))
                if len(event) > 2 and isinstance(event[2], list):
                    pred_a.update((t, label, normalize(a[0]), role_key(a[1])) for a in event[2])
            except (TypeError, IndexError, AttributeError):
                continue
        update_counts(list(pred_t), list(gold_t), counts[ds]["trigger"])
        update_counts(list({x[0] for x in pred_t}), list({x[0] for x in gold_t}), counts[ds]["trigger_id"])
        update_counts(list(pred_a), list(gold_a), counts[ds]["argument"])

    out = {}
    for ds, c in sorted(counts.items()):
        out[ds] = {}
        for k in ("trigger", "trigger_id", "argument"):
            if k == "argument" and ds not in has_args:
                continue
            p, r, f1 = compute_f1(c[k])
            out[ds][k] = {"precision": p, "recall": r, "f1": f1, "counts": dict(c[k])}
    out["macro_trigger_f1"] = sum(v["trigger"]["f1"] for v in out.values()) / max(1, len(counts))
    if has_args:
        out["macro_argument_f1"] = sum(out[ds]["argument"]["f1"] for ds in has_args) / len(has_args)
    return out

def re_evaluate_by_dataset(pred_list, rows):
    """Sentence-level relation extraction, per source dataset. Gold is the row's own response
    ({"relations": [[subject, relation, object], ...]}); RE labels are not merged, so no mapping.
    triple: (subject, relation, object) exact after lowercasing; pair: (subject, object) only.
    triple_gold_pair: triple, but a predicted triple whose (subject, object) is not a gold pair of the sentence
    is ignored instead of counted as a false positive. TACRED and FewRel label only one or a few pairs per
    sentence, so a correct triple on an unlabeled pair would otherwise count as wrong. Recall is the same as
    triple's; precision is an upper bound (made-up pairs are ignored too). Dev selection stays on triple."""
    counts = defaultdict(lambda: {k: {"tp": 0, "fp": 0, "fn": 0} for k in ("triple", "pair", "triple_gold_pair")})

    def triples(x):
        out = set()
        for z in safe_load(x).get("relations", []):
            try:
                out.add((normalize(z[0]), normalize(z[1]), normalize(z[2])))
            except (TypeError, IndexError, AttributeError):
                continue
        return out

    for pred_json, row in zip(pred_list, rows):
        pred, gold = triples(pred_json), triples(row["response"])
        c = counts[row["dataset"]]
        update_counts(list(pred), list(gold), c["triple"])
        gold_pairs = {(h, t) for h, _, t in gold}
        update_counts(list({(h, t) for h, _, t in pred}), list(gold_pairs), c["pair"])
        update_counts([x for x in pred if (x[0], x[2]) in gold_pairs], list(gold), c["triple_gold_pair"])
    out = {}
    for ds, c in sorted(counts.items()):
        out[ds] = {}
        for k in ("triple", "pair", "triple_gold_pair"):
            p, r, f1 = compute_f1(c[k])
            out[ds][k] = {"precision": p, "recall": r, "f1": f1, "counts": dict(c[k])}
    out["macro_triple_f1"] = sum(out[ds]["triple"]["f1"] for ds in counts) / max(1, len(counts))
    out["macro_triple_gold_pair_f1"] = sum(out[ds]["triple_gold_pair"]["f1"] for ds in counts) / max(1, len(counts))
    return out

def ner_evaluate_by_dataset(pred_list, rows, label_map):
    """NER per source dataset in its own labels ({"entities": [[span, type], ...]}). A predicted label the
    row's dataset does not use is dropped. Entities are matched as a multiset of (span, type) after
    lowercasing, so a name that occurs twice counts twice. span: the span alone (identification)."""
    counts = defaultdict(lambda: {k: {"tp": 0, "fp": 0, "fn": 0} for k in ("entity", "span")})
    for pred_json, row in zip(pred_list, rows):
        m = label_map[row["dataset"]]
        gold = [(normalize(g[0]), g[1]) for g in row["gold_eval"]]
        pred = []
        for e in safe_load(pred_json).get("entities", []):
            try:
                label = m.get(normalize(e[1]))
                if label is not None:
                    pred.append((normalize(e[0]), label))
            except (TypeError, IndexError, AttributeError):
                continue
        c = counts[row["dataset"]]
        update_counts(pred, gold, c["entity"])
        update_counts([p[0] for p in pred], [g[0] for g in gold], c["span"])
    out = {}
    for ds, c in sorted(counts.items()):
        out[ds] = {}
        for k in ("entity", "span"):
            p, r, f1 = compute_f1(c[k])
            out[ds][k] = {"precision": p, "recall": r, "f1": f1, "counts": dict(c[k])}
    out["macro_entity_f1"] = sum(v["entity"]["f1"] for v in out.values()) / max(1, len(counts))
    return out

def format_by_dataset(by_dataset, split, epoch):
    """One log line per dataset plus the macro averages, for reading results straight off log.txt."""
    lines = []
    for ds, m in by_dataset.items():
        if not isinstance(m, dict):
            continue
        parts = [f"{k} P {100 * v['precision']:.2f} R {100 * v['recall']:.2f} F1 {100 * v['f1']:.2f}"
                 for k, v in m.items()]
        lines.append(f"RESULT {split} epoch {epoch} | {ds:7s} | " + " | ".join(parts))
    macro = " | ".join(f"{k[len('macro_'):].replace('_', ' ')} {100 * v:.2f}"
                       for k, v in by_dataset.items() if k.startswith("macro_"))
    return "\n".join(lines + [f"RESULT {split} epoch {epoch} | macro   | {macro}"])
