import json
import os
from collections import Counter, defaultdict

# Sentence-level CRE scores full triples with the "argument" counts, so FGT needs them
# per relation. Off unless the caller sets this, so existing logs keep their exact shape.
ARG_PER_TYPE = os.environ.get("ED_EVAL_ARG_PER_TYPE") == "1"

def normalize(text):
    return text.strip().lower()

def safe_load(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except:
            return {"events": []}
    return x

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
                if isinstance(arg_text, (list, dict)):  # unhashable: set() below would raise
                    continue
                args.append((trigger, e_type, arg_text, role))
    except:
        pass

    return list(set(args))

def extract_entities(data):
    # CRE entity mentions: the subject span (event[0]) and every object span (event[2][j][0]),
    # normalized, without relation or role. In event extraction event[0] is a trigger, so there
    # this mixes triggers and arguments. Each event is checked on its own, so a malformed one
    # is skipped without dropping the events after it.
    entities = set()
    events = data.get("events", []) if isinstance(data, dict) else []
    if not isinstance(events, list):
        return []
    for event in events:
        if not isinstance(event, list) or not event:
            continue
        if isinstance(event[0], str):
            entities.add(normalize(event[0]))
        if len(event) > 2 and isinstance(event[2], list):
            for arg in event[2]:
                if isinstance(arg, list) and arg and isinstance(arg[0], str):
                    entities.add(normalize(arg[0]))
    entities.discard("")
    return list(entities)

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
    entity_counts = {"tp": 0, "fp": 0, "fn": 0}
    
    trigger_type_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    argument_type_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

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
        update_counts(extract_entities(pred), extract_entities(gt), entity_counts)

        all_types_in_doc = set([t[1] for t in pred_triggers] + [t[1] for t in gt_triggers])
        
        for e_type in all_types_in_doc:
            pred_triggers_by_type = [t for t in pred_triggers if t[1] == e_type]
            gt_triggers_by_type = [t for t in gt_triggers if t[1] == e_type]
            
            update_counts(pred_triggers_by_type, gt_triggers_by_type, trigger_type_counts[e_type])

        if ARG_PER_TYPE:
            for e_type in {a[1] for a in pred_args} | {a[1] for a in gt_args}:
                update_counts([a for a in pred_args if a[1] == e_type],
                              [a for a in gt_args if a[1] == e_type], argument_type_counts[e_type])

    trigger_metrics = compute_f1(trigger_counts)
    argument_metrics = compute_f1(argument_counts)
    trigger_text_metrics = compute_f1(trigger_text_counts)
    entity_metrics = compute_f1(entity_counts)

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

    result = {
        "trigger_counts": trigger_counts,
        "argument_counts": argument_counts,
        "entity_counts": entity_counts,
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
        "entity": {
            "precision": entity_metrics[0],
            "recall": entity_metrics[1],
            "f1": entity_metrics[2],
        },
        "trigger_per_type": trigger_per_type_metrics,
    }
    if ARG_PER_TYPE:
        result["argument_per_type"] = {}
        for e_type, counts in argument_type_counts.items():
            if e_type in global_gt_types:
                precision, recall, f1 = compute_f1(counts)
                result["argument_per_type"][e_type] = {
                    "counts": dict(counts), "precision": precision, "recall": recall, "f1": f1}
    return result