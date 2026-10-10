"""Teacher pseudo-labeling for CED (+ CL-DETR-style filtering).

The previous-task teacher runs inference on the CURRENT task's train inputs and
detects events of OLD types (the ones stripped from this task's gold responses
— measured on ACE: 18-24% of new-task sentences carry a stripped old-type
trigger). Detected old-type events are merged into the gold responses,
restoring the supervision the split removed.

Filters, in order:
- type must belong to streams[<task_id]
- trigger text must appear verbatim in the input sentence
- H2 (--conflict-dedup, default on): drop the event if its trigger overlaps a
  gold trigger of ANY type — never let a pseudo label contradict a gold label
  on the same token (CL-DETR drops pseudo boxes with IoU>0.7 against gold)
- exact (trigger, type) duplicates of gold are dropped
- lexicon filter: (trigger, type) must occur in old tasks' gold train data
- H1 (--conf-filter): teacher-confidence filter. Scores come free from
  generate() via compute_transition_scores — no extra forward. An event's
  score is the mean logprob of its trigger + type-name tokens (the decision
  tokens); "percentile" keeps the top p% events of the task, "thresh" applies
  an absolute mean-logprob cutoff.

--anchor pair (sentence-level CRE, default from $PL_ANCHOR, else trigger): a record
is anchored by its (subject, object) pair instead of its trigger/subject. The object
must also appear in the sentence, H2 drops a candidate only when that exact pair is
already labelled, and duplicates are (subject, relation, object). One subject usually
holds several relations, so trigger-style dedup would throw most old triples away.
Rows marked "is_memory" are skipped: their targets already hold every seen label.

Usage:
  python tools/ced_pseudo_label.py --teacher <merged_dir> --data-dir data/ace_b10_perm0/1 \
      --streams data/ace_b10_perm0/streams.json --task-id 1 --out data/r6_run/1 \
      [--conf-filter percentile --conf-percentile 70]
Writes train.jsonl (augmented) + dev/test copied unchanged, plus pl_stats.json.
"""
import argparse
import json
import os
import re
import shutil
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# the runners start this file as `python tools/ced_pseudo_label.py`, so sys.path[0] is tools/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gen_backend import find_vllm_python, meta_model, resolve_generation_config, run_vllm, vllm_params  # noqa: E402


def input_text_of(user_prompt):
    m = re.search(r"Given an input text: ?\n?(.*?)\n\nYour task", user_prompt, re.S)
    return m.group(1).strip() if m else None


def parse_events(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return []
    try:
        return json.loads(m.group(0)).get("events", [])
    except Exception:
        return []


def object_of(e):
    """Object span of a CRE record [subject, relation, [[object, "object"]], desc], else None."""
    if len(e) > 2 and isinstance(e[2], list) and e[2] and isinstance(e[2][0], list) \
            and e[2][0] and isinstance(e[2][0][0], str):
        return e[2][0][0]
    return None


def find_spans(text, values, search_start=0):
    """Char spans of each value in text, searched left-to-right."""
    spans = []
    pos = search_start
    for val in values:
        cs = text.find(val, pos)
        if cs == -1:
            cs = text.find(val)
        if cs == -1:
            spans.append(None)
        else:
            spans.append((cs, cs + len(val)))
            pos = cs + len(val)
    return spans


def event_conf_score(text, trig, ty, gen_ids, token_logprobs, tokenizer):
    """Mean logprob of the trigger + type-name tokens inside the generated text.

    Token positions are recovered by re-tokenizing the decoded text; if the
    re-tokenization disagrees with the actually generated ids (rare with clean
    greedy JSON), fall back to the mean logprob of the whole sequence.
    """
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids2 = enc["input_ids"]
    n = min(len(ids2), len(gen_ids))
    aligned = n > 0 and ids2[:n] == list(gen_ids[:n])

    valid = token_logprobs[:len(gen_ids)]
    if not aligned:
        return float(valid.mean()) if len(valid) else None

    offsets = enc["offset_mapping"]
    mask = [False] * n
    for span in find_spans(text, [trig, ty]):
        if span is None:
            continue
        cs, ce = span
        for j in range(n):
            s, e = offsets[j]
            if s < ce and e > cs and e > s:
                mask[j] = True
    idx = [j for j in range(min(n, len(valid))) if mask[j]]
    if not idx:
        return float(valid.mean()) if len(valid) else None
    return float(valid[idx].mean())


def chat_prompts(tokenizer, rows, idxs):
    """The teacher's prompt text for each candidate row."""
    return [tokenizer.apply_chat_template(
        [{"role": "system", "content": rows[i]["system_prompt"]}, {"role": "user", "content": rows[i]["user_prompt"]}],
        add_generation_prompt=True, tokenize=False, enable_thinking=False) for i in idxs]


def strip_stop(ids, tokenizer):
    """Generated ids without eos/pad: the view event_conf_score aligns with the text."""
    return [t for t in ids if t != tokenizer.eos_token_id and t != tokenizer.pad_token_id]


def generate_hf(model, tokenizer, prompts, max_new_tokens, need_scores, device):
    """[(text, gen_ids, token_logprobs)] from model.generate(); the ids and log-probabilities are
    None without the confidence filter."""
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id,
                             return_dict_in_generate=need_scores, output_scores=need_scores)
    if need_scores:
        sequences = out.sequences
        # [bs, gen_len] logprob of each generated token — no extra forward
        trans = model.compute_transition_scores(sequences, out.scores, normalize_logits=True).float().cpu()
    else:
        sequences = out
    gen_ids_batch = sequences[:, enc["input_ids"].shape[1]:].cpu()
    texts = tokenizer.batch_decode(gen_ids_batch, skip_special_tokens=True)
    if not need_scores:
        return [(text, None, None) for text in texts]
    return [(text, strip_stop(gen_ids_batch[pos].tolist(), tokenizer), trans[pos]) for pos, text in enumerate(texts)]


def generate_vllm(teacher, tokenizer, prompts, max_new_tokens, need_scores, work_dir, vllm_py):
    """generate_hf's results from vLLM, with the settings generate() would use (gen_backend.py)."""
    if not prompts:
        return []
    config = resolve_generation_config(meta_model(teacher), None, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    requests = [{"prompt_token_ids": ids, "max_tokens": max_new_tokens, "seed": 0}
                for ids in tokenizer(prompts, truncation=True, max_length=1024)["input_ids"]]
    outputs = run_vllm(work_dir, teacher, requests, vllm_params(config, logprobs=need_scores), vllm_py=vllm_py)
    shutil.rmtree(work_dir, ignore_errors=True)
    texts = tokenizer.batch_decode([output["token_ids"] for output in outputs], skip_special_tokens=True)
    if not need_scores:
        return [(text, None, None) for text in texts]
    return [(text, strip_stop(output["token_ids"], tokenizer), torch.tensor(output["logprobs"], dtype=torch.float32))
            for output, text in zip(outputs, texts)]


def keys_of(response):
    """(lower-cased trigger, type) of every record in a response string."""
    return {(str(e[0]).lower(), e[1]) for e in json.loads(response).get("events", [])
            if isinstance(e, list) and len(e) >= 2}


def oracle_dir_of(data_dir):
    """data/ace_b10_perm0/1 -> data/ace_oracle_b10_perm0/1 (build_ced_perms.py --oracle: same rows,
    old-type records kept), or None when the prefix has no oracle form."""
    base, task = os.path.split(os.path.normpath(data_dir))
    name = os.path.basename(base)
    oracle = re.sub(r"_(b\d+_perm\d+)$", r"_oracle_\1", name)
    return None if oracle == name else os.path.join(os.path.dirname(base), oracle, task)


def pl_quality(rows, gold_resp, oracle_rows, old_types, pending, all_scores, cand_idx):
    """Precision / recall of the pseudo-labels against the records the split stripped, matched on
    (trigger, type) as tools/ced_pl_quality.py does: memory rows skipped, recall over every stripped
    record (no-event rows included). Also at other shares q of the confidence filter: the same
    teacher answers, only the cutoff moves. None when the oracle rows do not line up."""
    if len(oracle_rows) != len(rows) or any(r["user_prompt"] != o["user_prompt"] for r, o in zip(rows, oracle_rows)):
        return None
    golds, stripped = {}, {}
    for i, (g, o) in enumerate(zip(gold_resp, oracle_rows)):
        gold = keys_of(g)
        if rows[i].get("is_memory") or (gold and {ty for _, ty in gold} <= old_types):
            continue
        golds[i], stripped[i] = gold, keys_of(o["response"]) - gold
    n_stripped = sum(len(s) for s in stripped.values())

    def score(added):  # row -> added keys
        n_added = sum(len(a) for a in added.values())
        hit = sum(len(a & stripped[i]) for i, a in added.items())
        return {"added": n_added, "hit": hit, "precision": round(100 * hit / max(n_added, 1), 2),
                "recall": round(100 * hit / max(n_stripped, 1), 2)}

    out = {"stripped": n_stripped, "stripped_in_candidates": sum(len(stripped.get(i, ())) for i in cand_idx),
           "kept": score({i: keys_of(rows[i]["response"]) - golds[i] for i in stripped})}
    if all_scores:
        for q in (30, 50, 70, 100):
            cut = all_scores[min(int(len(all_scores) * (1 - q / 100.0)), len(all_scores) - 1)]
            out[f"q{q}"] = score({i: {(str(ev[0]).lower(), ev[1]) for ev, s in evs if s is None or s >= cut} - golds[i]
                                  for i, evs in pending.items() if i in stripped})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--streams", required=True)
    ap.add_argument("--task-id", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--lexicon-filter", type=int, default=1,
                    help="1: only accept (trigger,type) pairs seen in old tasks' gold train data")
    ap.add_argument("--conflict-dedup", type=int, default=1,
                    help="H2: drop pseudo events whose trigger overlaps any gold trigger (any type)")
    ap.add_argument("--conf-filter", choices=["none", "percentile", "thresh"], default="none",
                    help="H1: teacher-confidence filter over decision-token logprobs")
    ap.add_argument("--conf-percentile", type=float, default=70.0,
                    help="keep the top p%% highest-scored events of the task")
    ap.add_argument("--conf-thresh", type=float, default=None,
                    help="absolute mean-logprob cutoff (e.g. -0.5)")
    ap.add_argument("--anchor", choices=["trigger", "pair"],
                    default=os.environ.get("PL_ANCHOR", "trigger"),
                    help="pair: anchor = (subject, object), for sentence-level CRE")
    ap.add_argument("--gen-backend", choices=["hf", "vllm"], default="hf",
                    help="vllm: the teacher's answers come from vLLM (gen_backend.py), same settings")
    ap.add_argument("--oracle-dir", default="auto",
                    help="unstripped split for PL precision/recall in the log: auto = the --oracle build "
                         "next to --data-dir (data/ace_oracle_b10_perm<p>/<t>) when it exists, none = off")
    args = ap.parse_args()

    if args.conf_filter == "thresh" and args.conf_thresh is None:
        ap.error("--conf-filter thresh requires --conf-thresh")

    streams = json.load(open(args.streams))
    old_types = set()
    for s in streams[:args.task_id]:
        old_types.update(s)
    assert old_types, "task 0 has no old types — pseudo-labeling starts at task 1"

    # trigger lexicon from old tasks' gold annotations: stripped triggers are real
    # ACE triggers so they re-occur in old train data; hallucinated pairs don't.
    lexicon = set()
    if args.lexicon_filter:
        base = os.path.dirname(os.path.normpath(args.data_dir))
        for t_prev in range(args.task_id):
            p = os.path.join(base, str(t_prev), "train.jsonl")
            for line in open(p):
                for e in json.loads(json.loads(line)["response"]).get("events", []):
                    if e[1] in old_types:
                        lexicon.add((e[0].lower(), e[1]))
        print(f"lexicon: {len(lexicon)} (trigger,type) pairs from tasks 0..{args.task_id-1}")

    rows = [json.loads(l) for l in open(os.path.join(args.data_dir, "train.jsonl"))]
    gold_resp = [r["response"] for r in rows]  # before the merge below rewrites them

    tokenizer = AutoTokenizer.from_pretrained(args.teacher, padding_side="left")

    need_scores = args.conf_filter != "none"

    # candidates: rows whose gold contains at least one NEW-type event (skip
    # replay exemplars and pure no-event rows keeps teacher calls low-risk)
    cand_idx = []
    for i, r in enumerate(rows):
        if r.get("is_memory"):  # sentence-level CRE memory: already fully annotated
            continue
        gold = json.loads(r["response"]).get("events", [])
        types = {e[1] for e in gold}
        if types - old_types:
            cand_idx.append(i)

    # pass 1: generate + hard filters; confidence filtering needs the full task
    # score distribution, so candidate events are buffered and merged in pass 2.
    pending = {}   # row_idx -> list of (event, score)
    n_dropped_conflict = 0
    n_seen = 0
    if args.gen_backend == "vllm":
        vllm_py, version = find_vllm_python()
        print(f"generation backend: vLLM {version} ({vllm_py})", flush=True)
        chunks = [(cand_idx, generate_vllm(args.teacher, tokenizer, chat_prompts(tokenizer, rows, cand_idx),
                                           args.max_new_tokens, need_scores, os.path.join(args.out, "vllm_tmp"),
                                           vllm_py))]
    else:
        device = f"cuda:{args.gpu}"
        model = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=torch.bfloat16,
                                                     device_map={"": device})
        model.eval()
        chunks = ((cand_idx[b:b + args.batch_size],
                   generate_hf(model, tokenizer, chat_prompts(tokenizer, rows, cand_idx[b:b + args.batch_size]),
                               args.max_new_tokens, need_scores, device))
                  for b in range(0, len(cand_idx), args.batch_size))
    n_done = 0
    for idxs, generated in chunks:
        for i, (text, gid, lp) in zip(idxs, generated):
            r = rows[i]
            sent = input_text_of(r["user_prompt"]) or ""
            gold = json.loads(r["response"]).get("events", [])
            if args.anchor == "pair":
                gold_keys = {(e[0], e[1], object_of(e)) for e in gold}
                gold_anchors = {(str(e[0]).lower(), str(object_of(e)).lower()) for e in gold}
            else:
                gold_keys = {(e[0], e[1]) for e in gold}
                gold_triggers = {str(e[0]).lower() for e in gold if isinstance(e, list) and e}
            for e in parse_events(text):
                if not isinstance(e, list) or len(e) < 2:
                    continue
                trig, ty = e[0], e[1]
                if not isinstance(ty, str) or ty not in old_types:
                    continue
                if not isinstance(trig, str) or trig not in sent:
                    continue
                if args.anchor == "pair":
                    obj = object_of(e)
                    if obj is None or obj not in sent:
                        continue
                    key, anchor = (trig, ty, obj), (trig.lower(), obj.lower())
                    if key in gold_keys:
                        continue
                    # H2 for pairs: never put a second label on an already labelled pair
                    if args.conflict_dedup and anchor in gold_anchors:
                        n_dropped_conflict += 1
                        continue
                else:
                    if (trig, ty) in gold_keys:
                        continue
                    # H2: never contradict a gold label on the same/overlapping token
                    if args.conflict_dedup:
                        tl = trig.lower()
                        if any(tl == g or tl in g or g in tl for g in gold_triggers):
                            n_dropped_conflict += 1
                            continue
                if args.lexicon_filter and (trig.lower(), ty) not in lexicon:
                    continue
                args_clean = []
                if len(e) > 2 and isinstance(e[2], list):
                    for a in e[2]:
                        if isinstance(a, list) and len(a) >= 2 \
                                and isinstance(a[0], str) and isinstance(a[1], str):
                            args_clean.append([a[0], a[1]])
                ev = [trig, ty, args_clean,
                      e[3] if len(e) > 3 and isinstance(e[3], str) else ""]
                score = None
                if need_scores:
                    score = event_conf_score(text, trig, ty, gid, lp, tokenizer)
                pending.setdefault(i, []).append((ev, score))
                if args.anchor == "pair":
                    gold_keys.add(key)
                    if args.conflict_dedup:
                        gold_anchors.add(anchor)
                else:
                    gold_keys.add((trig, ty))
                    if args.conflict_dedup:
                        # an accepted pseudo trigger also blocks later overlapping
                        # pseudo events (pseudo-vs-pseudo conflicts, not just gold)
                        gold_triggers.add(trig.lower())
                n_seen += 1
        n_done += len(idxs)
        print(f"pseudo-label {n_done}/{len(cand_idx)} "
              f"(candidates so far: {n_seen}, conflict-dropped: {n_dropped_conflict})", flush=True)

    # pass 2: H1 confidence filter over the whole task, then merge
    all_scores = sorted(s for evs in pending.values() for _, s in evs if s is not None)
    cutoff = None
    if args.conf_filter == "percentile" and all_scores:
        k = int(len(all_scores) * (1 - args.conf_percentile / 100.0))
        cutoff = all_scores[min(k, len(all_scores) - 1)]
    elif args.conf_filter == "thresh":
        cutoff = args.conf_thresh

    n_aug_rows = 0
    n_aug_events = 0
    n_dropped_conf = 0
    for i, evs in pending.items():
        kept = []
        for ev, score in evs:
            if cutoff is not None and score is not None and score < cutoff:
                n_dropped_conf += 1
                continue
            kept.append(ev)
        if kept:
            gold = json.loads(rows[i]["response"]).get("events", [])
            rows[i]["response"] = json.dumps({"events": gold + kept})
            n_aug_rows += 1
            n_aug_events += len(kept)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for split in ["dev.jsonl", "test.jsonl"]:
        shutil.copy(os.path.join(args.data_dir, split), os.path.join(args.out, split))

    stats = {"task_id": args.task_id, "candidates": len(cand_idx),
             "aug_rows": n_aug_rows, "aug_events": n_aug_events,
             "dropped_conflict": n_dropped_conflict, "dropped_conf": n_dropped_conf,
             "conf_filter": args.conf_filter, "conf_cutoff": cutoff}
    if all_scores:
        import statistics as st
        qs = {f"p{q}": all_scores[int(len(all_scores) * q / 100)]
              for q in (10, 30, 50, 70, 90) if len(all_scores) > 10}
        stats["score_dist"] = {"n": len(all_scores), "mean": st.mean(all_scores),
                               "min": all_scores[0], "max": all_scores[-1], **qs}
    oracle_dir = oracle_dir_of(args.data_dir) if args.oracle_dir == "auto" else \
        (None if args.oracle_dir == "none" else args.oracle_dir)
    if oracle_dir and os.path.exists(os.path.join(oracle_dir, "train.jsonl")):
        try:  # a log line only: never let it fail the pseudo-labeling
            q = pl_quality(rows, gold_resp, [json.loads(l) for l in open(os.path.join(oracle_dir, "train.jsonl"))],
                           old_types, pending, all_scores, cand_idx)
            if q is None:
                print(f"PL_QUALITY skipped: {oracle_dir} rows do not line up with {args.data_dir}")
            else:
                stats["pl_quality"] = q
                print(f"PL_QUALITY task{args.task_id} vs {oracle_dir}: stripped {q['stripped']} "
                      f"({q['stripped_in_candidates']} in candidate rows)")
                for k in ["kept"] + [k for k in q if k.startswith("q")]:
                    v = q[k]
                    print(f"PL_QUALITY task{args.task_id} {k:>5s}: added {v['added']} hit {v['hit']} "
                          f"P {v['precision']:.2f} R {v['recall']:.2f}")
        except Exception as e:
            print(f"PL_QUALITY failed: {e!r}")
    elif oracle_dir:
        print(f"PL_QUALITY skipped: no {oracle_dir}/train.jsonl")
    with open(os.path.join(args.out, "pl_stats.json"), "w") as f:
        json.dump(stats, f)
    print("STATS", json.dumps(stats))


if __name__ == "__main__":
    main()
