"""Native P-ALIGN and SSFT preprocessing, persisted before optimization."""
import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from rebuttal.common import ROOT, jsonl, rows, write


def palign(model, source, output, shard, shards):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from rebuttal.evaluate import grade_math
    from rebuttal.telemetry import inference
    spec = importlib.util.spec_from_file_location('palign_binary', ROOT / 'palign/src/binary_search.py')
    binary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(binary)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=True)
    context = min(65536, json.loads((Path(model) / 'config.json').read_text()).get('max_position_embeddings', 32768))
    llm = LLM(model=model, tensor_parallel_size=1, max_model_len=context, gpu_memory_utilization=.9, seed=42)
    binary.model = object()
    binary.tokenizer = tokenizer
    def complete(prompt, tokens, temperature):
        rendered = binary.apply_chat([{'role': 'user', 'content': prompt}])
        tokens = min(tokens, context - len(tokenizer.encode(rendered, add_special_tokens=False)))
        if tokens < 1:
            raise ValueError('P-ALIGN prompt exceeds native context; review explicit overlength policy')
        result = llm.generate([rendered], SamplingParams(n=1, temperature=temperature, top_p=.9,
                              repetition_penalty=1.05, max_tokens=tokens, seed=42))
        inference(result, llm.llm_engine.model_config.hf_config)
        return result[0].outputs[0].text
    judge_errors = []
    def judge_chat(prompt, unused):
        try:
            return complete(prompt, 256, 0)
        except Exception as error:
            judge_errors.append(error)
            raise
    binary.chat = judge_chat
    previous = {r['id']: r for r in rows(output, repair=True)}
    for row in rows(source)[shard::shards]:
        if row['id'] in previous:
            continue
        prefix, length, sufficient, judgment = binary.find_minimal_sufficient_prefix(
            row['question'], binary.split_sentences(row['thinking']), sleep_sec=0)
        if judge_errors:
            raise RuntimeError('P-ALIGN self-judgment failed; refusing a silent full-prefix fallback') from judge_errors[-1]
        prompt = ('Please continue from the draft and solve the problem step by step, '
                  'and put your final answer within \\boxed{}. '
                  'I will provide you with some prior knowledge as a draft to assist you in solving the question.'
                  f"*Question*:{row['question']}\n*Prefix*:{prefix}")
        continuation = complete(prompt, 32768, .6)
        correct, predicted = grade_math(continuation, row['gold'])
        record = dict(row, prefix=prefix, prefix_sentences=length, sufficient=sufficient,
                      judgment=judgment, continuation=continuation, accepted=correct, predicted_answer=predicted)
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with Path(output).open('a') as f:
            f.write(json.dumps(record, ensure_ascii=False)+'\n')
            f.flush()


def ssft_targets(records):
    from SegmentSelectiveSFT.prepare_s1k import last_boxed
    retained, rejected = [], []
    for row in records:
        target = last_boxed(row['solution']) or row['answer']
        if not target or len(target) > 200:
            rejected.append({'id': row['id'], 'reason': 'native SSFT attribution target missing or >200 chars'})
        else:
            retained.append(dict(row, answer=target))
    return retained, rejected


def ssft(model, source, output, settings=None, shard=0, shards=1):
    if shards < 1 or not 0 <= shard < shards:
        raise ValueError('Invalid SSFT shard')
    records = rows(source)[shard::shards]
    retained, rejected = ssft_targets(records)
    write(str(output)+'.adaptations.json', {'input_samples': len(records), 'retained_samples': len(retained), 'rejected': rejected,
          'target': 'last boxed answer in reasoning, fallback ground truth; native prepare_s1k policy'})
    records = retained
    # Small smokes or target rejection can leave a worker empty. The merged
    # result must be nonempty, but an individual worker need not have examples.
    if not records:
        if shards == 1:
            raise ValueError('SSFT target validation retained zero samples')
        jsonl(output, [])
        write(str(output)+'.cost.json', {'prep_wall_time_sec': 0.0, 'prep_peak_vram_gb': 0.0,
              'prep_flops': 0.0, 'prep_n_params': 0, 'tokens_forwarded': 0, 'tokens_backwarded': 0})
        return
    import sys
    root = ROOT / 'SegmentSelectiveSFT'
    for folder in (root, root / 'Attribution', root / 'SelectiveSFT', root / 'Eval'):
        sys.path.insert(0, str(folder))
    import one_pass
    args = SimpleNamespace(model=model, segment_mode='paragraph', full_sft=False,
        no_ig_grad_checkpoint=False, ig_steps=50, ig_batch_size=1, ig_max_tokens=0,
        cumulative_ratio=.7, coherence_max=.8)
    if settings:
        args.ig_steps = settings['ig_steps']
        args.cumulative_ratio = settings['mass']
        args.coherence_max = settings['coherence']
    cost = one_pass.stage_prep(args, records)
    jsonl(output, records)
    write(str(output)+'.cost.json', cost)


def merge_ssft(source, parts, output):
    """Validate complete shard coverage and restore original training order."""
    if not parts:
        raise ValueError('SSFT attribution requires at least one shard')
    source_rows = rows(source)
    expected, rejected = ssft_targets(source_rows)
    ids = [row['id'] for row in source_rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate SSFT source IDs')
    merged = {}
    for index, part in enumerate(parts):
        if not Path(part).is_file():
            raise FileNotFoundError(part)
        shard_expected, _ = ssft_targets(source_rows[index::len(parts)])
        values = rows(part)
        if {row['id'] for row in values} != {row['id'] for row in shard_expected}:
            raise ValueError('Incomplete/unexpected SSFT shard IDs')
        for row in values:
            if row['id'] in merged:
                raise ValueError('Duplicate SSFT attribution ID')
            merged[row['id']] = row
    if not expected:
        raise ValueError('SSFT target validation retained zero samples')
    costs = [json.loads(Path(str(part)+'.cost.json').read_text()) for part in parts]
    jsonl(output, [merged[row['id']] for row in expected])
    write(str(output)+'.adaptations.json', {'input_samples': len(source_rows),
          'retained_samples': len(expected), 'rejected': rejected,
          'target': 'last boxed answer in reasoning, fallback ground truth; native prepare_s1k policy'})
    # Timings and GPU-hours are measured by the runner around concurrent workers;
    # retain each native cost record instead of summing their wall times.
    write(str(output)+'.cost.json', {'workers': costs})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', required=True, choices=['palign', 'ssft', 'teacher-stats'])
    p.add_argument('--model', required=True)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, help='Defaults to 3 for P-ALIGN, 1 for standalone SSFT')
    p.add_argument('--config', type=Path)
    args = p.parse_args()
    if args.method == 'palign':
        palign(args.model, args.source, args.output, args.shard, args.shards if args.shards is not None else 3)
    elif args.method == 'ssft':
        ssft(args.model, args.source, args.output, json.loads(args.config.read_text())['method_specific'] if args.config else None,
             args.shard, args.shards if args.shards is not None else 1)
    else:
        from transformers import AutoTokenizer
        from rebuttal.data import teacher_statistics
        from rebuttal.evaluate import grade_math
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        write(args.output, teacher_statistics(rows(args.source), tokenizer, grade_math))


if __name__ == '__main__':
    main()
