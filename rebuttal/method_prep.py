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


def ssft(model, source, output, settings=None):
    import sys
    root = ROOT / 'SegmentSelectiveSFT'
    for folder in (root, root / 'Attribution', root / 'SelectiveSFT', root / 'Eval'):
        sys.path.insert(0, str(folder))
    import one_pass
    records = rows(source)
    from SegmentSelectiveSFT.prepare_s1k import last_boxed
    retained, rejected = [], []
    for row in records:
        target = last_boxed(row['solution']) or row['answer']
        if not target or len(target) > 200:
            rejected.append({'id': row['id'], 'reason': 'native SSFT attribution target missing or >200 chars'})
        else:
            retained.append(dict(row, answer=target))
    write(str(output)+'.adaptations.json', {'input_samples': len(records), 'retained_samples': len(retained), 'rejected': rejected,
          'target': 'last boxed answer in reasoning, fallback ground truth; native prepare_s1k policy'})
    records = retained
    if not records:
        raise ValueError('SSFT target validation retained zero samples')
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', required=True, choices=['palign', 'ssft', 'teacher-stats'])
    p.add_argument('--model', required=True)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=3)
    p.add_argument('--config', type=Path)
    args = p.parse_args()
    if args.method == 'palign':
        palign(args.model, args.source, args.output, args.shard, args.shards)
    elif args.method == 'ssft':
        ssft(args.model, args.source, args.output, json.loads(args.config.read_text())['method_specific'] if args.config else None)
    else:
        from transformers import AutoTokenizer
        from rebuttal.data import teacher_statistics
        from rebuttal.evaluate import grade_math
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        write(args.output, teacher_statistics(rows(args.source), tokenizer, grade_math))


if __name__ == '__main__':
    main()
