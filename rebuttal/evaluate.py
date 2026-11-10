"""Shared vLLM math/OOD evaluation with per-response durable records and regrading."""
import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

from rebuttal.common import BUDGETS, MATH, ROOT, SEEDS, jsonl, rows, sha, signature, write


def mc_answer(text, choices):
    text = text.rsplit('</think>', 1)[-1]
    matches = list(re.finditer(r'(?:\b(?:final\s+)?answer\s*(?:is|:|=)\s*\(?([A-E])\)?\b|\\boxed\{\s*([A-E])\s*\})', text, re.I))
    if not matches:
        return None
    answer = next(x for x in matches[-1].groups() if x).upper()
    return answer if answer in choices else None


def grade_math(response, gold):
    # Exactly the existing P-ALIGN parse('$gold$') / parse(prediction) rule.
    from math_verify import parse, verify
    try:
        predicted = parse(response)
        return bool(verify(parse('$' + str(gold) + '$'), predicted)), str(predicted) if predicted else None
    except Exception:
        return False, None


def grade_record(record):
    if record['benchmark'] in MATH:
        correct, predicted = grade_math(record['response'], record['ground_truth'])
    else:
        predicted = mc_answer(record['response'], record['choices'])
        correct = predicted == record['ground_truth']
    return dict(record, predicted_answer=predicted, is_correct=bool(correct), invalid_answer=predicted is None)


def key(record):
    return record['question_id'], record['eval_seed'], record['sample_id']


def development_limits(config):
    value = (config or {}).get('development_smoke')
    if value and not config.get('smoke'):
        raise ValueError('Development evaluation limits require a smoke run')
    return value


def math_schedule(config=None):
    return [(42, 1)] if development_limits(config) else list(zip(SEEDS, BUDGETS))


def evaluator_environment():
    import importlib.metadata
    import torch
    versions = {}
    for package in ('torch','torchvision','torchaudio','vllm','transformers','peft','math-verify'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {'python':sys.executable, 'packages':versions, 'torch_cuda':torch.version.cuda,
            'backend_source_sha256':sha(ROOT/'palign/src/test.py')}


def validate_records(records, questions, benchmark, scored=False, config=None):
    schedule = math_schedule(config) if benchmark in MATH else [(0, 1)]
    expected = {(row['question_id'], seed, sample) for row in questions for seed, count in schedule for sample in range(count)}
    actual = [key(row) for row in records]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError(f'{benchmark}: incomplete or duplicated response grid; expected {len(expected)}, got {len(actual)}')
    if scored and any(not isinstance(row.get('is_correct'), bool) or row['finish_reason'] == 'error' for row in records):
        raise ValueError('Unscored/error responses cannot enter final metrics')


def generate(config, checkpoint, benchmarks, destination, shard, shards, batch_size):
    # Each worker sees one physical GPU; TP=1. Seed/sample IDs do not depend on shard.
    from transformers import AutoTokenizer
    development = development_limits(config)
    context = development['context'] if development else 32768
    spec = importlib.util.spec_from_file_location('palign_eval_backend', ROOT / 'palign/src/test.py')
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    from vllm import SamplingParams
    adapter = checkpoint / 'adapter_config.json'
    adapter_config = json.loads(adapter.read_text()) if adapter.exists() else None
    base = Path(adapter_config['base_model_name_or_path']) if adapter_config else checkpoint
    if not base.is_dir():
        raise FileNotFoundError(f'Checkpoint base must be local: {base}')
    tokenizer = AutoTokenizer.from_pretrained(str(base), local_files_only=True, trust_remote_code=True)
    model = None
    request = None
    provenance = {'run_id': config['run_id'], 'checkpoint': str(checkpoint),
                  'checkpoint_id': signature({p.name: sha(p) for p in sorted(checkpoint.iterdir()) if p.is_file()
                    and (p.suffix in {'.safetensors', '.json', '.jinja', '.model'} or p.name == 'pytorch_model.bin')
                    and p.name != 'checkpoint_manifest.json'}),
                  'protocol': config['evaluation']}
    destination.mkdir(parents=True, exist_ok=True)
    metadata = destination / 'provenance.json'
    if metadata.exists() and json.loads(metadata.read_text()) != provenance:
        raise ValueError('Evaluation checkpoint/protocol changed; choose a fresh output directory')
    write(metadata, provenance)
    benchmark_manifest = json.loads((benchmarks / 'manifest.json').read_text()) if (benchmarks / 'manifest.json').exists() else {}
    for benchmark in config['evaluation']['math'] + config['evaluation']['ood']:
        source = benchmarks / f'{benchmark}.jsonl'
        question_rows = rows(source)
        question_rows = question_rows[shard::shards]
        output = destination / f'{benchmark}.rank{shard}.jsonl'
        previous = rows(output, repair=True)
        existing = {key(row) for row in previous}
        if len(existing) != len(previous):
            raise ValueError(f'Duplicate raw record in {output}')
        if any(row['benchmark_sha256'] != sha(source) or row['checkpoint_id'] != provenance['checkpoint_id'] for row in previous):
            raise ValueError('Benchmark/checkpoint changed since saved raw generations')
        math = benchmark in MATH
        caps = 4096 if math or benchmark == 'logiqa2' else 1024
        if development:
            caps = min(caps, development['new_tokens'])
        schedule = math_schedule(config) if math else [(0, 1)]
        for seed, count in schedule:
            pending = [row for row in question_rows if any((row['question_id'], seed, s) not in existing for s in range(count))]
            for start in range(0, len(pending), batch_size):
                batch = pending[start:start + batch_size]
                prompts = []
                for row in batch:
                    instruction = backend.PROMPT + row['question'] if math else (
                        'Reason through the question and choices. End your response with exactly "Answer: X", '
                        'where X is the letter of the correct choice.\n\n' + row['question'])
                    prompts.append(backend.apply_chat(tokenizer, [{'role': 'user', 'content': instruction}]))
                if model is None:
                    write(destination/'evaluator_environment.json', evaluator_environment())
                    options = {'enforce_eager': True, 'max_num_seqs': 1} if development else {}
                    model = backend._make_llm(str(base), context, 1, .9, seed=42, adapter_config=adapter_config, **options)
                    if adapter_config:
                        from vllm.lora.request import LoRARequest
                        request = LoRARequest('trained', 1, str(checkpoint))
                generation = {'n': count, 'seed': seed, 'temperature': .6 if math else 0.0,
                              'top_p': .9 if math else 1.0, 'repetition_penalty': 1.05,
                              'max_tokens': caps, 'max_model_len': context, 'tensor_parallel_size': 1,
                              'backend': 'vllm', 'do_sample': math}
                params = SamplingParams(**{k: generation[k] for k in ('n', 'seed', 'temperature', 'top_p', 'repetition_penalty', 'max_tokens')})
                outputs = model.generate(prompts, params, **({'lora_request': request} if request else {}))
                if len(outputs) != len(batch):
                    raise ValueError('vLLM output count mismatch')
                with output.open('a') as handle:
                    for row, prompt, result in zip(batch, prompts, outputs):
                        if len(result.outputs) != count:
                            raise ValueError('vLLM candidate count mismatch')
                        for sample, completion in enumerate(result.outputs):
                            record_key = row['question_id'], seed, sample
                            if record_key in existing:
                                continue
                            record = dict(provenance, benchmark=benchmark, benchmark_sha256=sha(source),
                                benchmark_version=benchmark_manifest.get(benchmark, {'sha256': sha(source)}),
                                question_id=row['question_id'], eval_seed=seed, sample_id=sample,
                                prompt=prompt, response=completion.text, predicted_answer=None,
                                ground_truth=row['answer'], is_correct=None, choices=row.get('choices'),
                                input_tokens=len(result.prompt_token_ids), generated_tokens=len(completion.token_ids),
                                finish_reason=completion.finish_reason, stop_reason=completion.stop_reason,
                                generation_config=generation)
                            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                            existing.add(record_key)
                    handle.flush()
                    os.fsync(handle.fileno())
        validate_records(rows(output), question_rows, benchmark, config=config)


def score(config, benchmark_dir, destination, workers):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for benchmark in config['evaluation']['math'] + config['evaluation']['ood']:
            raw = [record for path in sorted(destination.glob(f'{benchmark}.rank*.jsonl')) for record in rows(path)]
            validate_records(raw, rows(benchmark_dir / f'{benchmark}.jsonl'), benchmark, config=config)
            # Scoring is separate from generation; CPU parsing never stalls the GPU workers.
            jsonl(destination / f'{benchmark}.scored.jsonl', pool.map(grade_record, raw, chunksize=8))
    import importlib.metadata
    write(destination / 'grading_config.json', {'grader_source_sha256': sha(__file__),
          'math_verify_version': importlib.metadata.version('math-verify'),
          'math_rule': "verify(parse('$gold$'), parse(response))",
          'mc_rule': 'final explicit answer/boxed letter after think block; choices validated',
          'raw_sha256': {p.name: sha(p) for p in sorted(destination.glob('*.rank*.jsonl'))}})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--benchmarks', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--score-only', action='store_true')
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    if args.score_only:
        score(config, args.benchmarks, args.output, args.workers)
    else:
        if args.checkpoint is None or not 0 <= args.shard < args.shards or args.batch_size < 1:
            p.error('Need checkpoint and valid sharding/batch size')
        generate(config, args.checkpoint.resolve(), args.benchmarks, args.output, args.shard, args.shards, args.batch_size)


if __name__ == '__main__':
    main()
