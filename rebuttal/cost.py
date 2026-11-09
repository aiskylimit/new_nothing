"""Aggregate accounting under one definition; do not equate missing counters to zero."""
import json
from pathlib import Path

from rebuttal.common import rows, write


def aggregate(run_dir):
    run_dir = Path(run_dir)
    stages = [row for row in rows(run_dir / 'stage_events.jsonl') if not row['stage'].startswith('evaluation_')]
    completed = [r for r in stages if r['status'] == 'complete']
    result = {'definition': {'gpu_hours': 'sum(worker GPU count * worker wall seconds)/3600; reserved-job GPU-hours also reported',
        'flops': 'estimated: 2*N*tokens + 4*layers*hidden*query_tokens*key_tokens, with input-gradient/recompute equivalents; SGL SVD and CoReD merge use explicit dense estimates; excludes optimizer and IG bookkeeping; causal sparsity/prefix caching can reduce actual work',
        'tokens': 'unpadded model-forward token exposures; includes extra expert/teacher forwards, not unique corpus tokens',
        'memory': 'torch peak allocated/reserved per process; external nvidia-smi used memory separately'},
        'stages': [], 'estimated_flops': 0.0, 'training_tokens': 0, 'forward_tokens': 0,
        'optimization_steps': 0, 'peak_allocated_bytes': None, 'peak_reserved_bytes': None}
    complete_flops = not any(s['status'] != 'complete' and s['gpu_count'] for s in stages)
    result['estimated_flops_scope'] = 'completed recorded stages; failed-attempt counters are not reconstructed, so failures mark completeness false'
    expanded = []
    for stage in completed:
        if stage['stage'] == 'cored_pipeline':
            native = rows(run_dir / 'artifacts/cored/timings.jsonl')
            for row in native:
                if row['status'] == 'complete':
                    expanded.append(dict(stage, stage=row['stage'], seconds=row['seconds'],
                        gpu_count=0 if row['stage'] == 'prepare' else 1 if row['stage'] == 'merge' else stage['gpu_count'],
                        gpu_hours=row.get('gpu_hours', row['seconds']*(0 if row['stage']=='prepare' else 1 if row['stage']=='merge' else stage['gpu_count'])/3600),
                        telemetry_subdir=row['stage'], telemetry_dir=row.get('telemetry_dir')))
        else:
            expanded.append(stage)
    for stage in expanded:
        directory = Path(stage['telemetry_dir']) if stage.get('telemetry_dir') else run_dir / 'telemetry' / stage['attempt_id'] / stage.get('telemetry_subdir', '')
        telemetry = [json.loads(p.read_text()) for p in directory.glob('process-*.json')]
        # torchrun parent has no forwards. Zero means unavailable for uninstrumented kernels.
        active = [r for r in telemetry if r['forward_tokens'] or r['optimization_steps'] or r['estimated_flops']]
        record = dict(stage, gpu_hours=stage.get('gpu_hours', stage['gpu_count'] * stage['seconds'] / 3600),
                      estimated_flops=sum(r['estimated_flops'] for r in active) if active else None,
                      forward_tokens=sum(r['forward_tokens'] for r in active) if active else None,
                      autograd_tokens=sum(r['autograd_tokens'] for r in active) if active else None,
                      optimization_steps=max((r['optimization_steps'] for r in active), default=0),
                      sample_presentations=sum(r['sample_presentations'] for r in active), process_telemetry=telemetry)
        if stage['gpu_count'] and not record['estimated_flops']:
            record['estimated_flops'] = None
        if stage['gpu_count'] and not record['forward_tokens'] and stage['stage'] != 'merge':
            record['forward_tokens'] = None
        for key in ('peak_allocated_bytes', 'peak_reserved_bytes'):
            values = [r[key] for r in telemetry if r.get(key) is not None]
            record[key] = max(values) if values else None
            if values:
                result[key] = max(result[key] or 0, max(values))
        result['stages'].append(record)
        if record['estimated_flops'] is None and stage['gpu_count']:
            complete_flops = False
        result['estimated_flops'] += record['estimated_flops'] or 0
        result['training_tokens'] += record['autograd_tokens'] or 0
        result['forward_tokens'] += record['forward_tokens'] or 0
        result['optimization_steps'] += record['optimization_steps']
    result.update(completed_stage_gpu_hours=sum(s['gpu_hours'] for s in result['stages']),
                  all_attempt_gpu_hours=sum(s.get('gpu_hours', s['gpu_count'] * s['seconds'] / 3600) for s in stages),
                  stage_wall_sum_seconds=sum(s['seconds'] for s in completed),
                  flops_complete=complete_flops, flops_type='estimated')
    # Pipeline CoReD events are sequential; native stage markers include startup, IO.
    result['tokens_per_second'] = result['training_tokens'] / max(result['stage_wall_sum_seconds'], 1e-9)
    result['steps_per_second'] = result['optimization_steps'] / max(result['stage_wall_sum_seconds'], 1e-9)
    result['optimization_stage_tokens'] = sum(s['autograd_tokens'] or 0 for s in result['stages']
                                             if s['stage'] in {'optimization', 'phase1', 'distill'})
    result['token_counter_coverage_complete'] = all(s['forward_tokens'] is not None for s in result['stages']
                                                  if s['stage'] in {'optimization', 'phase1', 'distill'})
    if not result['token_counter_coverage_complete']:
        result['training_tokens'] = result['optimization_stage_tokens'] = result['tokens_per_second'] = None
    samples = [sample for event in stages for sample in json.loads(
        (run_dir / 'telemetry' / event['attempt_id'] / 'device_memory_samples.json').read_text()).get('samples', [])
        ] if stages and all((run_dir / 'telemetry' / e['attempt_id'] / 'device_memory_samples.json').exists() for e in stages) else []
    result['sampled_device_used_memory_peak_mib'] = max((s['used_memory_mib'] for s in samples), default=None)
    jobs = rows(run_dir / 'job_events.jsonl')
    result['job_wall_seconds_including_failed_attempts'] = sum(r['wall_seconds'] for r in jobs)
    result['allocated_gpu_hours'] = sum(r['allocated_gpu_count'] * r['wall_seconds'] / 3600 for r in jobs)
    if (run_dir / 'run.json').exists():
        config = json.loads((run_dir / 'run.json').read_text())
        result.update(source_samples=config['data']['expected_samples'], precision=config['train']['precision'],
                      batch_configuration=config['train'], teacher=config['data'], environment=config.get('environment'))
        result['retained_training_samples'] = result['source_samples']
        for audit in (run_dir / 'adaptations.json', run_dir / 'selected.jsonl.adaptations.json'):
            if audit.exists():
                result['retained_training_samples'] = json.loads(audit.read_text())['retained_samples']
        if (run_dir / 'masks/masks.jsonl').exists():
            result['retained_training_samples'] = len(rows(run_dir / 'masks/masks.jsonl'))
    result['sample_presentations'] = sum(s['sample_presentations'] for s in result['stages'])
    write(run_dir / 'training_cost.json', result)
    return result


def tables(root, destination, configurations=None):
    from rebuttal.common import matrix
    root, destination = Path(root), Path(destination)
    configs = matrix() if configurations is None else configurations
    output = {}
    for config in configs:
        directory = root / config['run_id']
        if (directory / 'final_checkpoint.json').exists():
            output[config['run_id']] = aggregate(directory)
    write(destination / 'costs.json', output)
    for setting in sorted({c['setting'] for c in configs}):
        table = ['|Method|Allocated GPU-hours|Active stage GPU-hours|Estimated FLOPs|Peak allocated bytes|Autograd token exposures|Overhead vs Long-CoT|',
                 '|---|---:|---:|---:|---:|---:|---:|']
        baseline = next(c['run_id'] for c in configs if c['setting'] == setting and c['method'] == 'longcot')
        denominator = output.get(baseline, {}).get('allocated_gpu_hours')
        for config in configs:
            if config['setting'] != setting:
                continue
            value = output.get(config['run_id'])
            if value is None:
                table.append(f"|{config['method']}|TBD|TBD|TBD|TBD|TBD|TBD|")
                continue
            flop = f"{value['estimated_flops']:.4g}" if value['flops_complete'] else 'incomplete (see JSON)'
            ratio = f"{value['allocated_gpu_hours']/denominator:.3f}" if denominator else 'TBD'
            table.append(f"|{config['method']}|{value['allocated_gpu_hours']:.5f}|{value['completed_stage_gpu_hours']:.5f}|{flop}|{value['peak_allocated_bytes']}|{value['training_tokens']}|{ratio}|")
        destination.mkdir(parents=True, exist_ok=True)
        (destination / f'cost_setting_{setting}.md').write_text('\n'.join(table)+'\n')
        core = next(c['run_id'] for c in configs if c['setting'] == setting and c['method'] == 'cored')
        if core in output:
            breakdown = ['|Stage|Wall seconds|Active GPU-hours|Estimated FLOPs|Peak allocated bytes|Tokens/sec|', '|---|---:|---:|---:|---:|---:|']
            for row in output[core]['stages']:
                speed = f"{row['forward_tokens']/max(row['seconds'], 1e-9):.2f}" if row['forward_tokens'] is not None else 'TBD'
                breakdown.append(f"|{row['stage']}|{row['seconds']:.3f}|{row['gpu_hours']:.5f}|{row['estimated_flops']}|{row['peak_allocated_bytes']}|{speed}|")
            (destination / f'cored_stages_{setting}.md').write_text('\n'.join(breakdown)+'\n')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--runs', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    tables(args.runs, args.output)
