"""Development-only end-to-end tests on two 24GB GPUs, never main experiments."""
import argparse
import copy
import fcntl
import json
import os
import subprocess
from datetime import timedelta
from pathlib import Path

from rebuttal.common import ROOT, jsonl, matrix, sha, write
from rebuttal.data import adapt
from rebuttal.runner import Job, python_for


def probe_worker():
    import torch
    import torch.distributed as dist
    local = int(os.environ['LOCAL_RANK'])
    if torch.cuda.device_count() != 2 or not torch.distributed.is_nccl_available():
        raise RuntimeError('This test needs two visible CUDA devices and NCCL')
    torch.cuda.set_device(local)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('Both GPUs must support bf16')
    dist.init_process_group('nccl', timeout=timedelta(seconds=120))
    try:
        value = torch.tensor(float(local + 1), device='cuda', dtype=torch.bfloat16)
        dist.all_reduce(value)
        if value.item() != 3:
            raise RuntimeError('Two-rank NCCL all_reduce failed')
        x = torch.ones(16, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        x.square().sum().backward()
        if not torch.isfinite(x.grad).all():
            raise RuntimeError('bf16 backward produced invalid gradients')
        print(json.dumps({'rank': local, 'gpu': torch.cuda.get_device_name(local),
                          'torch': torch.__version__, 'cuda': torch.version.cuda,
                          'nccl_all_reduce': 'passed', 'bf16_backward': 'passed'}), flush=True)
    finally:
        dist.destroy_process_group()


def fixtures(destination):
    """Short synthetic arithmetic, explicitly unrelated to teacher robustness."""
    for variant in ('original', 'gemini'):
        values = []
        for i, (a, b) in enumerate(((2, 2), (1, 2), (3, 2), (4, 2))):
            gold = str(a + b)
            thinking = (f'We need to add {a} and {b}.\n\n'
                        f'The sum is {gold}.\n\nTherefore the answer is \\boxed{{{gold}}}.')
            if variant == 'gemini':
                thinking = f'Compute the requested sum.\n\n{a} + {b} = {gold}.\n\nAnswer: \\boxed{{{gold}}}.'
            values.append({'id': f'fixture-{i}', 'question': f'What is {a} + {b}? Give the final answer in \\boxed{{}}.',
                           'thinking': thinking, 'answer_text': f'\\boxed{{{gold}}}', 'gold': gold,
                           'teacher': 'synthetic fixture', 'source_type': 'synthetic arithmetic smoke'})
        adapt(values, destination / variant)
    manifest = {'kind': 'synthetic arithmetic smoke; NOT s1K or actual Gemini/DeepSeek traces',
                'files': {v: sha(destination / v / 'canonical.jsonl') for v in ('original', 'gemini')}}
    write(destination / 'manifest.json', manifest)
    for name in ('aime24', 'aime25', 'amc12', 'math500', 'arc_challenge', 'logiqa2'):
        values = []
        for i in range(2):
            item = {'question_id': f'fixture-{name}-{i}', 'question': f'What is {i + 1} + 1?', 'answer': str(i + 2)}
            if name in ('arc_challenge', 'logiqa2'):
                item.update(question=item['question'] + f'\nA. {i + 2}\nB. 9\nC. 10\nD. 11',
                            answer='A', choices=list('ABCD'))
            values.append(item)
        jsonl(destination / 'benchmarks' / f'{name}.jsonl', values)
    write(destination / 'benchmarks/manifest.json', {name: {'count': 2, 'synthetic': True,
          'sha256': sha(destination / 'benchmarks' / f'{name}.jsonl')}
          for name in ('aime24', 'aime25', 'amc12', 'math500', 'arc_challenge', 'logiqa2')})


def configurations(settings, selected=None):
    configs = []
    for original in matrix():
        if original['setting'] not in settings or (selected and original['run_id'] not in selected):
            continue
        value = copy.deepcopy(original)
        value.update(run_id='dev2_' + original['run_id'], smoke=True, smoke_samples=4, smoke_steps=4,
                     development_smoke={'context': 2048, 'new_tokens': 256})
        value['data'].update(repo='synthetic/arithmetic-smoke', revision='fixture-v1', expected_samples=4,
                             teacher='synthetic fixture; NOT actual teacher data')
        value['train'].update(global_batch=2, micro_batch=1, gradient_accumulation=1, cutoff=512)
        value['hardware'].update(training_gpu_count=2, type='two-GPU development test; not B200 reproduction')
        value['evaluation'].update(seeds=[42], budgets=[1], math_max_tokens=256, arc_max_tokens=256,
                                   logiqa_max_tokens=256, context=2048, synthetic_fixtures=True)
        if value['method'] == 'ssft':
            value['method_specific'].update(ig_steps=4, require_valid_attribution=True)
        configs.append(value)
    return configs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe-worker', action='store_true')
    p.add_argument('--gpus', default='0,1')
    p.add_argument('--assets', type=Path, default=ROOT / 'rebuttal/assets')
    p.add_argument('--runs', type=Path, default=ROOT / 'rebuttal/smoke-2x3090')
    p.add_argument('--settings', nargs='+', choices=list('ABC'), default=list('ABC'))
    p.add_argument('--run-id', nargs='+', help='Main IDs such as C_cored_1.5b_seed42; outputs get dev2_ prefix')
    args = p.parse_args()
    if args.probe_worker:
        probe_worker()
        return
    gpu_ids = args.gpus.split(',')
    if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2 or any(not s.isdigit() for s in gpu_ids):
        p.error('Use two distinct GPU indices, e.g. --gpus 0,1')
    selected = configurations(args.settings, args.run_id)
    if not selected or (args.run_id and set(args.run_id) != {c['run_id'][5:] for c in selected}):
        p.error('Unknown run ID or run ID outside --settings')
    assets, runs = args.assets.resolve(), args.runs.resolve()
    if runs in {(ROOT / 'rebuttal/runs').resolve(), (ROOT / 'rebuttal/smoke-runs').resolve()}:
        raise ValueError('Use a separate two-GPU directory, not main/smoke B200 outputs')
    runs.mkdir(parents=True, exist_ok=True)
    marker = runs / 'SMOKE_ONLY.json'
    if not marker.exists() and any(runs.iterdir()):
        raise ValueError('Choose an empty isolated smoke directory, never a main runs directory')
    if marker.exists() and json.loads(marker.read_text()).get('kind') != 'two-gpu-development-smoke':
        raise ValueError('Not a two-GPU development directory')
    with (runs / '.gpu.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(marker, {'kind': 'two-gpu-development-smoke', 'not_for_paper_metrics': True})
        summary = {'status': 'running', 'gpus': args.gpus, 'completed': [],
                   'limitation': 'Synthetic short data, 4 updates, IG4, single eval response; not full fit/performance validation'}
        write(runs / 'summary.json', summary)
        current = 'preflight'
        try:
            from rebuttal.provision import verify_snapshot
            for repo, revision in {(c['model']['repo'], c['model']['revision']) for c in selected}:
                verify_snapshot(assets / 'models' / repo.split('/')[-1], repo, revision)
            environments = {python_for(c['method']) for c in selected} | {python_for('palign')}
            probe_env = {**os.environ, 'CUDA_VISIBLE_DEVICES': args.gpus,
                         'PYTHONPATH': os.pathsep.join([str(ROOT), os.environ.get('PYTHONPATH', '')]), 'OMP_NUM_THREADS': '1'}
            for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
                probe_env.pop(key, None)
            for i, python in enumerate(sorted(environments)):
                print(f'[dev2] NCCL/bf16 probe: {python}', flush=True)
                with (runs / f'probe-{i}.log').open('w') as log:
                    subprocess.run([python, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                        '-m', 'rebuttal.smoke_2gpu', '--probe-worker'], cwd=ROOT, env=probe_env,
                        stdout=log, stderr=subprocess.STDOUT, check=True, timeout=240)
            data = runs / 'fixtures'
            fixtures(data)
            for config in selected:
                current = config['run_id']
                config['hardware']['gpus'] = args.gpus
                print(f'[dev2] START {current}', flush=True)
                job = Job(config, assets, data, runs, args.gpus)
                final = job.root / 'final_checkpoint.json'
                if final.exists():
                    checkpoint = Path(json.loads(final.read_text())['path'])
                    manifest = json.loads((checkpoint / 'checkpoint_manifest.json').read_text())
                    if any(sha(checkpoint / f) != checksum for f, checksum in manifest['files_sha256'].items()):
                        raise ValueError('Saved smoke checkpoint changed')
                else:
                    job.train()
                job.evaluate()
                summary['completed'].append(current)
                write(runs / 'summary.json', summary)
                print(f'[dev2] PASS {current}: preprocessing, training, checkpoint reload, vLLM and scoring', flush=True)
            summary['status'] = 'passed'
        except BaseException as error:
            summary.update(status='failed', failed_run=current, error=repr(error))
            raise
        finally:
            write(runs / 'summary.json', summary)
        print(f'[dev2] ALL SELECTED RUNS PASSED. Summary: {runs / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
