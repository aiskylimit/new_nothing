"""Full Setting B on two H100s: real Gemini data, five methods, no smoke limits."""
import argparse
import copy
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

from rebuttal.common import ROOT, matrix, sha, write
from rebuttal.cost import aggregate, tables
from rebuttal.preflight import validate
from rebuttal.runner import Job
from rebuttal.statistics import report


def configurations(gpus='0,1'):
    configs = []
    for original in matrix():
        if original['setting'] != 'B':
            continue
        config = copy.deepcopy(original)
        config['formal_profile'] = 'setting_b_2h100'
        config['hardware'].update(gpus=gpus, training_gpu_count=2, type='NVIDIA H100')
        config['train'].update(global_batch=32, micro_batch=1,
                               gradient_accumulation=16 if config['method'] != 'cored' else 'adaptive token budget')
        config['protocol_basis'] += '; formal Setting B variant: 2 H100, batch 32 (1/GPU x accumulation 16); CoReD exact batch 32'
        configs.append(config)
    return configs


def completed_checkpoint(job):
    metadata = job.root / 'final_checkpoint.json'
    if not metadata.exists():
        return False
    checkpoint = Path(json.loads(metadata.read_text())['path'])
    manifest = json.loads((checkpoint / 'checkpoint_manifest.json').read_text())
    if any(sha(checkpoint / name) != expected for name, expected in manifest['files_sha256'].items()):
        raise ValueError(f'Completed checkpoint was modified: {checkpoint}')
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['preflight', 'train', 'eval', 'all', 'report'])
    p.add_argument('--gpus', default='0,1')
    p.add_argument('--assets', type=Path, default=ROOT / 'rebuttal/assets')
    p.add_argument('--data', type=Path, default=ROOT / 'rebuttal/prepared')
    p.add_argument('--runs', type=Path, default=ROOT / 'rebuttal/runs-B-2h100')
    p.add_argument('--reports', type=Path, default=ROOT / 'rebuttal/reports-B-2h100')
    p.add_argument('--run-id', nargs='+', help='Only B_*_7b_seed42 IDs; omit to run all five methods')
    args = p.parse_args()
    devices = args.gpus.split(',')
    if len(devices) != 2 or len(set(devices)) != 2 or any(not value.isdigit() for value in devices):
        p.error('Reserve two distinct H100 GPU indices, e.g. 0,1')
    configs = configurations(args.gpus)
    selected = [c for c in configs if args.run_id is None or c['run_id'] in args.run_id]
    if not selected or (args.run_id and set(args.run_id) != {c['run_id'] for c in selected}):
        p.error('Unknown Setting B run ID')
    if args.run_id and args.action in ('preflight', 'report'):
        p.error('Preflight/report cover all five Setting B runs; omit --run-id')
    assets, data, runs, reports = [path.resolve() for path in (args.assets, args.data, args.runs, args.reports)]
    if runs in {(ROOT/'rebuttal/runs').resolve(), (ROOT/'rebuttal/smoke-runs').resolve(),
                (ROOT/'rebuttal/smoke-2x3090').resolve()}:
        raise ValueError('Use separate formal H100 outputs, never B200/smoke outputs')
    if reports == (ROOT/'rebuttal/reports').resolve():
        raise ValueError('Use separate formal H100 reports')
    runs.mkdir(parents=True, exist_ok=True)
    marker = runs/'FORMAL_PROFILE.json'
    identity = {'profile': 'setting_b_2h100', 'gpus': args.gpus, 'global_batch': 32,
                'seed': 42, 'epochs': 3, 'data': 'real s1K-1.1 gemini fields; 1000 canonical input rows',
                'evaluation': 'four math benchmarks, 64 total responses/question, five eval seeds; no OOD'}
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise ValueError('Formal profile differs; choose a fresh runs directory')
    elif any(runs.iterdir()):
        raise ValueError('Choose an empty formal H100 runs directory')
    with (runs/'.gpu.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(marker, identity)
        for config in configs:
            write(runs/'configs'/(config['run_id']+'.json'), config)
        summary = {'profile': identity, 'action': args.action, 'status': 'running', 'completed': []}
        current = 'prepare/preflight'
        write(runs/'launcher_status.json', summary)
        try:
            if args.action == 'all':
                # CPU preparation, from attested real local snapshots; never fixtures.
                subprocess.run([sys.executable, '-m', 'rebuttal.data', '--assets', str(assets),
                                '--out', str(data)], cwd=ROOT, check=True)
            if args.action != 'report':
                validate(assets, data, runs, environments=True, gpus=args.gpus, formal_2h100=True)
            if args.action in ('train', 'eval', 'all'):
                for config in selected:
                    current = config['run_id']
                    print(f'[formal B / 2 H100] START {current}', flush=True)
                    job = Job(config, assets, data, runs, args.gpus)
                    if args.action in ('train', 'all'):
                        if completed_checkpoint(job):
                            print(f'[{current}] reuse completed training', flush=True)
                        else:
                            started = time.time()
                            status = 'failed'
                            try:
                                job.train()
                                status = 'complete'
                            finally:
                                with (job.root/'job_events.jsonl').open('a') as handle:
                                    handle.write(json.dumps({'wall_seconds': time.time()-started,
                                        'started_unix': started, 'status': status, 'allocated_gpu_count': 2})+'\n')
                                aggregate(job.root)
                    if args.action in ('eval', 'all'):
                        if not completed_checkpoint(job):
                            raise FileNotFoundError(f'Train first: {current}')
                        job.evaluate()
                    summary['completed'].append(current)
                    write(runs/'launcher_status.json', summary)
            if args.action == 'report' or (args.action == 'all' and not args.run_id):
                current = 'report'
                report(runs, reports, configurations=configs)
                tables(runs, reports, configurations=configs)
                write(reports/'formal_profile.json', identity)
            summary['status'] = 'complete'
        except BaseException as error:
            summary.update(status='failed', failed_run_or_stage=current, error=repr(error))
            raise
        finally:
            write(runs/'launcher_status.json', summary)
        print(f'[formal B / 2 H100] {args.action} complete; status: {runs / "launcher_status.json"}', flush=True)


if __name__ == '__main__':
    main()
