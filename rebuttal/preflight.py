"""Offline validation of the complete experiment grid; GPU checks run on server."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

from rebuttal.common import DATA_REV, ROOT, TEACHER_FIELDS, matrix, rows, sha, write


def validate(assets, data, runs, environments=False, gpus='3,4,5', formal_2h100=False):
    expected_devices = 2 if formal_2h100 else 3
    gpu_type = 'H100' if formal_2h100 else 'B200'
    if len(gpus.split(',')) != expected_devices or len(set(gpus.split(','))) != expected_devices:
        raise ValueError(f'Reserve {expected_devices} distinct GPUs')
    checks = {'configurations': 5 if formal_2h100 else 15, 'training_seed_count': 1, 'math_seeds': [1, 2, 3, 4, 5],
              'math_responses': 64, 'gpu_ids': gpus, 'checks': []}
    manifest = json.loads((data / 'manifest.json').read_text())
    if (manifest.get('dataset') != {'repo': 'simplescaling/s1K-1.1', 'revision': DATA_REV, 'split': 'train'}
            or manifest.get('revisions') != {'original': DATA_REV, 'gemini': DATA_REV}
            or manifest.get('teacher_fields') != {k: list(v) for k, v in TEACHER_FIELDS.items()}):
        raise ValueError('Stale prepared data: rerun prepare to build both teachers from the same s1K-1.1 snapshot')
    original = rows(data / 'original/canonical.jsonl')
    gemini = rows(data / 'gemini/canonical.jsonl')
    if len(original) != 1000 or [r['id'] for r in original] != [r['id'] for r in gemini]:
        raise ValueError('Expected 1000 matched Original/Gemini question IDs')
    for variant in ('original', 'gemini'):
        if sha(data / variant / 'canonical.jsonl') != manifest['files'][variant]:
            raise ValueError('Prepared teacher source changed')
    checks['checks'].append('1000 matched questions, explicit teacher fields, canonical hashes')
    for variant, size in [('original', '7b'), ('gemini', '7b'), ('original', '1.5b')]:
        statistics = json.loads((data / 'statistics' / f'{variant}_{size}.json').read_text())
        if statistics['data_sha256'] != sha(data / variant / 'canonical.jsonl'):
            raise ValueError('Teacher statistics do not match canonical training data')
    for size, model in {c['model']['size']: c['model'] for c in matrix()}.items():
        folder = assets / 'models' / model['repo'].split('/')[-1]
        if not (folder / 'config.json').is_file() or not (folder / 'tokenizer_config.json').is_file():
            raise FileNotFoundError(folder)
        from rebuttal.provision import verify_snapshot
        verify_snapshot(folder, model['repo'], model['revision'])
    checks['checks'].append('both immutable model snapshots hash-verified')
    benchmarks = json.loads((data / 'benchmarks/manifest.json').read_text())
    for name, info in benchmarks.items():
        if sha(data / 'benchmarks' / f'{name}.jsonl') != info['sha256']:
            raise ValueError('Benchmark changed')
    checks['checks'].append('4 math and 2 OOD benchmark hashes')
    if environments:
        from rebuttal.runner import python_for
        for method in ('longcot', 'sgl', 'ssft', 'palign', 'cored'):
            py = python_for(method)
            imports = ('import unsloth; ' if method == 'ssft' else '') + 'import torch,transformers,peft; print(torch.__version__,transformers.__version__)'
            subprocess.run([py, '-c', imports], check=True)
        import os
        subprocess.run([python_for('cored'), '-m', 'train.hardware_probe', '--out', str(runs / 'hardware.json'),
                        '--expected_devices', str(expected_devices), '--required_gpu_name', gpu_type],
            cwd=ROOT / 'cored', env={**os.environ, 'CUDA_VISIBLE_DEVICES': gpus}, check=True)
        hardware = json.loads((runs / 'hardware.json').read_text())
        if len(hardware['devices']) != expected_devices or any(gpu_type not in device['name'] for device in hardware['devices']):
            raise ValueError(f'Expected {expected_devices} {gpu_type} GPUs; hardware mismatch recorded')
        checks['checks'].append(f'five provisioned environments and {expected_devices} {gpu_type} CUDA/NCCL/bf16 GPUs')
        from rebuttal.eval_preflight import check_eval
        check_eval(assets, runs, gpus, sizes=('7b',) if formal_2h100 else ('1.5b','7b'))
        checks['checks'].append('separate shared evaluator: CUDA kernel and zero-LoRA generation on every reserved GPU')
    checks['persistent_storage'] = {'runs': str(runs.resolve()), 'free_bytes': shutil.disk_usage(runs).free,
                                  'note': 'Required space depends on generated trace lengths; free space is reported, not assumed sufficient.'}
    checks['gpu_training_smoke'] = 'not executed by static preflight; use runner smoke in a separate runs root'
    write(runs / 'preflight.json', checks)
    return checks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--environments', action='store_true')
    p.add_argument('--gpus', default='3,4,5')
    args = p.parse_args()
    args.assets, args.data, args.runs = args.assets.resolve(), args.data.resolve(), args.runs.resolve()
    args.runs.mkdir(parents=True, exist_ok=True)
    print(json.dumps(validate(args.assets, args.data, args.runs, args.environments, args.gpus), indent=2))


if __name__ == '__main__':
    main()
