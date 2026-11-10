"""Evaluate a verified saved checkpoint without resuming/restarting training."""
import argparse
import json
from pathlib import Path

from rebuttal.common import ROOT, write
from rebuttal.eval_preflight import check_eval
from rebuttal.formal_b_2h100 import completed_checkpoint
from rebuttal.runner import Job


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--data', type=Path, default=ROOT/'rebuttal/prepared')
    p.add_argument('--gpus', required=True)
    p.add_argument('--fresh-output', action='store_true', help='Keep old eval intact; write under <run>/eval-only-shared/')
    args = p.parse_args()
    root = args.run_dir.resolve()
    config = json.loads((root/'run.json').read_text())
    count = config['hardware']['training_gpu_count']
    if len(args.gpus.split(',')) != count or len(set(args.gpus.split(','))) != count:
        p.error(f'Reserve {count} distinct GPUs for this saved run')
    # Intentionally bypass only the *training* code fingerprint: there is no train
    # action here. Checkpoint bytes are verified before loading the saved adapter.
    job = object.__new__(Job)
    job.root = root
    if not completed_checkpoint(job):
        raise FileNotFoundError('No completed training checkpoint to evaluate')
    checkpoint = json.loads((root/'final_checkpoint.json').read_text())
    provenance = json.loads((Path(checkpoint['path'])/'checkpoint_manifest.json').read_text())
    base = Path(provenance['base_local_path'])
    assets = base.parent.parent
    check_eval(assets, root/'eval-recovery-preflight', args.gpus, sizes=(config['model']['size'],))
    if args.fresh_output:
        job.root = root/'eval-only-shared'
        write(job.root/'run.json', config)
        write(job.root/'final_checkpoint.json', checkpoint)
    job.config, job.gpus = config, args.gpus
    job.fingerprint = config.get('fingerprint', 'verified-checkpoint-eval-only')
    job.data = root.parent/'fixtures' if config.get('development_smoke') else args.data.resolve()
    job.evaluate()
    print(f'Checkpoint evaluated without training: {job.root / "evaluation"}', flush=True)


if __name__ == '__main__':
    main()
