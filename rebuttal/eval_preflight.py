"""Launch shared vLLM/LoRA probes on each reserved GPU before native training."""
import argparse
import os
import signal
import subprocess
import time
from pathlib import Path

from rebuttal.common import MODELS, ROOT


def check_eval(assets, runs, gpus, sizes=('1.5b','7b')):
    from rebuttal.runner import python_for
    python = python_for('eval')
    directory = Path(runs)/'eval-preflight'
    directory.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, 'HF_HUB_OFFLINE':'1', 'HF_DATASETS_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1',
                   'WANDB_MODE':'disabled', 'TOKENIZERS_PARALLELISM':'false', 'OMP_NUM_THREADS':'1',
                   'PYTHONPATH':os.pathsep.join([str(ROOT), os.environ.get('PYTHONPATH','')])}
    for key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT','REBUTTAL_STAGE_DIR'):
        environment.pop(key, None)
    for size in sizes:
        model = Path(assets)/'models'/MODELS[size][0].split('/')[-1]
        processes, logs = [], []
        try:
            for rank, gpu in enumerate(gpus.split(',')):
                output = directory/f'{size}-gpu{gpu}.json'
                log_path = directory/f'{size}-gpu{gpu}.log'
                log = log_path.open('w')
                logs.append(log)
                print(f'[eval preflight] {size}, GPU {gpu}, {python}; log: {log_path}', flush=True)
                command = [python, '-m', 'rebuttal.eval_probe', '--model', str(model), '--out', str(output)]
                processes.append(subprocess.Popen(command, cwd=ROOT, env={**environment,'CUDA_VISIBLE_DEVICES':gpu},
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
            pending = set(range(len(processes)))
            deadline = time.monotonic()+900
            while pending:
                for rank in list(pending):
                    code = processes[rank].poll()
                    if code is not None:
                        pending.remove(rank)
                        if code:
                            raise RuntimeError(f'Shared evaluator failed on GPU {gpus.split(",")[rank]}; see {directory}')
                if time.monotonic() > deadline:
                    raise TimeoutError(f'Shared evaluator probe timed out; see {directory}')
                if pending:
                    time.sleep(.2)
        finally:
            for process in processes:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
            for log in logs:
                log.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--gpus', required=True)
    p.add_argument('--sizes', nargs='+', choices=list(MODELS), default=['1.5b','7b'])
    args = p.parse_args()
    check_eval(args.assets.resolve(), args.runs.resolve(), args.gpus, args.sizes)


if __name__ == '__main__':
    main()
