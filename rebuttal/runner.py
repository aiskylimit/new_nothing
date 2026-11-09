"""Native-method adapters and serial 15-run scheduling on one reserved GPU set."""
import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import time
import threading
import uuid
from pathlib import Path

from rebuttal.common import ROOT, matrix, rows, sha, signature, write
from rebuttal.cost import aggregate


def python_for(method):
    folder = {'longcot': 'sft', 'ssft': 'SegmentSelectiveSFT'}.get(method, method)
    value = os.environ.get('REBUTTAL_PYTHON_' + method.upper(), f'/mnt/local/uvenvs/{folder}/bin/python')
    if not Path(value).is_file():
        raise FileNotFoundError(f'Provision {folder}.txt or set REBUTTAL_PYTHON_{method.upper()}: {value}')
    return value


def checkpoint_manifest(path, config, base):
    path = Path(path)
    if not path.is_dir() or not list(path.glob('*.safetensors')):
        raise FileNotFoundError(f'Final checkpoint absent: {path}')
    for source in base.iterdir():
        if source.is_file() and (source.suffix in {'.json', '.jinja', '.model'} or source.name == 'merges.txt'):
            target = path / ('base_config.json' if source.name == 'config.json' else source.name)
            if not target.exists():
                shutil.copy2(source, target)
    adapter = path / 'adapter_config.json'
    if adapter.exists():
        value = json.loads(adapter.read_text())
        value['base_model_name_or_path'] = str(base)
        write(adapter, value)
    manifest = {'run_id': config['run_id'], 'model': config['model'], 'base_local_path': str(base),
                'files_sha256': {p.name: sha(p) for p in path.iterdir() if p.is_file() and p.name != 'checkpoint_manifest.json'},
                'kind': 'lora_with_immutable_local_base' if adapter.exists() else 'full_model',
                'load_requires': 'checkpoint and base snapshot only; no teacher/mask/train artifacts'}
    write(path / 'checkpoint_manifest.json', manifest)


class Job:
    def __init__(self, config, assets, data, runs, gpus):
        self.config, self.assets, self.data, self.gpus = config, assets, data, gpus
        self.root = runs / config['run_id']
        self.root.mkdir(parents=True, exist_ok=True)
        self.model = assets / 'models' / config['model']['repo'].split('/')[-1]
        self.input = data / config['data']['variant']
        if config.get('smoke'):
            from rebuttal.data import adapt
            small = self.root / 'smoke_data'
            adapt(rows(self.input / 'canonical.jsonl')[:2], small)
            self.input = small
        self.fingerprint = signature({'config': config, 'input': sha(self.input / 'canonical.jsonl'),
                                     'data_manifest': sha(data / 'manifest.json'),
                                     'code': {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / 'rebuttal').glob('*.py'))
                                              if p.name not in {'statistics.py', 'evaluate.py', 'cost.py', 'preflight.py', 'provision.py'}},
                                     'native_code': {str(p.relative_to(ROOT)): sha(p) for folder in
                                          ('cored', 'sgl/src', 'palign/src', 'SegmentSelectiveSFT/Attribution', 'SegmentSelectiveSFT/SelectiveSFT')
                                          for p in sorted((ROOT / folder).rglob('*.py'))}})
        saved = self.root / 'run.json'
        previous = json.loads(saved.read_text()) if saved.exists() else {}
        if saved.exists() and json.loads(saved.read_text())['fingerprint'] != self.fingerprint:
            raise ValueError(f'Configuration/data/code changed for {self.root}; choose another runs root')
        try:
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
            dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)
        except subprocess.CalledProcessError:
            commit, dirty = None, 'git unavailable'
        write(saved, dict(config, fingerprint=self.fingerprint, git_commit=commit, git_dirty=dirty,
                          canonical_data_sha256=sha(self.input / 'canonical.jsonl'),
                          environment=previous.get('environment', {'status': 'collected before training'})))

    def execute(self, name, commands, gpu_count, instrument=True):
        marker = self.root / f'{name}.complete.json'
        if marker.exists():
            print(f'[{self.config["run_id"]}] reuse {name}', flush=True)
            return
        attempt = name + '-' + uuid.uuid4().hex[:10]
        telemetry = self.root / 'telemetry' / attempt
        telemetry.mkdir(parents=True)
        (self.root / 'logs').mkdir(exist_ok=True)
        base_env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
                    'TRANSFORMERS_OFFLINE': '1', 'WANDB_MODE': 'disabled', 'TOKENIZERS_PARALLELISM': 'false',
                    'PYTHONPATH': os.pathsep.join([str(ROOT / 'sgl/src'), str(ROOT), str(ROOT / 'cored'), os.environ.get('PYTHONPATH', '')]),
                    'OMP_NUM_THREADS': '4', 'REBUTTAL_STAGE_DIR': str(telemetry)}
        if not instrument:
            base_env.pop('REBUTTAL_STAGE_DIR', None)
        for variable in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
            base_env.pop(variable, None)
        event = {'stage': name, 'attempt_id': attempt, 'gpu_count': gpu_count, 'gpus': self.gpus,
                 'started_unix': time.time(), 'commands': commands, 'status': 'failed'}
        started = time.perf_counter()
        stop = threading.Event()
        device_samples = []
        def monitor():
            while not stop.is_set():
                try:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,utilization.gpu',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
                    for line in result.stdout.splitlines():
                        fields = [x.strip() for x in line.split(',')]
                        if len(fields) == 4 and fields[0] in self.gpus.split(','):
                            device_samples.append({'time': time.time(), 'gpu': fields[0], 'name': fields[1],
                                                   'used_memory_mib': float(fields[2]), 'utilization': float(fields[3])})
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    pass
                stop.wait(1)
        observer = threading.Thread(target=monitor, daemon=True)
        observer.start()
        processes, handles, worker_starts, worker_ends = [], [], [], {}
        write(self.root / 'status.json', {'status': 'running', 'stage': name, 'attempt_id': attempt})
        try:
            for i, (command, gpu) in enumerate(commands):
                handle = (self.root / 'logs' / f'{attempt}-{i}.log').open('w')
                handles.append(handle)
                processes.append(subprocess.Popen(list(map(str, command)), cwd=ROOT,
                    env={**base_env, 'CUDA_VISIBLE_DEVICES': gpu}, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True))
                worker_starts.append(time.perf_counter())
            pending = set(range(len(processes)))
            while pending:
                for i in list(pending):
                    result = processes[i].poll()
                    if result is not None:
                        pending.remove(i)
                        worker_ends[i] = time.perf_counter()
                        if result:
                            raise RuntimeError(f'{name} worker {i} failed ({result}); see {self.root / "logs" / (attempt+"-"+str(i)+".log")}')
                if pending:
                    time.sleep(.2)
            event['status'] = 'complete'
            write(marker, {'fingerprint': self.fingerprint, 'attempt_id': attempt})
        except BaseException as error:
            event['error'] = repr(error)
            write(self.root / 'status.json', {'status': 'failed', 'stage': name, 'error': repr(error)})
            raise
        finally:
            stop.set()
            observer.join(timeout=6)
            write(telemetry / 'device_memory_samples.json', {'samples': device_samples,
                  'measurement': 'nvidia-smi device-used memory sampled each second; not torch allocated/reserved'})
            for process in processes:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
            for handle in handles:
                handle.close()
            event['seconds'] = time.perf_counter() - started
            event['workers'] = [{'seconds': worker_ends.get(i, time.perf_counter())-t,
                                 'gpu_count': len(commands[i][1].split(',')) if gpu_count else 0}
                                for i,t in enumerate(worker_starts)]
            event['gpu_hours'] = sum(w['seconds']*w['gpu_count']/3600 for w in event['workers'])
            with (self.root / 'stage_events.jsonl').open('a') as f:
                f.write(json.dumps(event) + '\n')
            aggregate(self.root)

    def instrument(self, python, module=None, script=None, args=(), multi=False, unsloth=False):
        command = [python]
        if multi:
            command += ['-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=3']
        command += ['-m', 'rebuttal.instrumented', '--module' if module else '--script', module or str(script)]
        if unsloth:
            command += ['--unsloth-first']
        return command + list(map(str, args))

    def train(self):
        config, run, method = self.config, self.root, self.config['method']
        python = python_for(method)
        train = config['train']
        environment_script = (
            "import torch,json,importlib.metadata as m; "
            "names=('torch','transformers','peft','accelerate','deepspeed','vllm','unsloth'); "
            "installed={d.metadata['Name'].lower():d.version for d in m.distributions()}; "
            "print(json.dumps({'cuda':torch.version.cuda,'packages':{n:installed.get(n) for n in names}}))")
        environment = json.loads(subprocess.check_output([python, '-c', environment_script], text=True))
        actual_config = json.loads((run / 'run.json').read_text())
        actual_config['environment'] = environment
        write(run / 'run.json', actual_config)
        gpus = self.gpus.split(',')
        if not (self.model / 'config.json').is_file():
            raise FileNotFoundError(self.model)
        # Native cost budgets remain explicit rather than forcing one trainer stack.
        if method == 'cored':
            command = [python, '-m', 'train.pipeline_b200', '--model', self.model,
                '--dataset', self.input / 'sgl/train.jsonl', '--gpus', self.gpus,
                '--output_root', run / 'artifacts', '--run_id', 'cored', '--epochs', train['epochs'],
                '--seed', train['seed'], '--global_batch', train['global_batch'], '--cutoff_len', train['cutoff'],
                '--lr_phase1', train['lr'], '--lr_distill', config['method_specific']['student_lr'],
                '--lora_r', train['lora_rank'], '--lora_alpha', train['lora_alpha'], '--lora_dropout', train['lora_dropout']]
            if config.get('smoke'):
                self.execute('cored_smoke_phase1', [(command + ['--dry_run', '2', '--stages', 'prepare', 'prefix', 'key', 'phase1'], self.gpus)], 3)
                self.execute('cored_smoke_student', [(command + ['--dry_run', '2', '--stages', 'weights', 'merge', 'distill'], self.gpus)], 3)
            else:
                self.execute('cored_pipeline', [(command, self.gpus)], 3)
            final = run / 'artifacts/cored/distill/student'
        elif method in {'longcot', 'palign'}:
            import yaml
            folder = run / 'lf_data'
            folder.mkdir(exist_ok=True)
            source = self.input / 'longcot.json'
            if method == 'palign':
                parts = [run / f'palign-{i}.jsonl' for i in range(3)]
                commands = [(self.instrument(python, module='rebuttal.method_prep', args=[
                    '--method', 'palign', '--model', self.model, '--source', self.input / 'canonical.jsonl',
                    '--output', parts[i], '--shard', i, '--shards', 3]), gpu) for i, gpu in enumerate(gpus)]
                self.execute('prefix_alignment', commands, 3)
                all_rows = [r for part in parts for r in rows(part)]
                expected = {r['id'] for r in rows(self.input / 'canonical.jsonl')}
                if {r['id'] for r in all_rows} != expected or len(all_rows) != len(expected):
                    raise ValueError('Incomplete/duplicate P-ALIGN preprocessing')
                kept = [r for r in all_rows if r['accepted']]
                if not kept:
                    raise ValueError('P-ALIGN correctness filter retained zero samples')
                write(run / 'adaptations.json', {'input_samples': len(expected), 'retained_samples': len(kept),
                      'rejected_ids': [r['id'] for r in all_rows if not r['accepted']], 'reason': 'P-ALIGN Eq.9 final-answer filtering'})
                source = run / 'palign.json'
                write(source, [{'instruction': 'Please reason step by step, and put your final answer within \\boxed{}.'+r['question'],
                      'input': '', 'output': r['prefix']+'\n'+r['continuation']} for r in kept])
            shutil.copy2(source, folder / 'train.json')
            write(folder / 'dataset_info.json', {'train': {'file_name': 'train.json'}})
            template_path = ROOT / ('sft/configs/qwen25_7b_longcot_sft.yaml' if method == 'longcot' else 'palign/configs/qwen25_7b_palign_sft.yaml')
            settings = yaml.safe_load(template_path.read_text())
            settings.update(model_name_or_path=str(self.model), template='deepseekr1' if config['setting'] == 'C' else 'qwen',
                dataset='train', dataset_dir=str(folder), output_dir=str(run / 'checkpoint'), overwrite_output_dir=False)
            settings.update(num_train_epochs=train['epochs'], seed=train['seed'], data_seed=train['seed'],
                per_device_train_batch_size=train['micro_batch'], gradient_accumulation_steps=train['gradient_accumulation'],
                cutoff_len=train['cutoff'], learning_rate=train['lr'], weight_decay=train['weight_decay'],
                warmup_ratio=train['warmup_ratio'], lr_scheduler_type=train['scheduler'],
                lora_rank=train['lora_rank'], lora_alpha=train['lora_alpha'], lora_dropout=train['lora_dropout'])
            if config.get('smoke'):
                settings['max_steps'] = 2
            checkpoints = sorted((run / 'checkpoint').glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]))
            if checkpoints:
                settings['resume_from_checkpoint'] = str(checkpoints[-1])
            path = run / 'native_train.yaml'
            path.write_text(yaml.safe_dump(settings, sort_keys=False))
            self.execute('optimization', [(self.instrument(python, script=ROOT / 'palign/src/train.py', args=[path], multi=True), self.gpus)], 3)
            final = run / 'checkpoint'
        elif method == 'sgl':
            masks = run / 'masks'
            self.execute('spectral_masks', [(self.instrument(python, module='sgl.build_masks', multi=True, args=[
                '--model-name-or-path', self.model, '--model-revision', config['model']['revision'],
                '--dataset-revision', config['data']['revision'],
                '--dataset-name', self.input / 'sgl/train.jsonl', '--output-dir', masks,
                '--max-samples', str(config['data']['expected_samples']), '--max-length', '32768', '--rank-threshold', '.95',
                '--selection-threshold', '.8', '--svd-device', 'cuda', '--svd-driver', 'gesvd', '--dtype', 'bfloat16']), self.gpus)], 3)
            args = ['--mask-dir', masks, '--output-dir', run / 'checkpoint', '--model-name-or-path', self.model,
                '--per-device-train-batch-size', '11', '--global-batch-size', '33', '--num-train-epochs', '3',
                '--learning-rate', '5e-5', '--warmup-ratio', '.1', '--gradient-checkpointing', '--seed', '42',
                '--lora-rank', '16', '--lora-alpha', '16', '--lora-dropout', '.05']
            checkpoints = sorted((run / 'checkpoint').glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]))
            if checkpoints:
                args += ['--resume-from-checkpoint', checkpoints[-1]]
            if config.get('smoke'):
                args += ['--max-steps', '2']
            self.execute('optimization', [(self.instrument(python, module='sgl.training', args=args, multi=True), self.gpus)], 3)
            final = run / 'checkpoint'
        else:
            selected = run / 'selected.jsonl'
            parts = [run / f'selected-{i}.jsonl' for i in range(3)]
            self.execute('attribution', [(self.instrument(python, module='rebuttal.method_prep', unsloth=True, args=[
                '--method', 'ssft', '--model', self.model, '--source', self.input / 'ssft.jsonl', '--output', parts[i],
                '--shard', i, '--shards', 3, '--config', run / 'run.json']), gpu) for i, gpu in enumerate(gpus)], 3)
            from rebuttal.method_prep import merge_ssft
            merge_ssft(self.input / 'ssft.jsonl', parts, selected)
            args = ['--model_name_or_path', self.model, '--data_names', selected, '--epochs', '3', '--seed', '42',
                '--learning_rate', '5e-5', '--warmup_ratio', '.1', '--lr_scheduler_type', 'cosine',
                '--optim', 'adamw_torch', '--weight_decay', '0', '--max_grad_norm', '1',
                '--per_device_train_batch_size', '11', '--gradient_accumulation_steps', '1',
                '--max_seq_length', '32768', '--save_strategy', 'epoch', '--save_total_limit', '2',
                '--output_dir', run / 'checkpoint', '--segment_mode', 'paragraph', '--think_prefix', 'none', '--mask']
            checkpoints = sorted((run / 'checkpoint').glob('checkpoint-[0-9]*'), key=lambda p: int(p.name.split('-')[-1]))
            if checkpoints:
                args += ['--resume_from_checkpoint', checkpoints[-1]]
            if config.get('smoke'):
                args += ['--max_steps', '2']
            self.execute('optimization', [(self.instrument(python, script=ROOT / 'SegmentSelectiveSFT/SelectiveSFT/train_mask.py',
                args=args, multi=True, unsloth=True), self.gpus)], 3)
            final = run / 'checkpoint/checkpoint-final'
        checkpoint_manifest(final, config, self.model)
        write(run / 'training_budget.json', {'canonical_input_samples': config['data']['expected_samples'],
            'method': method, 'teacher': config['data'], 'precision': train['precision'], 'batch': train,
            'sampler_note': 'Native distributed samplers may pad/repeat the final batch; common telemetry records actual forward token exposures.',
            'ssft_representation': 'reasoning-only solution plus native trace/GT attribution target; explicit preprocessing audit',
            'cored_representation': 'reasoning + teacher attempt; prefix/key/council artifacts retained'})
        write(run / 'final_checkpoint.json', {'path': str(final), 'run_id': config['run_id']})
        write(run / 'status.json', {'status': 'smoke_trained' if config.get('smoke') else 'trained', 'checkpoint': str(final)})

    def evaluate(self):
        checkpoint = json.loads((self.root / 'final_checkpoint.json').read_text())['path']
        python = python_for('palign')
        output = self.root / 'evaluation'
        commands = [([python, '-m', 'rebuttal.evaluate', '--config', self.root / 'run.json', '--checkpoint', checkpoint,
                     '--benchmarks', self.data / 'benchmarks', '--output', output, '--shard', str(i), '--shards', '3'], gpu)
                    for i, gpu in enumerate(self.gpus.split(','))]
        # Evaluation has separate events; it is never counted as training cost.
        self.execute('evaluation_generate', commands, 3, instrument=False)
        self.execute('evaluation_score', [([python, '-m', 'rebuttal.evaluate', '--config', self.root / 'run.json',
            '--benchmarks', self.data / 'benchmarks', '--output', output, '--score-only'], '')], 0, instrument=False)
        write(self.root / 'status.json', {'status': 'evaluated', 'checkpoint': checkpoint})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['configs', 'train', 'eval', 'all', 'smoke'])
    p.add_argument('--assets', type=Path, default=ROOT / 'rebuttal/assets')
    p.add_argument('--data', type=Path, default=ROOT / 'rebuttal/prepared')
    p.add_argument('--runs', type=Path, default=ROOT / 'rebuttal/runs')
    p.add_argument('--gpus', default='3,4,5')
    p.add_argument('--run-id', nargs='+')
    args = p.parse_args()
    args.assets, args.data, args.runs = args.assets.resolve(), args.data.resolve(), args.runs.resolve()
    if len(args.gpus.split(',')) != 3 or len(set(args.gpus.split(','))) != 3:
        p.error('Reserve three distinct GPUs')
    configs = matrix()
    if args.action == 'configs':
        for config in configs:
            write(ROOT / 'rebuttal/configs' / (config['run_id'] + '.json'), config)
        return
    for expected in configs:
        saved = json.loads((ROOT / 'rebuttal/configs' / (expected['run_id']+'.json')).read_text())
        if saved != expected:
            raise ValueError('Frozen main config differs from the shared protocol; update matrix and regenerate configs explicitly')
    selected = [c for c in configs if args.run_id is None or c['run_id'] in args.run_id]
    if args.run_id and set(args.run_id) != {c['run_id'] for c in selected}:
        p.error('Unknown run ID')
    args.runs.mkdir(parents=True, exist_ok=True)
    with (args.runs / '.gpu.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from rebuttal.preflight import validate
        validate(args.assets, args.data, args.runs, environments=args.action != 'eval', gpus=args.gpus)
        for config in selected:
            if args.action == 'smoke':
                config = dict(config, run_id='smoke_'+config['run_id'], smoke=True,
                              data=dict(config['data'], expected_samples=2))
            job = Job(config, args.assets.resolve(), args.data.resolve(), args.runs.resolve(), args.gpus)
            started = time.time()
            if args.action in {'train', 'all', 'smoke'}:
                status = 'failed'
                final = job.root / 'final_checkpoint.json'
                if final.exists():
                    checkpoint = Path(json.loads(final.read_text())['path'])
                    provenance = json.loads((checkpoint / 'checkpoint_manifest.json').read_text())
                    if any(sha(checkpoint / file) != checksum for file, checksum in provenance['files_sha256'].items()):
                        raise ValueError('Completed checkpoint was modified')
                    print(f'[{config["run_id"]}] reuse completed training; no extra training cost charged', flush=True)
                else:
                    try:
                        job.train()
                        status = 'complete'
                    except BaseException as error:
                        write(job.root / 'status.json', {'status': 'failed', 'error': repr(error)})
                        raise
                    finally:
                        with (job.root / 'job_events.jsonl').open('a') as f:
                            f.write(json.dumps({'wall_seconds': time.time()-started, 'started_unix': started,
                                                'status': status, 'allocated_gpu_count': 3})+'\n')
                        aggregate(job.root)
            if args.action in {'eval', 'all'}:
                job.evaluate()


if __name__ == '__main__':
    main()
