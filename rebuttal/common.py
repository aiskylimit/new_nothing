import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATH = ('aime24', 'aime25', 'amc12', 'math500')
OOD = ('arc_challenge', 'logiqa2')
SEEDS = (1, 2, 3, 4, 5)
BUDGETS = (13, 13, 13, 13, 12)
MODELS = {
    '7b': ('Qwen/Qwen2.5-7B-Instruct', 'a09a35458c702b33eeacc393d103063234e8bc28'),
    '1.5b': ('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', 'ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562'),
}
DATA_REV = '96c411f1fe4c49d20f0e2a1565f61e1a28b0b84d'
GEMINI_REV = '278d72baaa2b887a7e76a70a0ae254a5a45536e4'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.')
    with os.fdopen(fd, 'w') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, default=str)
        f.write('\n')
    os.replace(temporary, path)


def rows(path, repair=False):
    """Recover only an incomplete final JSONL record; interior corruption is fatal."""
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines(keepends=True)
    result, valid_bytes = [], 0
    for index, line in enumerate(lines):
        if not line.strip():
            valid_bytes += len(line)
            continue
        try:
            result.append(json.loads(line))
        except (ValueError, UnicodeDecodeError):
            if repair and index == len(lines) - 1 and not line.endswith(b'\n'):
                with path.open('r+b') as f:
                    f.truncate(valid_bytes)
                break
            raise
        valid_bytes += len(line)
        if repair and index == len(lines)-1 and not line.endswith(b'\n'):
            with path.open('ab') as f:
                f.write(b'\n')
    return result


def jsonl(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        for value in values:
            f.write(json.dumps(value, ensure_ascii=False) + '\n')


def matrix():
    result = []
    methods = ('longcot', 'sgl', 'ssft', 'palign', 'cored')
    for setting, size, teacher in [('A', '7b', 'original'), ('B', '7b', 'gemini'), ('C', '1.5b', 'original')]:
        for method in methods:
            model, revision = MODELS[size]
            result.append({
                'run_id': f'{setting}_{method}_{size}_seed42', 'setting': setting, 'method': method,
                'model': {'repo': model, 'revision': revision, 'tokenizer': model, 'size': size},
                'data': {'repo': 'simplescaling/s1K-1.1' if teacher == 'original' else 'simplescaling/s1K',
                         'revision': DATA_REV if teacher == 'original' else GEMINI_REV, 'split': 'train',
                         'expected_samples': 1000, 'teacher': 'DeepSeek-R1' if teacher == 'original' else 'Gemini Flash Thinking',
                         'trace_source': 'deepseek_thinking_trajectory' if teacher == 'original' else 'thinking_trajectories[0]',
                         'answer_source': 'deepseek_attempt' if teacher == 'original' else 'attempt', 'variant': teacher},
                'train': {'seed': 42, 'epochs': 3, 'global_batch': 33, 'micro_batch': 11,
                          'gradient_accumulation': 1 if method != 'cored' else 'adaptive token budget',
                          'optimizer': 'AdamW', 'lr': 5e-5, 'weight_decay': 0,
                          'scheduler': 'cosine', 'warmup_ratio': 0.1, 'precision': 'bf16',
                          'cutoff': 16384 if method == 'cored' else 32768,
                          'truncation': 'method_native_recorded_in_adaptations',
                          'lora_rank': 16, 'lora_alpha': 16, 'lora_dropout': 0.05},
                'method_specific': {'ssft': {'ig_steps': 50, 'mass': .7, 'coherence': .8},
                    'sgl': {'rank_threshold': .95, 'selection_threshold': .8},
                    'palign': {'prefix': 'existing_sentence_binary_self_judgment', 'continuation_max_tokens': 32768,
                               'filter': 'verified_continuation', 'filter_is_method_specific': True},
                    'cored': {'M': 3, 'p_drop': .1, 'lambda_rep': .01, 'key_frac': .1,
                              'diversity': 'svgd_func', 'warmup_div_epochs': 1, 'lambda_kl': 1,
                              'student_lr': 1e-4, 'merge': 'ties', 'density': .2}}.get(method, {}),
                'evaluation': {'math': list(MATH), 'ood': list(OOD) if setting != 'B' else [],
                               'seeds': list(SEEDS), 'budgets': list(BUDGETS), 'math_max_tokens': 4096,
                               'temperature': .6, 'top_p': .9, 'repetition_penalty': 1.05,
                               'ood_temperature': 0, 'arc_max_tokens': 1024, 'logiqa_max_tokens': 4096},
                'hardware': {'gpus': '3,4,5', 'type': 'NVIDIA B200 180GB', 'training_gpu_count': 3},
                'protocol_basis': 'current rebuttal implementations; differences retained, not a paper-equivalence certification',
            })
    return result
