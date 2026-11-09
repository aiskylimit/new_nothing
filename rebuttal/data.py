"""One question/teacher manifest, explicit adapters, immutable benchmark IDs."""
import argparse
import unicodedata
from pathlib import Path

import numpy as np

from rebuttal.common import DATA_REV, TEACHER_FIELDS, jsonl, rows, sha, signature, write


def question_key(text):
    # The pinned source has two distinct rows whose questions differ only in
    # interior whitespace. Preserve those rows and their separate trajectories.
    return unicodedata.normalize('NFC', text).strip()


def snapshot(path, split='train'):
    import pyarrow.parquet as pq
    path = Path(path)
    if path.is_file():
        return pq.read_table(path).to_pylist() if path.suffix == '.parquet' else rows(path)
    search_root = path / 'data' if (path / 'data').is_dir() else path
    files = sorted(search_root.rglob(f'{split}*.parquet'))
    if not files:
        files = sorted(search_root.rglob(f'{split}*.jsonl'))
    if not files:
        raise FileNotFoundError(f'Missing local {split} snapshot: {path}')
    return [row for file in files for row in snapshot(file)]


def normalize_teachers(source):
    """Produce both teacher views from the same s1K-1.1 row, without fallbacks."""
    output = {'original': [], 'gemini': []}
    seen = set()
    for row in source:
        key = question_key(row['question'])
        if not key or key in seen:
            raise ValueError('s1K-1.1 questions must be nonempty and unique')
        seen.add(key)
        for variant, (trace_field, answer_field, grade_field) in TEACHER_FIELDS.items():
            thinking, attempt = row.get(trace_field), row.get(answer_field)
            if not isinstance(thinking, str) or not thinking.strip() or not isinstance(attempt, str) or not attempt.strip():
                raise ValueError(f'Missing {variant} teacher fields: {trace_field}, {answer_field}')
            output[variant].append({'id': signature(key)[:24], 'question': row['question'].strip(),
                'thinking': thinking.strip(), 'answer_text': attempt.strip(), 'gold': str(row['solution']).strip(),
                'source_type': row.get('source_type'), 'teacher_grade': row.get(grade_field),
                'teacher': variant})
    return output


def adapt(values, destination):
    destination = Path(destination)
    # All adapters derive from these same IDs; no correctness/length filtering here.
    jsonl(destination / 'canonical.jsonl', values)
    jsonl(destination / 'sgl/train.jsonl', [dict(row, deepseek_thinking_trajectory=row['thinking'],
          deepseek_attempt=row['answer_text'], solution=row['gold']) for row in values])
    jsonl(destination / 'ssft.jsonl', [dict(id=row['id'], question=row['question'],
          solution=row['thinking'], answer=row['gold']) for row in values])
    jsonl(destination / 'palign_input.jsonl', [dict(id=row['id'], question=row['question'],
          **{'Long-CoT': row['thinking']}, answer=row['gold']) for row in values])
    instruction = 'Please reason step by step, and put your final answer within \\boxed{}.'
    write(destination / 'longcot.json', [{'instruction': instruction + row['question'], 'input': '',
          'output': '<think>\n' + row['thinking'] + '\n</think>\n\n' + row['answer_text']} for row in values])
    write(destination / 'dataset_info.json', {'longcot': {'file_name': 'longcot.json'}, 'palign': {'file_name': 'palign.json'}})


def teacher_statistics(values, tokenizer, grade):
    lengths, correctness = [], []
    for row in values:
        lengths.append(len(tokenizer.encode(row['thinking'] + '\n\n' + row['answer_text'], add_special_tokens=False)))
        correctness.append(grade(row['answer_text'], row['gold'])[0])
    return {'samples': len(values), 'tokens': {'mean': float(np.mean(lengths)), 'p50': float(np.percentile(lengths, 50)),
            'p95': float(np.percentile(lengths, 95)), 'max': max(lengths)},
            'correctness_math_verify': float(np.mean(correctness)),
            'correctness_limitation': 'math-verify is not a proof grader; unparseable proof/text targets count as incorrect',
            'per_question': [{'id': row['id'], 'tokens': length, 'is_correct': correct}
                             for row, length, correct in zip(values, lengths, correctness)]}


def mc_item(benchmark, index, row):
    if benchmark == 'arc_challenge':
        labels = [str(x) for x in row['choices']['label']]
        options = row['choices']['text']
        if len(labels) != len(set(labels)) or len(options) != len(labels):
            raise ValueError('Malformed ARC choices')
        answer_index = labels.index(str(row['answerKey']))
        question = row['question']
    else:
        options = row['options']
        if len(options) != 4 or not isinstance(row['answer'], int) or not 0 <= row['answer'] < 4:
            raise ValueError('Malformed English MRC LogiQA2 record')
        answer_index = row['answer']
        question = row['text'] + '\n\n' + row['question']
    # Canonical display letters preserve ARC's original (including numeric) label mapping.
    letters = list('ABCDE')[:len(options)]
    question += '\n\n' + '\n'.join(f'{label}. {text}' for label, text in zip(letters, options))
    return {'question_id': f'{benchmark}:{index}:{row.get("id", index)}', 'source_question_id': row.get('id'), 'question': question,
            'answer': letters[answer_index], 'choices': letters,
            'source_choice_labels': labels if benchmark == 'arc_challenge' else list(range(4))}


def prepare_benchmarks(store, destination):
    from rebuttal.provision import BENCHMARKS, URLS
    store, destination = Path(store), Path(destination)
    specifications = {'aime24': ('AIME_2024', 'train', 30), 'aime25': ('aime_2025', 'train', 30),
                      'amc12': ('aimo-validation-amc', 'train', 83), 'math500': ('MATH-500', 'test', 500)}
    metadata = {}
    for name, (folder, split, count) in specifications.items():
        source = store / folder
        if name == 'aime24':
            source /= 'aime_2024_problems.parquet'
        raw = snapshot(source, split)
        records = []
        for i, row in enumerate(raw):
            lower = {key.lower(): value for key, value in row.items()}
            question = lower.get('problem') or lower.get('question')
            answer = lower.get('answer')
            if not isinstance(question, str) or not question.strip() or answer is None or str(answer) == '':
                raise ValueError(f'{name}: missing problem/answer fields at row {i}')
            records.append({'question_id': f'{name}:{i}', 'question': question,
                            'answer': str(answer).removesuffix('.0')})
        if len(records) != count or len({r['question'] for r in records}) != count:
            raise ValueError(f'{name}: expected {count} unique questions')
        jsonl(destination / f'{name}.jsonl', records)
        repo, revision = BENCHMARKS[folder]
        metadata[name] = {'count': count, 'sha256': sha(destination / f'{name}.jsonl'), 'source': repo,
                          'revision': revision, 'split': split}
    for name, path, count in [('arc_challenge', store / 'ARC-Challenge/test-00000-of-00001.parquet', 1172),
                              ('logiqa2', store / 'LogiQA2.0/test.txt', 1572)]:
        records = [mc_item(name, i, r) for i, r in enumerate(snapshot(path))]
        if len(records) != count or len({r['question_id'] for r in records}) != count:
            raise ValueError(f'{name}: expected {count} unique questions')
        jsonl(destination / f'{name}.jsonl', records)
        metadata[name] = {'count': count, 'source_sha256': sha(path), 'sha256': sha(destination / f'{name}.jsonl'),
                          'source_url': URLS[str(path.relative_to(store))][0]}
    write(destination / 'manifest.json', metadata)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--assets', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--skip-statistics', action='store_true', help='Fixture/data-only checks; server preflight still requires statistics')
    args = p.parse_args()
    from rebuttal.provision import BENCHMARKS, URLS, verify_snapshot
    source = args.assets / 'datasets/s1K-1.1'
    verify_snapshot(source, 'simplescaling/s1K-1.1', DATA_REV)
    paired = normalize_teachers(snapshot(source))
    for folder, (repo, revision) in BENCHMARKS.items():
        verify_snapshot(args.assets / 'benchmarks' / folder, repo, revision)
    for relative, (_, expected) in URLS.items():
        if sha(args.assets / 'benchmarks' / relative) != expected:
            raise ValueError('OOD snapshot changed')
    if len(paired['original']) != 1000:
        raise ValueError('Expected 1000 matched training questions')
    for variant, values in paired.items():
        adapt(values, args.out / variant)
    write(args.out / 'manifest.json', {'revisions': {'original': DATA_REV, 'gemini': DATA_REV},
          'dataset': {'repo': 'simplescaling/s1K-1.1', 'revision': DATA_REV, 'split': 'train'},
          'teacher_fields': TEACHER_FIELDS,
          'counts': {k: len(v) for k, v in paired.items()}, 'question_ids': [r['id'] for r in paired['original']],
          'files': {v: sha(args.out / v / 'canonical.jsonl') for v in paired}})
    prepare_benchmarks(args.assets / 'benchmarks', args.out / 'benchmarks')
    if not args.skip_statistics:
        from transformers import AutoTokenizer
        from rebuttal.common import MODELS
        from rebuttal.evaluate import grade_math
        for size, variant in [('7b', 'original'), ('7b', 'gemini'), ('1.5b', 'original')]:
            repo, revision = MODELS[size]
            tokenizer = AutoTokenizer.from_pretrained(str(args.assets / 'models' / repo.split('/')[-1]), local_files_only=True)
            statistics = teacher_statistics(paired[variant], tokenizer, grade_math)
            statistics.update(model=repo, model_revision=revision, data_sha256=sha(args.out / variant / 'canonical.jsonl'))
            write(args.out / 'statistics' / f'{variant}_{size}.json', statistics)


if __name__ == '__main__':
    main()
