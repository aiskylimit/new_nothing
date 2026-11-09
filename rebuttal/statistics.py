"""Question-level paired bootstrap, shared Avg@32/64 samples and failure exports."""
import argparse
import math
from pathlib import Path

import numpy as np

from rebuttal.common import BUDGETS, MATH, SEEDS, jsonl, matrix, rows, sha, write


def sample_order():
    return [(seed, index) for index in range(max(BUDGETS))
            for seed, budget in zip(SEEDS, BUDGETS) if index < budget]


def question_metrics(records, math_benchmark):
    grouped = {}
    for row in records:
        grouped.setdefault(row['question_id'], []).append(row)
    output = {}
    for question, responses in grouped.items():
        lookup = {(r['eval_seed'], r['sample_id']): r for r in responses}
        order = sample_order() if math_benchmark else [(0, 0)]
        if len(lookup) != len(responses) or set(lookup) != set(order):
            raise ValueError('Incomplete/duplicated seed/sample grid')
        if any(not isinstance(r['is_correct'], bool) or r['finish_reason'] == 'error' for r in responses):
            raise ValueError('Unscored/error response')
        correct = np.array([lookup[item]['is_correct'] for item in order], dtype=float)
        metrics = {'pass@1': float(correct.mean())}
        if math_benchmark:
            c = int(correct.sum())
            metrics.update({'Avg@32': float(correct[:32].mean()), 'Avg@64': float(correct.mean()),
                            'pass@3': 1 - math.comb(64 - c, 3) / math.comb(64, 3)})
        output[question] = metrics
    return output


def bootstrap(values, repetitions=5000, seed=2027):
    """A vector can be a paired per-question difference; never resample responses."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        raise ValueError('Empty bootstrap sample')
    rng = np.random.default_rng(seed)
    # Chunk to bound memory for LogiQA and larger repetition counts.
    estimates = np.concatenate([values[rng.integers(0, len(values), (min(128, repetitions - start), len(values)))].mean(1)
                                for start in range(0, repetitions, 128)])
    return estimates


def interval(values, repetitions=5000):
    distribution = bootstrap(values, repetitions)
    return {'mean': float(np.mean(values)), 'ci95': np.quantile(distribution, [.025, .975]).tolist(),
            'unit': 'fraction', 'resampling_unit': 'question', 'bootstrap_repetitions': repetitions}


def paired(left, right, metric, repetitions=5000):
    if set(left) != set(right):
        raise ValueError('Paired comparison needs exactly matching question IDs')
    return interval([left[key][metric] - right[key][metric] for key in sorted(left)], repetitions)


def describe(records):
    lengths = [r['generated_tokens'] for r in records]
    seed_means = [float(np.mean([r['is_correct'] for r in records if r['eval_seed'] == seed]))
                  for seed in sorted({r['eval_seed'] for r in records})]
    return {'invalid_answer_rate': float(np.mean([r['invalid_answer'] for r in records])),
            'truncation_rate': float(np.mean([r['finish_reason'] == 'length' for r in records])),
            'mean_generated_tokens': float(np.mean(lengths)), 'p95_generated_tokens': float(np.percentile(lengths, 95)),
            'seed_accuracy_std_not_training_variance': float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else None,
            'eval_seed_accuracy': {str(seed): float(np.mean([r['is_correct'] for r in records if r['eval_seed'] == seed]))
                                   for seed in sorted({r['eval_seed'] for r in records})}}


def report(root, destination, repetitions=5000, allow_partial=False):
    root, destination = Path(root), Path(destination)
    configs = matrix()
    summaries, per_question, missing = {}, {}, []
    for config in configs:
        run = config['run_id']
        folder = root / run / 'evaluation'
        expected = config['evaluation']['math'] + config['evaluation']['ood']
        if not all((folder / f'{name}.scored.jsonl').exists() for name in expected):
            missing.append(run)
            continue
        summaries[run], per_question[run] = {}, {}
        for benchmark in expected:
            records = rows(folder / f'{benchmark}.scored.jsonl')
            if any('run_id' in record and record['run_id'] != run for record in records):
                raise ValueError('Scored responses belong to a different checkpoint run')
            points = question_metrics(records, benchmark in MATH)
            if not points:
                raise ValueError('Empty benchmark')
            per_question[run][benchmark] = points
            expected_count = {'aime24': 30, 'aime25': 30, 'amc12': 83, 'math500': 500,
                              'arc_challenge': 1172, 'logiqa2': 1572}[benchmark]
            if len(points) != expected_count:
                raise ValueError(f'{run}/{benchmark}: wrong question count')
            summary = {metric: interval([r[metric] for r in points.values()], repetitions)
                       for metric in next(iter(points.values()))}
            summary['diagnostics'] = describe(records)
            summaries[run][benchmark] = summary
            jsonl(destination / 'failures' / f'{run}_{benchmark}.jsonl',
                  (r for r in records if not r['is_correct'] or r['invalid_answer'] or r['finish_reason'] == 'length'))
        summaries[run]['macro_math'] = {}
        for metric in ('Avg@32', 'Avg@64', 'pass@1', 'pass@3'):
            distributions = [bootstrap([r[metric] for r in per_question[run][b].values()], repetitions, 2027 + i)
                             for i, b in enumerate(MATH)]
            samples = np.mean(distributions, axis=0)
            summaries[run]['macro_math'][metric] = {'mean': float(np.mean([summaries[run][b][metric]['mean'] for b in MATH])),
                'ci95': np.quantile(samples, [.025, .975]).tolist(), 'resampling': 'stratified_by_benchmark_questions; equal benchmark weights'}
    if missing and not allow_partial:
        raise FileNotFoundError(f'Missing complete evaluations: {missing}; --allow-partial is for inspection only')
    comparisons = {}
    pairs = []
    for setting in ('A', 'B', 'C'):
        target = next(c for c in configs if c['setting'] == setting and c['method'] == 'cored')
        pairs += [('cored_minus_baseline', target['run_id'], c['run_id']) for c in configs
                  if c['setting'] == setting and c['method'] != 'cored']
    for method in ('longcot', 'sgl', 'ssft', 'palign', 'cored'):
        ids = {c['setting']: c['run_id'] for c in configs if c['method'] == method}
        pairs += [('gemini_minus_original', ids['B'], ids['A']), ('small_minus_large', ids['C'], ids['A'])]
    for kind, left, right in pairs:
        if left not in per_question or right not in per_question:
            continue
        result = {}
        for benchmark in sorted(set(per_question[left]) & set(per_question[right])):
            a, b = per_question[left][benchmark], per_question[right][benchmark]
            result[benchmark] = {metric: paired(a, b, metric, repetitions) for metric in next(iter(a.values()))}
            jsonl(destination / 'discordant' / f'{left}__{right}_{benchmark}.jsonl',
                  ({'question_id': q, 'left': a[q], 'right': b[q]} for q in sorted(a) if a[q]['pass@1'] != b[q]['pass@1']))
        result['macro_math'] = {}
        for metric in ('Avg@32', 'Avg@64', 'pass@1', 'pass@3'):
            distributions = [bootstrap([per_question[left][b][q][metric] - per_question[right][b][q][metric]
                                        for q in sorted(per_question[left][b])], repetitions, 2027 + i) for i, b in enumerate(MATH)]
            distribution = np.mean(distributions, axis=0)
            result['macro_math'][metric] = {'mean': float(np.mean([result[b][metric]['mean'] for b in MATH])),
                                          'ci95': np.quantile(distribution, [.025, .975]).tolist()}
        comparisons[f'{kind}:{left}:{right}'] = result
    write(destination / 'metrics.json', {'runs': summaries, 'paired_comparisons': comparisons, 'missing': missing,
          'analysis_source_sha256': sha(__file__), 'bootstrap_seed': 2027,
          'complete': not missing, 'seed_variability_note': 'Five 12-13-sample seed estimates are not independent Avg@64 runs or training-seed stability.',
          'metrics_note': 'Pass@1 = mean sampled accuracy = Avg@64. Pass@3 is the unbiased 64-response estimator. Values are fractions.'})
    table = [f'Status: {"INCOMPLETE" if missing else "complete"}. Missing evaluations: {len(missing)}.', '',
             '|Run|Avg@32|Avg@64|Pass@3|ARC|LogiQA|', '|---|---:|---:|---:|---:|---:|']
    for run, result in summaries.items():
        def fmt(value):
            return f"{100*value['mean']:.2f} [{100*value['ci95'][0]:.2f}, {100*value['ci95'][1]:.2f}]"
        table.append('|'+run+'|'+'|'.join([fmt(result['macro_math'][key]) for key in ('Avg@32', 'Avg@64', 'pass@3')]
                     + [fmt(result[b]['pass@1']) if b in result else 'N/A' for b in ('arc_challenge', 'logiqa2')])+'|')
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'results.md').write_text('\n'.join(table) + '\n')
    differences = ['|Comparison|Avg@64 difference (pp), 95% CI|Pass@3 difference (pp), 95% CI|', '|---|---:|---:|']
    for name, result in comparisons.items():
        differences.append('|'+name+'|'+fmt(result['macro_math']['Avg@64'])+'|'+fmt(result['macro_math']['pass@3'])+'|')
    (destination / 'paired_comparisons.md').write_text('\n'.join(differences)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--bootstrap', type=int, default=5000)
    p.add_argument('--allow-partial', action='store_true')
    args = p.parse_args()
    report(args.runs, args.output, args.bootstrap, args.allow_partial)


if __name__ == '__main__':
    main()
