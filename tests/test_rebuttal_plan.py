import json
from types import SimpleNamespace

import numpy as np
import pytest

from rebuttal.common import BUDGETS, DATA_REV, MATH, ROOT, SEEDS, jsonl, matrix, rows, write
from rebuttal.data import adapt, mc_item, normalize_teachers, prepare_benchmarks
from rebuttal.evaluate import grade_record, mc_answer, validate_records
from rebuttal.statistics import bootstrap, paired, question_metrics, sample_order


def test_final_matrix_and_committed_configs():
    configurations = matrix()
    assert len(configurations) == len({c['run_id'] for c in configurations}) == 15
    for setting in 'ABC':
        selected = [c for c in configurations if c['setting'] == setting]
        assert len(selected) == 5
        assert all(c['hardware']['gpus'] == '3,4,5' and c['train']['seed'] == 42 for c in selected)
        assert all(bool(c['evaluation']['ood']) == (setting != 'B') for c in selected)
    for config in configurations:
        assert json.loads((ROOT / 'rebuttal/configs' / (config['run_id']+'.json')).read_text()) == config
        assert '8B' not in config['model']['repo']
        assert config['data']['repo'] == 'simplescaling/s1K-1.1'
        assert config['data']['revision'] == DATA_REV
        if config['setting'] == 'B':
            assert config['data']['trace_source'] == 'gemini_thinking_trajectory'
            assert config['data']['answer_source'] == 'gemini_attempt'


@pytest.mark.parametrize('action', ['train', 'eval'])
@pytest.mark.parametrize('settings,count', [([], 15), (['A','C'], 10), (['A'], 5)])
def test_main_setting_selection_keeps_train_and_eval_separate(tmp_path, monkeypatch, action, settings, count):
    import sys
    from rebuttal import runner, preflight, eval_preflight
    calls = []
    class Job:
        def __init__(self, config, assets, data, runs, gpus):
            self.config = config
            self.root = runs/config['run_id']
            self.root.mkdir()
        def train(self):
            calls.append(('train', self.config['setting'], self.config['run_id']))
        def evaluate(self):
            calls.append(('eval', self.config['setting'], self.config['run_id']))
    monkeypatch.setattr(runner, 'Job', Job)
    monkeypatch.setattr(runner, 'aggregate', lambda *a: None)
    monkeypatch.setattr(preflight, 'validate', lambda *a, **kw: None)
    monkeypatch.setattr(eval_preflight, 'check_eval', lambda *a, **kw: None)
    monkeypatch.setattr(sys, 'argv', ['runner', action, '--runs', str(tmp_path/'runs'),
                                    *(['--settings', *settings] if settings else [])])
    runner.main()
    expected = {c['run_id'] for c in matrix() if not settings or c['setting'] in settings}
    assert len(calls) == count and {run_id for _, _, run_id in calls} == expected
    assert all(stage == action for stage, _, _ in calls)


def test_paired_teacher_mapping_no_gold_as_training_trace(tmp_path):
    source = [{'question': ' Q  one ', 'deepseek_thinking_trajectory': 'D reasoning', 'deepseek_attempt': 'D answer', 'solution': '42',
               'gemini_thinking_trajectory': 'G reasoning', 'gemini_attempt': 'G answer', 'gemini_grade': 'Yes',
               'thinking_trajectories': ['wrong legacy reasoning'], 'attempt': 'wrong legacy answer'},
              {'question': 'Q two', 'deepseek_thinking_trajectory': 'D2', 'deepseek_attempt': 'DA2', 'solution': '7',
               'gemini_thinking_trajectory': 'G2', 'gemini_attempt': 'GA2'}]
    normalized = normalize_teachers(source)
    assert [r['id'] for r in normalized['original']] == [r['id'] for r in normalized['gemini']]
    assert normalized['gemini'][0]['thinking'] == 'G reasoning'
    assert normalized['gemini'][0]['teacher_grade'] == 'Yes'
    adapt(normalized['gemini'], tmp_path)
    assert rows(tmp_path / 'sgl/train.jsonl')[0]['deepseek_attempt'] == 'G answer'
    assert rows(tmp_path / 'ssft.jsonl')[0]['solution'] == 'G reasoning'
    assert 'G reasoning' in json.loads((tmp_path / 'longcot.json').read_text())[0]['output']
    broken = [dict(source[0], gemini_attempt=None)]
    with pytest.raises(ValueError, match='gemini_attempt'):
        normalize_teachers(broken)
    with pytest.raises(ValueError, match='unique'):
        normalize_teachers([source[0], source[0]])
    # The real snapshot includes a pair differing only in interior whitespace.
    same_words = normalize_teachers([source[0], dict(source[0], question='Q one')])
    assert len({r['id'] for r in same_words['original']}) == 2
    assert [r['id'] for r in same_words['original']] == [r['id'] for r in same_words['gemini']]


def test_old_separate_gemini_cache_requires_preparation_again(tmp_path):
    from rebuttal.preflight import validate
    data = tmp_path / 'data'
    write(data / 'manifest.json', {'revisions': {'original': DATA_REV, 'gemini': 'old-separate-revision'}})
    with pytest.raises(ValueError, match='rerun prepare'):
        validate(tmp_path / 'assets', data, tmp_path / 'runs')


def test_shared_benchmark_preparation_handles_real_aime_layouts(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    store = tmp_path / 'palign/datasets'
    def parquet(path, values):
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(values), path)
    parquet(store / 'AIME_2024/aime_2024_problems.parquet',
            [{'Problem': f'AIME24 {i}', 'Answer': i} for i in range(30)])
    aime25 = [{'problem': f'AIME25 {i}', 'answer': str(i)} for i in range(30)]
    parquet(store / 'aime_2025/data/train-00000.parquet', aime25)
    parquet(store / 'aime_2025/part1/train-00000.parquet', aime25[:15])
    parquet(store / 'aime_2025/part2/train-00000.parquet', aime25[15:])
    parquet(store / 'aimo-validation-amc/data/train-00000.parquet',
            [{'problem': f'AMC {i}', 'answer': float(i)} for i in range(83)])
    jsonl(store / 'MATH-500/test.jsonl', [{'problem':f'MATH {i}', 'answer':str(i)} for i in range(500)])
    parquet(store / 'ARC-Challenge/test-00000-of-00001.parquet',
            [{'id':str(i), 'question':f'ARC {i}', 'choices':{'label':['1','2','3'], 'text':['a','b','c']}, 'answerKey':'2'} for i in range(1172)])
    jsonl(store / 'LogiQA2.0/test.txt', [{'id': 'repeated', 'text':'P', 'question':f'L {i}',
           'options':['a','b','c','d'], 'answer':0} for i in range(1572)])
    destination = tmp_path / 'normalized'
    prepare_benchmarks(store, destination)
    assert len(rows(destination / 'aime25.jsonl')) == 30
    assert rows(destination / 'aime24.jsonl')[0]['question'] == 'AIME24 0'
    assert rows(destination / 'aime24.jsonl')[0]['answer'] == '0'
    assert len({r['question_id'] for r in rows(destination / 'logiqa2.jsonl')}) == 1572
    monkeypatch.syspath_prepend(str(ROOT / 'cored'))
    monkeypatch.setenv('CORED_ASSET_ROOT', str(tmp_path / 'cored'))
    from eval.benchmarks import load_benchmark
    for benchmark in ('aime24', 'aime25', 'math500'):
        expected = [(r['question'], r['answer']) for r in rows(destination / f'{benchmark}.jsonl')]
        assert [(r.question, r.answer) for r in load_benchmark(benchmark)] == expected
    assert [(r.question, r.answer) for r in load_benchmark('amc')] == [
        (r['question'], r['answer']) for r in rows(destination / 'amc12.jsonl')]


def test_arc_numeric_choice_labels_and_logiqa_passage():
    arc = mc_item('arc_challenge', 0, {'id': 'a', 'question': 'Q', 'choices': {'label': ['1', '2', '3'], 'text': ['X', 'Y', 'Z']}, 'answerKey': '2'})
    assert arc['answer'] == 'B' and arc['source_choice_labels'] == ['1', '2', '3']
    logiqa = mc_item('logiqa2', 1, {'id': 'b', 'text': 'Passage', 'question': 'Which?', 'options': ['a','b','c','d'], 'answer': 0})
    assert logiqa['answer'] == 'A' and logiqa['question'].startswith('Passage\n\nWhich?')
    assert mc_answer('Reasoning mentions A.\nAnswer: B', arc['choices']) == 'B'
    assert mc_answer('<think>Answer: A</think>\nAnswer: C', arc['choices']) == 'C'
    assert mc_answer('Answer: D', arc['choices']) is None
    assert mc_answer('Maybe B would work', arc['choices']) is None


def labels(question='q', correct=16):
    order = sample_order()
    return [dict(question_id=question, eval_seed=seed, sample_id=sample, is_correct=i < correct,
                 finish_reason='stop', generated_tokens=8, invalid_answer=False)
            for i, (seed, sample) in enumerate(order)]


def test_round_robin_and_same_64_draws_for_both_averages():
    order = sample_order()
    assert len(order) == len(set(order)) == sum(BUDGETS) == 64
    assert [sum(seed == s for seed, _ in order[:32]) for s in SEEDS] == [7, 7, 6, 6, 6]
    metrics = question_metrics(labels(correct=16), True)['q']
    assert metrics['Avg@32'] == .5 and metrics['Avg@64'] == metrics['pass@1'] == .25
    import math
    assert metrics['pass@3'] == pytest.approx(1-math.comb(48,3)/math.comb(64,3))
    with pytest.raises(ValueError):
        question_metrics(labels()[:-1], True)


def test_paired_bootstrap_preserves_question_correlations():
    left = {'q1': {'p': .3}, 'q2': {'p': .9}}
    right = {'q1': {'p': .1}, 'q2': {'p': .7}}
    result = paired(left, right, 'p', 1000)
    assert result['mean'] == pytest.approx(.2)
    assert result['ci95'] == pytest.approx([.2, .2])
    assert np.array_equal(bootstrap([0,1], 200), bootstrap([0,1], 200))
    with pytest.raises(ValueError):
        paired(left, {'other': {'p': .1}}, 'p')


def test_jsonl_resume_recovers_only_trailing_incomplete_record(tmp_path):
    file = tmp_path / 'raw.jsonl'
    file.write_text('{"x": 1}\n{"x":')
    assert rows(file, repair=True) == [{'x': 1}]
    assert file.read_text() == '{"x": 1}\n'
    file.write_text('{"x": 1}')
    assert rows(file, repair=True) == [{'x':1}]
    assert file.read_text().endswith('\n')
    file.write_text('broken\n{"x":1}\n')
    with pytest.raises(ValueError):
        rows(file, repair=True)


def test_generation_schema_shards_resume_and_caps(tmp_path, monkeypatch):
    import sys
    from rebuttal.evaluate import generate
    calls = []
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return '<chat>' + messages[0]['content']
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer())))
    class Engine:
        def __init__(self, **kwargs):
            pass
        def generate(self, prompts, params, **kwargs):
            calls.append(vars(params))
            return [SimpleNamespace(prompt_token_ids=[1,2], outputs=[SimpleNamespace(text='Answer: B', token_ids=[3,4],
                    finish_reason='stop', stop_reason=None) for _ in range(params.n)]) for _ in prompts]
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(LLM=Engine, SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs)))
    checkpoint = tmp_path / 'model'
    checkpoint.mkdir()
    (checkpoint / 'model.safetensors').write_bytes(b'fixture-only')
    data = tmp_path / 'data'
    for benchmark in ('aime24', 'arc_challenge', 'logiqa2'):
        jsonl(data / (benchmark+'.jsonl'), [{'question_id': benchmark+':0', 'question':'Q', 'answer':'B', 'choices':['A','B','C']}])
    config = matrix()[0]
    config['evaluation'] = dict(config['evaluation'], math=['aime24'], ood=['arc_challenge','logiqa2'])
    output = tmp_path / 'eval'
    generate(config, checkpoint, data, output, 0, 1, 8)
    assert len(rows(output / 'aime24.rank0.jsonl')) == 64
    assert [r['max_tokens'] for r in calls[-2:]] == [1024, 4096]
    assert calls[-1]['n'] == 1 and calls[-1]['temperature'] == 0
    count = len(calls)
    generate(config, checkpoint, data, output, 0, 1, 8)
    assert len(calls) == count
    raw = rows(output / 'arc_challenge.rank0.jsonl')
    assert grade_record(raw[0])['is_correct'] is True
    assert set(['run_id','checkpoint_id','benchmark','question_id','eval_seed','sample_id','prompt','response',
                'predicted_answer','ground_truth','is_correct','input_tokens','generated_tokens','finish_reason','generation_config']) <= raw[0].keys()
    validate_records(raw, rows(data / 'arc_challenge.jsonl'), 'arc_challenge')


def test_cost_excludes_eval_and_does_not_multiply_wall_time_twice(tmp_path):
    from rebuttal.cost import aggregate
    events = [{'stage':'optimization','attempt_id':'a','status':'complete','gpu_count':3,'seconds':120},
              {'stage':'evaluation_generate','attempt_id':'b','status':'complete','gpu_count':3,'seconds':900}]
    jsonl(tmp_path / 'stage_events.jsonl', events)
    jsonl(tmp_path / 'job_events.jsonl', [{'wall_seconds':125,'allocated_gpu_count':3}])
    for rank in range(3):
        write(tmp_path/'telemetry/a'/f'process-{rank}.json', {'forward_tokens':100,'autograd_tokens':100,
              'optimization_steps':5,'estimated_flops':600,'peak_allocated_bytes':1000,'peak_reserved_bytes':1200,'sample_presentations':10})
    result = aggregate(tmp_path)
    assert result['completed_stage_gpu_hours'] == pytest.approx(.1)
    assert result['allocated_gpu_hours'] == pytest.approx(125*3/3600)
    assert result['optimization_steps'] == 5
    assert result['training_tokens'] == 300
    assert len(result['stages']) == 1


def test_logiqa_duplicate_source_ids_remain_distinct():
    row = {'id': 'duplicate', 'text': 'P', 'question':'Q', 'options':['a','b','c','d'], 'answer':1}
    assert mc_item('logiqa2', 0, row)['question_id'] != mc_item('logiqa2', 1, row)['question_id']


def test_immutable_snapshot_detects_extra_or_changed_files(tmp_path):
    from rebuttal.provision import attest, verify_snapshot
    (tmp_path/'config.json').write_text('{}')
    attest(tmp_path, 'repo/model', 'abc')
    verify_snapshot(tmp_path, 'repo/model', 'abc')
    (tmp_path/'old.safetensors').write_bytes(b'extra old weights')
    with pytest.raises(ValueError, match='file set'):
        verify_snapshot(tmp_path, 'repo/model', 'abc')
    (tmp_path/'old.safetensors').unlink()
    (tmp_path/'config.json').write_text('{"changed":true}')
    with pytest.raises(ValueError, match='content changed'):
        verify_snapshot(tmp_path, 'repo/model', 'abc')


def test_telemetry_counts_unpadded_inputs_and_optimizer_updates(tmp_path, monkeypatch):
    import torch
    from rebuttal import telemetry
    monkeypatch.setenv('REBUTTAL_STAGE_DIR', str(tmp_path))
    monkeypatch.setattr(torch.optim.Optimizer, '__init__', torch.optim.Optimizer.__init__)
    monkeypatch.setattr(telemetry, 'STATE', {'forward_tokens':0, 'autograd_tokens':0,
                         'sample_presentations':0, 'optimization_steps':0, 'estimated_flops':0., 'installed':False})
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(4, 3)
        def forward(self, input_ids, attention_mask=None):
            return self.embedding(input_ids).sum()
    telemetry.install()
    model = telemetry.activate(Tiny())
    optimizer = torch.optim.AdamW(model.parameters())
    model(input_ids=torch.tensor([[1,2,0]]), attention_mask=torch.tensor([[1,1,0]])).backward()
    optimizer.step()
    telemetry.flush()
    record = json.loads(next(tmp_path.glob('process-*.json')).read_text())
    assert record['forward_tokens'] == record['autograd_tokens'] == 2
    assert record['optimization_steps'] == 1
    assert record['estimated_flops'] > 0


def test_full_math_report_paired_macro_and_partial_label(tmp_path):
    from rebuttal.statistics import report
    count = {'aime24':30, 'aime25':30, 'amc12':83, 'math500':500}
    runs = tmp_path / 'runs'
    for method, correct in [('longcot', 0), ('cored', 64)]:
        for benchmark in MATH:
            jsonl(runs / f'B_{method}_7b_seed42/evaluation/{benchmark}.scored.jsonl',
                  (row for i in range(count[benchmark]) for row in labels(f'{benchmark}:{i}', correct)))
    report(runs, tmp_path/'report', repetitions=100, allow_partial=True)
    result = json.loads((tmp_path/'report/metrics.json').read_text())
    assert result['complete'] is False
    key = 'cored_minus_baseline:B_cored_7b_seed42:B_longcot_7b_seed42'
    assert result['paired_comparisons'][key]['macro_math']['Avg@64']['ci95'] == [1.,1.]
    assert 'INCOMPLETE' in (tmp_path/'report/results.md').read_text()
    with pytest.raises(FileNotFoundError):
        report(runs, tmp_path/'reject', repetitions=10)
