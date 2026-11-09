import copy
import json
import sys

import pytest

from rebuttal.common import MATH, ROOT, jsonl, matrix, rows, write
from rebuttal.formal_b_2h100 import configurations
from rebuttal.runner import Job


def test_formal_profile_keeps_real_gemini_protocol_and_main_configs():
    main = copy.deepcopy(matrix())
    actual = configurations()
    assert len(actual) == 5 and matrix() == main
    for config in actual:
        original = next(c for c in main if c['run_id'] == config['run_id'])
        assert config['setting'] == 'B'
        assert config['data'] == original['data']
        assert config['data']['trace_source'] == 'gemini_thinking_trajectory'
        assert config['data']['expected_samples'] == 1000
        assert config['evaluation'] == original['evaluation']
        assert config['evaluation']['ood'] == []
        assert sum(config['evaluation']['budgets']) == 64
        assert config['train']['epochs'] == 3 and config['train']['seed'] == 42
        assert config['train']['global_batch'] == 32 and config['train']['micro_batch'] == 1
        assert config['train']['cutoff'] == original['train']['cutoff']
        assert not config.get('smoke') and not config.get('development_smoke')
        assert config['method_specific'] == original['method_specific']
        if config['method'] != 'cored':
            assert config['train']['gradient_accumulation'] == 16


@pytest.mark.parametrize('method', ['longcot', 'palign', 'sgl', 'ssft', 'cored'])
def test_formal_native_commands_have_no_smoke_limits(tmp_path, monkeypatch, method):
    from rebuttal import runner, method_prep
    from rebuttal.smoke_2gpu import fixtures
    # Fixture input is only for command construction, never the formal launcher.
    fixtures(tmp_path/'data')
    config = next(c for c in configurations() if c['method'] == method)
    model = tmp_path/'assets/models'/config['model']['repo'].split('/')[-1]
    model.mkdir(parents=True)
    write(model/'config.json', {})
    monkeypatch.setattr(runner, 'python_for', lambda _: sys.executable)
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *a, **kw: '{}')
    monkeypatch.setattr(method_prep, 'merge_ssft', lambda *a: None)
    job = Job(config, tmp_path/'assets', tmp_path/'data', tmp_path/'runs', '0,1')

    class Captured(Exception):
        pass

    def execute(stage, commands, gpu_count):
        assert gpu_count == 2
        for command, _ in commands:
            assert '--development_smoke' not in command and '--dry_run' not in command
            assert '--max_steps' not in command and '--max-steps' not in command
        if stage == 'prefix_alignment':
            assert len(commands) == 2
            values = rows(job.input/'canonical.jsonl')
            for i in range(2):
                jsonl(job.root/f'palign-{i}.jsonl', [dict(r, accepted=True, prefix=r['thinking'],
                      continuation=r['answer_text']) for r in values[i::2]])
            return
        if stage == 'spectral_masks':
            cmd = commands[0][0]
            assert cmd[cmd.index('--max-length')+1] == '32768'
            return
        if stage == 'attribution':
            assert len(commands) == 2
            assert config['method_specific']['ig_steps'] == 50
            return
        command = commands[0][0]
        if method == 'cored':
            assert stage == 'cored_pipeline' and '--formal_2h100' in command
            assert command[command.index('--global_batch')+1] == 32
            assert command[command.index('--epochs')+1] == 3
        else:
            assert stage == 'optimization' and '--nproc_per_node=2' in command
            if method == 'sgl':
                assert command[command.index('--global-batch-size')+1] == '32'
                assert command[command.index('--per-device-train-batch-size')+1] == '1'
            elif method == 'ssft':
                assert command[command.index('--gradient_accumulation_steps')+1] == '16'
                assert command[command.index('--max_seq_length')+1] == '32768'
            else:
                import yaml
                value = yaml.safe_load((job.root/'native_train.yaml').read_text())
                assert value['gradient_accumulation_steps'] == 16 and value['per_device_train_batch_size'] == 1
                assert value['cutoff_len'] == 32768 and 'max_steps' not in value
        raise Captured

    monkeypatch.setattr(job, 'execute', execute)
    with pytest.raises(Captured):
        job.train()


def test_formal_launcher_orders_real_prepare_train_eval_and_report(tmp_path, monkeypatch):
    from rebuttal import formal_b_2h100 as formal
    monkeypatch.setattr(sys, 'argv', ['formal', 'all', '--runs', str(tmp_path/'runs'),
                                     '--reports', str(tmp_path/'reports')])
    events = []
    def prepare(command, **kwargs):
        assert command[2] == 'rebuttal.data' and 'fixtures' not in ' '.join(command)
        events.append('prepare')
    monkeypatch.setattr(formal.subprocess, 'run', prepare)
    def preflight(*a, **kwargs):
        assert kwargs['formal_2h100'] and kwargs['gpus'] == '0,1'
        events.append('preflight')
    monkeypatch.setattr(formal, 'validate', preflight)
    class FakeJob:
        def __init__(self, config, assets, data, runs, gpus):
            self.config, self.root = config, runs/config['run_id']
            self.root.mkdir(parents=True)
        def train(self):
            events.append(('train', self.config['method']))
            checkpoint = self.root/'checkpoint'
            write(checkpoint/'checkpoint_manifest.json', {'files_sha256':{}})
            write(self.root/'final_checkpoint.json', {'path': str(checkpoint)})
        def evaluate(self):
            events.append(('eval', self.config['method']))
    monkeypatch.setattr(formal, 'Job', FakeJob)
    def report(*a, **kwargs):
        assert len(kwargs['configurations']) == 5
        assert not kwargs.get('allow_partial')
        events.append('report')
    monkeypatch.setattr(formal, 'report', report)
    monkeypatch.setattr(formal, 'tables', lambda *a, **kw: events.append('cost'))
    formal.main()
    assert events[:2] == ['prepare', 'preflight']
    assert events[2:12] == [(stage, method) for method in ['longcot','sgl','ssft','palign','cored']
                           for stage in ('train','eval')]
    assert events[-2:] == ['report','cost']
    for config in configurations():
        assert rows(tmp_path/'runs'/config['run_id']/'job_events.jsonl')[0]['allocated_gpu_count'] == 2
    assert json.loads((tmp_path/'runs/launcher_status.json').read_text())['status'] == 'complete'


def test_cost_expansion_counts_two_h100s_not_three_b200s(tmp_path):
    from rebuttal.cost import aggregate
    jsonl(tmp_path/'stage_events.jsonl', [{'stage':'cored_pipeline','attempt_id':'a','status':'complete',
        'gpu_count':2,'seconds':10,'gpu_hours':20/3600}])
    jsonl(tmp_path/'job_events.jsonl', [{'wall_seconds':12,'allocated_gpu_count':2}])
    jsonl(tmp_path/'artifacts/cored/timings.jsonl', [
        {'stage':'phase1','status':'complete','seconds':6}, {'stage':'merge','status':'complete','seconds':2}])
    cost = aggregate(tmp_path)
    assert [stage['gpu_count'] for stage in cost['stages']] == [2,1]
    assert cost['allocated_gpu_hours'] == pytest.approx(24/3600)
    assert cost['completed_stage_gpu_hours'] == pytest.approx(14/3600)


def test_report_requires_only_the_five_complete_setting_b_runs(tmp_path):
    from rebuttal.statistics import report
    from rebuttal.common import SEEDS, BUDGETS
    configs = configurations()
    counts = dict(aime24=30, aime25=30, amc12=83, math500=500)
    for config in configs:
        for benchmark in MATH:
            records = (dict(run_id=config['run_id'], question_id=f'{benchmark}:{q}', eval_seed=seed,
                sample_id=i, is_correct=config['method']=='cored', finish_reason='stop',
                generated_tokens=3, invalid_answer=False)
                for q in range(counts[benchmark]) for seed, count in zip(SEEDS, BUDGETS) for i in range(count))
            jsonl(tmp_path/'runs'/config['run_id']/'evaluation'/f'{benchmark}.scored.jsonl', records)
    report(tmp_path/'runs', tmp_path/'reports', repetitions=10, configurations=configs)
    result = json.loads((tmp_path/'reports/metrics.json').read_text())
    assert result['complete'] and len(result['runs']) == 5 and not result['missing']
    assert len(result['paired_comparisons']) == 4
    assert all(key.startswith('cored_minus_baseline:B_') for key in result['paired_comparisons'])
    # One missing benchmark must prevent a final report; it cannot silently pass.
    (tmp_path/'runs'/configs[0]['run_id']/'evaluation/aime24.scored.jsonl').unlink()
    with pytest.raises(FileNotFoundError, match='Missing complete evaluations'):
        report(tmp_path/'runs', tmp_path/'incomplete-report', repetitions=10, configurations=configs)


def test_note_keeps_short_smoke_section_and_removes_old_long_runbook():
    text = (ROOT/'note.txt').read_text()
    assert 'Smoke test in 2x3090' in text
    assert 'formal_B_2h100.sh all' in text
    assert 'PILOT_PY' not in text and len(text.splitlines()) < 65


@pytest.mark.parametrize('formal,gpu_name', [(True,'H100'), (False,'B200'), (True,'B200')])
def test_preflight_selects_h100_profile_without_loosening_b200_default(tmp_path, monkeypatch, formal, gpu_name):
    from rebuttal import preflight, provision, runner
    from rebuttal.common import DATA_REV, MODELS, TEACHER_FIELDS, sha
    assets, data, runs = tmp_path/'assets', tmp_path/'data', tmp_path/'runs'
    runs.mkdir()
    hashes = {}
    for variant in ('original','gemini'):
        jsonl(data/variant/'canonical.jsonl', [{'id':str(i)} for i in range(1000)])
        hashes[variant] = sha(data/variant/'canonical.jsonl')
    write(data/'manifest.json', {'dataset':{'repo':'simplescaling/s1K-1.1','revision':DATA_REV,'split':'train'},
          'revisions':{'original':DATA_REV,'gemini':DATA_REV}, 'teacher_fields':TEACHER_FIELDS, 'files':hashes})
    for variant, size in [('original','7b'),('gemini','7b'),('original','1.5b')]:
        write(data/'statistics'/f'{variant}_{size}.json', {'data_sha256':hashes[variant]})
    for repo, revision in MODELS.values():
        write(assets/'models'/repo.split('/')[-1]/'config.json', {})
        write(assets/'models'/repo.split('/')[-1]/'tokenizer_config.json', {})
    write(data/'benchmarks/manifest.json', {})
    monkeypatch.setattr(provision, 'verify_snapshot', lambda *a: None)
    monkeypatch.setattr(runner, 'python_for', lambda method: '/fixture/python')
    count, expected_name = (2,'H100') if formal else (3,'B200')
    def command(args, **kwargs):
        if '-m' in args:
            assert args[args.index('--expected_devices')+1] == str(count)
            assert args[args.index('--required_gpu_name')+1] == expected_name
            write(args[args.index('--out')+1], {'devices':[{'name':gpu_name} for _ in range(count)]})
    monkeypatch.setattr(preflight.subprocess, 'run', command)
    if gpu_name != expected_name:
        with pytest.raises(ValueError, match='hardware mismatch'):
            preflight.validate(assets, data, runs, environments=True, gpus='0,1', formal_2h100=formal)
    else:
        result = preflight.validate(assets, data, runs, environments=True,
                                    gpus='0,1' if formal else '0,1,2', formal_2h100=formal)
        assert result['configurations'] == (5 if formal else 15)
