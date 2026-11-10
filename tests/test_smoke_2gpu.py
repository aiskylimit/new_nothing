import copy
import json
import sys
from types import SimpleNamespace

import pytest

from rebuttal.common import ROOT, matrix, rows, write
from rebuttal.evaluate import development_limits, math_schedule, validate_records
from rebuttal.runner import Job
from rebuttal.smoke_2gpu import configurations, fixtures


def test_development_profiles_do_not_change_main_matrix_or_expert_count():
    main = copy.deepcopy(matrix())
    dev = configurations('ABC')
    assert len(dev) == 15 and matrix() == main
    for c in dev:
        assert c['run_id'].startswith('dev2_') and c['smoke']
        assert c['train']['global_batch'] == 2 and c['train']['micro_batch'] == 1
        assert c['train']['cutoff'] == 512 and c['smoke_steps'] == 4
        if c['method'] == 'cored':
            assert c['method_specific']['M'] == 3
        if c['method'] == 'ssft':
            assert c['method_specific']['require_valid_attribution']
    assert len(configurations('C')) == 5
    assert len(configurations('ABC', ['C_cored_1.5b_seed42'])) == 1


def test_synthetic_fixtures_are_isolated_and_eval_grid_is_smaller_only_for_dev(tmp_path):
    fixtures(tmp_path)
    assert 'NOT s1K' in json.loads((tmp_path/'manifest.json').read_text())['kind']
    assert len(rows(tmp_path/'original/canonical.jsonl')) == 4
    assert len(rows(tmp_path/'gemini/ssft.jsonl')) == 4
    questions = rows(tmp_path/'benchmarks/math500.jsonl')
    result = [dict(question_id=r['question_id'], eval_seed=42, sample_id=0) for r in questions]
    dev = configurations('C')[0]
    validate_records(result, questions, 'math500', config=dev)
    with pytest.raises(ValueError, match='response grid'):
        validate_records(result, questions, 'math500')
    assert sum(n for _, n in math_schedule()) == 64
    with pytest.raises(ValueError, match='smoke run'):
        development_limits(dict(dev, smoke=False))


@pytest.mark.parametrize('method,stage', [('longcot','optimization'), ('palign','prefix_alignment'),
    ('sgl','spectral_masks'), ('ssft','attribution'), ('cored','cored_smoke_phase1')])
def test_native_development_commands_use_two_workers_and_small_limits(tmp_path, monkeypatch, method, stage):
    from rebuttal import runner
    fixtures(tmp_path/'data')
    config = next(c for c in configurations('C') if c['method'] == method)
    model = tmp_path/'assets/models'/config['model']['repo'].split('/')[-1]
    model.mkdir(parents=True)
    write(model/'config.json', {})
    monkeypatch.setattr(runner, 'python_for', lambda _: sys.executable)
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *a, **kw: '{}')
    job = Job(config, tmp_path/'assets', tmp_path/'data', tmp_path/'runs', '0,1')

    class Captured(Exception):
        pass

    def execute(name, commands, gpu_count):
        assert name == stage and gpu_count == 2
        if method in ('ssft','palign'):
            assert [gpu for _, gpu in commands] == ['0','1']
            for i, (cmd, _) in enumerate(commands):
                assert cmd[cmd.index('--shard')+1] == str(i)
                assert cmd[cmd.index('--shards')+1] == '2'
        elif method == 'cored':
            cmd = commands[0][0]
            assert '--development_smoke' in cmd
            assert cmd[cmd.index('--dry_run')+1] == 4
            assert cmd[cmd.index('--global_batch')+1] == 2
        else:
            cmd = commands[0][0]
            assert '--nproc_per_node=2' in cmd
            if method == 'sgl':
                assert cmd[cmd.index('--max-length')+1] == '512'
            else:
                import yaml
                settings = yaml.safe_load((job.root/'native_train.yaml').read_text())
                assert settings['max_steps'] == 4
                assert settings['per_device_train_batch_size'] == 1
        raise Captured

    monkeypatch.setattr(job, 'execute', execute)
    with pytest.raises(Captured):
        job.train()


def test_main_job_refuses_two_gpus(tmp_path):
    with pytest.raises(ValueError, match='3 distinct'):
        Job(matrix()[0], tmp_path, tmp_path, tmp_path, '0,1')


def test_execute_records_path_arguments_as_json_without_gpu(tmp_path):
    job = object.__new__(Job)
    job.root, job.config, job.gpus = tmp_path, {'run_id':'fixture'}, '0,1'
    job.fingerprint = 'fixture-code-data'
    job.execute('cpu_fixture', [([sys.executable, '-c', 'print("ok")', tmp_path], '')], 0, instrument=False)
    event = rows(tmp_path/'stage_events.jsonl')[0]
    assert event['status'] == 'complete'
    assert event['commands'][0][0][-1] == str(tmp_path)


def test_cored_two_gpu_gate_requires_development_and_positive_dry_run(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'cored'))
    from train import pipeline_b200
    monkeypatch.setattr(pipeline_b200, 'local_model_path', lambda path: path)
    for flags in ([], ['--development_smoke'], ['--development_smoke','--dry_run','-1']):
        monkeypatch.setattr(sys, 'argv', ['pipeline_b200', '--gpus','0,1', *flags])
        with pytest.raises(ValueError):
            pipeline_b200.main()


def test_development_orchestration_records_all_five_methods(tmp_path, monkeypatch):
    from rebuttal import smoke_2gpu as dev
    monkeypatch.setattr(sys, 'argv', ['smoke_2gpu','--settings','C','--assets',str(tmp_path/'assets'),
                                    '--runs',str(tmp_path/'runs')])
    monkeypatch.setattr(dev, 'python_for', lambda _: '/fixture/python')
    from rebuttal import provision
    monkeypatch.setattr(provision, 'verify_snapshot', lambda *a: None)
    from rebuttal import eval_preflight
    monkeypatch.setattr(eval_preflight, 'check_eval', lambda *a, **kw: None)
    probes = []
    def probe(command, **kwargs):
        probes.append(command)
        assert '--nproc_per_node=2' in command and kwargs['env']['CUDA_VISIBLE_DEVICES'] == '0,1'
    monkeypatch.setattr(dev.subprocess, 'run', probe)
    trained, evaluated = [], []
    class FakeJob:
        def __init__(self, config, assets, data, runs, gpus):
            self.config, self.root = config, runs/config['run_id']
        def train(self):
            trained.append(self.config['method'])
        def evaluate(self):
            evaluated.append(self.config['method'])
    monkeypatch.setattr(dev, 'Job', FakeJob)
    dev.main()
    assert len(probes) == 1
    assert trained == evaluated == ['longcot','sgl','ssft','palign','cored']
    result = json.loads((tmp_path/'runs/summary.json').read_text())
    assert result['status'] == 'passed' and len(result['completed']) == 5


def test_development_generation_uses_short_context_and_two_complete_shards(tmp_path, monkeypatch):
    from rebuttal.evaluate import generate, grade_record
    from rebuttal.common import jsonl
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return messages[0]['content']
    engines, calls = [], []
    class Engine:
        def __init__(self, **kwargs):
            engines.append(kwargs)
        def generate(self, prompts, params, **kwargs):
            calls.append(vars(params))
            return [SimpleNamespace(prompt_token_ids=[1,2], outputs=[SimpleNamespace(
                text=r'\\boxed{2}', token_ids=[3], finish_reason='stop', stop_reason=None)]) for _ in prompts]
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer())))
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(LLM=Engine,
        SamplingParams=lambda **kw: SimpleNamespace(**kw)))
    checkpoint = tmp_path/'checkpoint'
    checkpoint.mkdir()
    (checkpoint/'model.safetensors').write_bytes(b'fixture')
    data = tmp_path/'data'
    fixtures(data)
    config = configurations('A')[0]
    output = tmp_path/'evaluation'
    for rank in range(2):
        generate(config, checkpoint, data/'benchmarks', output, rank, 2, 1)
    for kwargs in engines:
        assert kwargs['max_model_len'] == 2048 and kwargs['max_num_seqs'] == 1
        assert kwargs['enforce_eager'] is True and kwargs['tensor_parallel_size'] == 1
    assert len(calls) == 12 and all(c['n'] == 1 and c['max_tokens'] == 256 for c in calls)
    for name in config['evaluation']['math'] + config['evaluation']['ood']:
        records = [r for path in sorted(output.glob(f'{name}.rank*.jsonl')) for r in rows(path)]
        validate_records(records, rows(data/'benchmarks'/f'{name}.jsonl'), name, config=config)
        scored = [grade_record(r) for r in records]
        assert all(isinstance(r['is_correct'], bool) for r in scored)
        jsonl(output/f'{name}.scored.jsonl', scored)
