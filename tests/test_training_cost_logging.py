"""Regression checks for IG counters, durable stage accounting, and coverage."""
import importlib.util
import json
import socket
import sys
from pathlib import Path

import pytest

from rebuttal.common import ROOT, jsonl, matrix, rows, write
from rebuttal.cost import CORED_STAGES, aggregate, tables
from rebuttal.runner import Job


def event(stage='optimization', attempt='a', gpus=3):
    return dict(stage=stage, attempt_id=attempt, status='complete', gpu_count=gpus, seconds=10)


def counter(**kwargs):
    return dict(dict(forward_tokens=100, autograd_tokens=100, sample_presentations=1,
                     optimization_steps=1, estimated_flops=600., peak_allocated_bytes=None,
                     peak_reserved_bytes=None), **kwargs)


def test_actual_ssft_ig_counts_full_sequence_without_changing_attribution(tmp_path, monkeypatch):
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from rebuttal import telemetry

    spec = importlib.util.spec_from_file_location('cost_test_ig', ROOT / 'SegmentSelectiveSFT/Attribution/grad_analyze.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(42)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64)).to(torch.bfloat16)
    calculator = module.IntegratedGradientsAttribution.__new__(module.IntegratedGradientsAttribution)
    calculator.model = model
    calculator.tokens_forwarded = calculator.tokens_backwarded = 0
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **kw: self)
    monkeypatch.setattr(telemetry, 'install', lambda: None)
    monkeypatch.setattr(telemetry, 'STATE', dict(forward_tokens=0, autograd_tokens=0,
        sample_presentations=0, optimization_steps=0, estimated_flops=0., installed=False))
    monkeypatch.delenv('REBUTTAL_STAGE_DIR', raising=False)
    args = ([1,2,3,4,5,6], [(0,3)], (4,6))
    baseline = calculator.batch_compute_step_to_answer_attribution_integrated(*args, steps=3, batch_size=2)
    assert telemetry.STATE['forward_tokens'] == 0
    calculator.tokens_forwarded = calculator.tokens_backwarded = 0
    monkeypatch.setenv('REBUTTAL_STAGE_DIR', str(tmp_path / 'telemetry/a'))
    monkeypatch.setenv('REBUTTAL_WORKER_ID', '0')
    monkeypatch.delenv('RANK', raising=False)
    telemetry.activate(model)
    observed = calculator.batch_compute_step_to_answer_attribution_integrated(*args, steps=3, batch_size=2)
    torch.testing.assert_close(torch.tensor(observed), torch.tensor(baseline))
    assert torch.isfinite(torch.tensor(observed)).all()
    assert calculator.tokens_forwarded == calculator.tokens_backwarded == 18
    assert telemetry.STATE['forward_tokens'] == telemetry.STATE['autograd_tokens'] == 18
    assert telemetry.STATE['estimated_flops'] > 0
    telemetry.flush(final=True)
    telemetry.flush()  # A later flush must not undo finalization.
    jsonl(tmp_path / 'stage_events.jsonl', [event('attribution', gpus=1)])
    cost = aggregate(tmp_path)
    assert cost['flops_complete'] and cost['token_counter_coverage_complete']
    assert cost['training_tokens'] == 18


def job_at(path):
    job = object.__new__(Job)
    job.root, job.config, job.gpus, job.fingerprint = path, {'run_id': 'fixture'}, '0', 'fixture-fingerprint'
    return job


def test_event_is_committed_before_marker_and_recovers_without_reexecution(tmp_path, monkeypatch):
    from rebuttal import runner
    job = job_at(tmp_path)
    original = runner.write
    marker = tmp_path / 'cpu_fixture.complete.json'
    interrupted = []

    def interrupt_marker(path, value):
        if Path(path) == marker and not interrupted:
            # This assertion fails if the completion marker precedes accounting.
            committed = rows(tmp_path / 'stage_events.jsonl')
            assert committed[-1]['status'] == 'complete'
            assert committed[-1]['attempt_id'] == value['attempt_id']
            interrupted.append(True)
            raise RuntimeError('simulated interruption before marker')
        original(path, value)

    monkeypatch.setattr(runner, 'write', interrupt_marker)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        job.execute('cpu_fixture', [([sys.executable, '-c', 'pass'], '')], 0, instrument=False)
    assert not marker.exists()
    committed = rows(tmp_path / 'stage_events.jsonl')
    job.execute('cpu_fixture', [([sys.executable, '-c', 'raise RuntimeError("must not execute")'], '')], 0, instrument=False)
    assert marker.exists() and rows(tmp_path / 'stage_events.jsonl') == committed
    job.execute('cpu_fixture', [([sys.executable, '-c', 'raise RuntimeError("must not execute")'], '')], 0, instrument=False)
    assert rows(tmp_path / 'stage_events.jsonl') == committed


def test_orphaned_legacy_marker_is_not_silently_reused(tmp_path):
    job = job_at(tmp_path)
    write(tmp_path / 'optimization.complete.json', {'fingerprint': job.fingerprint, 'attempt_id': 'lost'})
    with pytest.raises(ValueError, match='no completed cost event'):
        job.execute('optimization', [([sys.executable, '-c', 'pass'], '0')], 1)
    cost = aggregate(tmp_path)
    assert not cost['stage_accounting_complete'] and not cost['flops_complete']
    assert cost['orphaned_completion_markers'] == ['optimization.complete.json']


def test_two_actual_cpu_workers_finalize_distinct_rank_counters(tmp_path):
    # Fail promptly when localhost is forbidden, instead of waiting for rendezvous retries.
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    script = tmp_path / 'tiny_train.py'
    script.write_text('''
import torch
from rebuttal.telemetry import activate
class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)
    def forward(self, input_ids):
        return self.embedding(input_ids).sum()
model = activate(Tiny())
optimizer = torch.optim.AdamW(model.parameters())
model(input_ids=torch.tensor([[1, 2]])).backward()
optimizer.step()
''')
    job = job_at(tmp_path)
    job.gpus = '0,1'
    command = job.instrument(sys.executable, script=script, multi=True)
    # Use IPv4 explicitly; macOS standalone rendezvous may resolve an IPv6 reverse name.
    command.remove('--standalone')
    command[3:3] = ['--rdzv_backend=static', f'--rdzv_endpoint=127.0.0.1:{port}']
    # Real CPU processes; GPU identifiers are accounting fixture values only.
    job.execute('optimization', [(command, '0,1')], 2)
    cost = aggregate(tmp_path)
    assert cost['flops_complete'] and cost['token_counter_coverage_complete']
    assert cost['training_tokens'] == 4 and cost['optimization_steps'] == 1
    active = [r for r in cost['stages'][0]['process_telemetry'] if r['forward_tokens']]
    assert {r['distributed_rank'] for r in active} == {0, 1}
    assert all(r['worker_id'] == '0' and r['finalized'] for r in active)


@pytest.mark.parametrize('workers', [1, 3])
def test_all_expected_legacy_worker_counters_are_required(tmp_path, workers):
    jsonl(tmp_path / 'stage_events.jsonl', [event()])
    for rank in range(workers):
        write(tmp_path / 'telemetry/a' / f'process-{rank}.json', counter())
    cost = aggregate(tmp_path)
    assert cost['flops_complete'] is (workers == 3)
    assert cost['token_counter_coverage_complete'] is (workers == 3)
    assert cost['training_tokens'] == (300 if workers == 3 else None)
    assert cost['stage_accounting_complete']  # Timing remains usable when counters are missing.


def test_finalized_idle_worker_is_valid_but_periodic_snapshot_is_not(tmp_path):
    jsonl(tmp_path / 'stage_events.jsonl', [event('attribution', gpus=2)])
    write(tmp_path / 'telemetry/a/process-0.json', counter(worker_id='0', distributed_rank=None, finalized=True))
    idle = counter(forward_tokens=0, autograd_tokens=0, optimization_steps=0,
                   estimated_flops=0., worker_id='1', distributed_rank=None, finalized=False)
    write(tmp_path / 'telemetry/a/process-1.json', idle)
    assert not aggregate(tmp_path)['flops_complete']
    idle['finalized'] = True
    write(tmp_path / 'telemetry/a/process-1.json', idle)
    cost = aggregate(tmp_path)
    assert cost['flops_complete'] and cost['token_counter_coverage_complete']
    assert cost['training_tokens'] == 100


def test_launcher_or_duplicate_rank_cannot_replace_a_missing_rank(tmp_path):
    jsonl(tmp_path / 'stage_events.jsonl', [event(gpus=2)])
    rank = counter(worker_id='0', distributed_rank=0, finalized=True)
    write(tmp_path / 'telemetry/a/process-0.json', rank)
    write(tmp_path / 'telemetry/a/process-parent.json', counter(forward_tokens=0, autograd_tokens=0,
        optimization_steps=0, estimated_flops=0., worker_id='0', distributed_rank=None, finalized=True))
    assert not aggregate(tmp_path)['flops_complete']
    write(tmp_path / 'telemetry/a/process-1.json', rank)
    assert not aggregate(tmp_path)['flops_complete']
    write(tmp_path / 'telemetry/a/process-1.json', counter(worker_id='0', distributed_rank=1, finalized=True))
    assert aggregate(tmp_path)['flops_complete']
    write(tmp_path / 'telemetry/a/process-1.json', counter(worker_id='0', distributed_rank=2, finalized=True))
    assert not aggregate(tmp_path)['flops_complete']


def test_preprocessing_counters_are_required_for_method_total(tmp_path):
    jsonl(tmp_path / 'stage_events.jsonl', [event('attribution', 'a', 1), event('optimization', 'b', 1)])
    write(tmp_path / 'telemetry/b/process-0.json', counter())
    cost = aggregate(tmp_path)
    assert not cost['flops_complete'] and not cost['token_counter_coverage_complete']
    assert cost['training_tokens'] is None
    assert cost['optimization_token_counter_coverage_complete']
    assert cost['optimization_stage_tokens'] == 100


def test_required_method_stage_is_checked_even_when_all_present_counters_exist(tmp_path):
    config = next(c for c in matrix() if c['method'] == 'ssft')
    write(tmp_path / 'run.json', config)
    jsonl(tmp_path / 'stage_events.jsonl', [event(gpus=1)])
    write(tmp_path / 'telemetry/a/process-0.json', counter())
    cost = aggregate(tmp_path)
    assert cost['missing_required_stages'] == ['attribution']
    assert not cost['stage_accounting_complete'] and not cost['flops_complete']


@pytest.mark.parametrize('include_native', [False, True])
def test_cored_requires_all_native_stages_and_worker_counters(tmp_path, include_native):
    jsonl(tmp_path / 'stage_events.jsonl', [event('cored_pipeline')])
    if include_native:
        jsonl(tmp_path / 'artifacts/cored/timings.jsonl',
              [dict(stage=name, status='complete', seconds=2) for name in sorted(CORED_STAGES)])
        for name in CORED_STAGES - {'prepare'}:
            for rank in range(1 if name == 'merge' else 3):
                write(tmp_path / 'telemetry/a' / name / f'process-{rank}.json', counter())
    cost = aggregate(tmp_path)
    assert cost['stage_accounting_complete'] is include_native
    assert cost['flops_complete'] is include_native
    assert cost['token_counter_coverage_complete'] is include_native
    assert cost['missing_required_stages'] == ([] if include_native else sorted(CORED_STAGES))


def test_tables_do_not_display_missing_accounting_as_zero_gpu_hours(tmp_path):
    configs = [c for c in matrix() if c['setting'] == 'A']
    config = next(c for c in configs if c['method'] == 'cored')
    directory = tmp_path / 'runs' / config['run_id']
    write(directory / 'run.json', config)
    write(directory / 'final_checkpoint.json', {})
    jsonl(directory / 'stage_events.jsonl', [event('cored_pipeline')])
    tables(tmp_path / 'runs', tmp_path / 'reports', configs)
    line = next(s for s in (tmp_path / 'reports/cost_setting_A.md').read_text().splitlines() if s.startswith('|cored|'))
    assert line.split('|')[2:4] == ['incomplete', 'incomplete']
