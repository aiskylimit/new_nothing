"""SSFT sharding preserves native segment selection and target filtering."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rebuttal.common import ROOT, jsonl, matrix, rows, write
from rebuttal.method_prep import merge_ssft, ssft


@pytest.fixture
def attribution_backend(monkeypatch):
    # Exercise the real stage_prep/segment-selection code with deterministic
    # per-example IG scores; no CUDA/model download is needed for these tests.
    import torch
    for folder in ('', 'Attribution', 'SelectiveSFT'):
        monkeypatch.syspath_prepend(str(ROOT / 'SegmentSelectiveSFT' / folder))
    calls = []

    class Calculator:
        def __init__(self, model, gradient_checkpointing):
            self.model = torch.nn.Linear(1, 1)
            self.tokens_forwarded = self.tokens_backwarded = 0

    def attribute(calculator, row, **kwargs):
        calls.append((row['id'], kwargs))
        calculator.tokens_forwarded += 10
        calculator.tokens_backwarded += 20
        return [[float(i + 1), -float(i + 1)] for i in range(len(row['segments']))], 'ok'

    monkeypatch.setitem(sys.modules, 'grad_analyze', SimpleNamespace(
        IntegratedGradientsAttribution=Calculator, attribute_example=attribute))
    return calls


def examples():
    return [dict(id=str(i), question=f'Q{i}', solution='Step one.\n\nStep two.\n\nConclusion.',
                 answer='x' * 201 if i == 2 else str(i)) for i in range(8)]


def test_three_shards_match_serial_native_selection(tmp_path, attribution_backend):
    source, serial, merged = (tmp_path / name for name in ('source.jsonl', 'serial.jsonl', 'merged.jsonl'))
    jsonl(source, examples())
    settings = {'ig_steps': 50, 'mass': .7, 'coherence': .8}
    ssft('local-model', source, serial, settings)
    parts = [tmp_path / f'part-{i}.jsonl' for i in range(3)]
    for i, part in enumerate(parts):
        ssft('local-model', source, part, settings, i, 3)
    merge_ssft(source, parts, merged)
    assert rows(merged) == rows(serial)
    assert [r['id'] for r in rows(merged)] == ['0', '1', '3', '4', '5', '6', '7']
    assert all(r['selected_spans_ids'] == [1, 2] for r in rows(merged))
    assert json.loads(Path(str(serial)+'.adaptations.json').read_text()) == json.loads(
        Path(str(merged)+'.adaptations.json').read_text())
    assert all(k['ig_steps'] == 50 and k['ig_batch_size'] == 1 and k['max_input_tokens'] == 0
               for _, k in attribution_backend)


def test_empty_shards_support_two_sample_smoke(tmp_path, attribution_backend):
    source, output = tmp_path / 'source.jsonl', tmp_path / 'selected.jsonl'
    jsonl(source, examples()[:2])
    parts = [tmp_path / f'part-{i}.jsonl' for i in range(3)]
    for i, part in enumerate(parts):
        ssft('local-model', source, part, shard=i, shards=3)
    assert rows(parts[2]) == []
    assert len(attribution_backend) == 2
    merge_ssft(source, parts, output)
    assert [r['id'] for r in rows(output)] == ['0', '1']


@pytest.mark.parametrize('problem', ['missing', 'duplicate', 'wrong_shard', 'missing_file', 'all_rejected'])
def test_merge_rejects_incomplete_or_invalid_results(tmp_path, problem):
    source = tmp_path / 'source.jsonl'
    values = examples()[:2]
    if problem == 'all_rejected':
        values = [dict(r, answer='x' * 201) for r in values]
    jsonl(source, values)
    parts = [tmp_path / f'part-{i}.jsonl' for i in range(3)]
    for i, part in enumerate(parts):
        jsonl(part, [] if problem == 'all_rejected' else values[i::3])
        write(str(part)+'.cost.json', {})
    if problem == 'missing':
        jsonl(parts[0], [])
    elif problem == 'duplicate':
        jsonl(parts[0], [values[0], values[0]])
    elif problem == 'wrong_shard':
        jsonl(parts[0], [values[1]])
    elif problem == 'missing_file':
        parts[0].unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        merge_ssft(source, parts, tmp_path / 'selected.jsonl')
    assert not (tmp_path / 'selected.jsonl').exists()


def test_runner_dispatches_attribution_to_three_distinct_gpus(tmp_path, monkeypatch):
    from rebuttal import runner
    job = object.__new__(runner.Job)
    job.config = next(c for c in matrix() if c['run_id'] == 'A_ssft_7b_seed42')
    job.root, job.model, job.input, job.gpus = tmp_path, tmp_path / 'model', tmp_path / 'data', '3,4,5'
    job.model.mkdir()
    write(job.model / 'config.json', {})
    write(job.root / 'run.json', job.config)
    monkeypatch.setattr(runner, 'python_for', lambda method: sys.executable)
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *a, **kw: '{}')

    class Dispatched(Exception):
        pass

    def execute(stage, commands, gpu_count):
        assert stage == 'attribution' and gpu_count == 3
        assert [gpu for command, gpu in commands] == ['3', '4', '5']
        destinations = []
        for i, (command, _) in enumerate(commands):
            assert command[command.index('--shard') + 1] == str(i)
            assert command[command.index('--shards') + 1] == '3'
            destinations.append(command[command.index('--output') + 1])
        assert len(set(destinations)) == 3
        raise Dispatched

    monkeypatch.setattr(job, 'execute', execute)
    with pytest.raises(Dispatched):
        job.train()
