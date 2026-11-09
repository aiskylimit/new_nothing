"""Exercise shell orchestration with fake uv/Python, never network or GPUs."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def launcher_environment(tmp_path, installed):
    binary = tmp_path/'bin'
    binary.mkdir()
    template = tmp_path/'fake-python'
    template.write_text(f'#!{sys.executable}\n' + '''import json, os, sys
args = sys.argv[1:]
module = args[1] if args[:1] == ['-m'] else ''
stage = {'rebuttal.provision':'download','rebuttal.data':'prepare',
         'rebuttal.preflight':'preflight','rebuttal.statistics':'statistics',
         'rebuttal.cost':'cost'}.get(module, args[2] if len(args)>2 else '')
with open(os.environ['TEST_EVENTS'], 'a') as f:
    f.write(json.dumps({'tool':'python','stage':stage,'args':args,
        'offline':os.environ.get('HF_HUB_OFFLINE'),
        'ssft_python':os.environ.get('REBUTTAL_PYTHON_SSFT')})+'\\n')
if stage == os.environ.get('TEST_FAIL_STAGE'): sys.exit(7)
''')
    template.chmod(0o755)
    uv = binary/'uv'
    uv.write_text(f'#!{sys.executable}\n' + '''import json, os, shutil, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ['TEST_EVENTS'], 'a') as f:
    f.write(json.dumps({'tool':'uv','args':args})+'\\n')
if args[0] == 'venv':
    env = Path(args[-1]); (env/'bin').mkdir(parents=True)
    shutil.copy2(os.environ['TEST_PYTHON_TEMPLATE'], env/'bin/python')
    (env/'bin/activate').write_text('export PATH="'+str(env/'bin')+':$PATH"\\n')
elif '-r' in args:
    assert Path(args[args.index('-r')+1]).is_file()
if os.environ.get('TEST_FAIL_SETUP') and args[:2] == ['pip','install']: sys.exit(8)
''')
    uv.chmod(0o755)
    env_root = tmp_path/'envs'
    if installed:
        folder = env_root/'rebuttal/bin'
        folder.mkdir(parents=True)
        import shutil
        shutil.copy2(template, folder/'python')
        (folder/'activate').write_text(f'export PATH="{folder}:$PATH"\n')
    env = {key:value for key,value in os.environ.items()
           if not key.startswith('REBUTTAL_') and key not in {'PY','GPUS'}}
    env.update(PATH=str(binary)+os.pathsep+os.environ['PATH'], REBUTTAL_ENV_ROOT=str(env_root),
        REBUTTAL_ASSETS=str(tmp_path/'assets'), REBUTTAL_DATA=str(tmp_path/'prepared'),
        REBUTTAL_RUNS=str(tmp_path/'runs'), REBUTTAL_REPORTS=str(tmp_path/'reports'),
        REBUTTAL_LAUNCHER_LOGS=str(tmp_path/'logs'), GPUS='0,1,2',
        TEST_EVENTS=str(tmp_path/'events.jsonl'), TEST_PYTHON_TEMPLATE=str(template))
    return env


def invoke(tmp_path, args, env):
    return subprocess.run(['bash', str(ROOT/'rebuttal/project_commands.sh'), *args],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=15)


def events(tmp_path):
    path = tmp_path/'events.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@pytest.mark.parametrize('mode', [[], ['--offline']])
def test_offline_end_to_end_never_installs_or_downloads(tmp_path, mode):
    env = launcher_environment(tmp_path, installed=True)
    result = invoke(tmp_path, ['endtoend', *mode], env)
    assert result.returncode == 0, result.stderr + result.stdout
    recorded = events(tmp_path)
    assert [e['stage'] for e in recorded] == ['prepare','preflight','all','statistics','cost']
    assert all(e['tool']=='python' and e['offline']=='1' for e in recorded)
    assert recorded[1]['args'][-2:] == ['--gpus','0,1,2']
    assert all(e['ssft_python'] == str(tmp_path/'envs/SegmentSelectiveSFT/bin/python') for e in recorded)
    assert (tmp_path/'logs/endtoend.log').is_file()


def test_online_end_to_end_bootstraps_six_requirements_before_download(tmp_path):
    env = launcher_environment(tmp_path, installed=False)
    result = invoke(tmp_path, ['endtoend','--online'], env)
    assert result.returncode == 0, result.stderr + result.stdout
    recorded = events(tmp_path)
    requirements = [e['args'][e['args'].index('-r')+1] for e in recorded
                    if e['tool']=='uv' and '-r' in e['args']]
    assert requirements == ['rebuttal.txt','palign.txt','sft.txt','sgl.txt','SegmentSelectiveSFT.txt','cored.txt']
    python = [e for e in recorded if e['tool']=='python']
    assert [e['stage'] for e in python] == ['download','prepare','preflight','all','statistics','cost']
    assert python[0]['offline']=='0' and all(e['offline']=='1' for e in python[1:])
    assert recorded[0]['tool']=='uv'


@pytest.mark.parametrize('failed,expected', [('prepare',['prepare']), ('preflight',['prepare','preflight'])])
def test_failure_stops_before_training_and_report(tmp_path, failed, expected):
    env = launcher_environment(tmp_path, installed=True)
    env['TEST_FAIL_STAGE'] = failed
    result = invoke(tmp_path, ['endtoend','--offline'], env)
    assert result.returncode == 7
    assert [e['stage'] for e in events(tmp_path)] == expected


def test_failed_setup_never_downloads_or_trains(tmp_path):
    env = launcher_environment(tmp_path, installed=False)
    env['TEST_FAIL_SETUP']='1'
    result = invoke(tmp_path, ['endtoend','--online'], env)
    assert result.returncode == 8
    assert all(e['tool']=='uv' for e in events(tmp_path))


def test_existing_all_action_still_only_calls_training_and_eval_runner(tmp_path):
    env = launcher_environment(tmp_path, installed=True)
    result = invoke(tmp_path, ['all','--run-id','A_cored_7b_seed42'], env)
    assert result.returncode == 0, result.stderr
    recorded = events(tmp_path)
    assert len(recorded)==1 and recorded[0]['stage']=='all'
    assert recorded[0]['args'][-2:] == ['--run-id','A_cored_7b_seed42']


def test_managed_download_attestation_option_is_forwarded(tmp_path):
    env = launcher_environment(tmp_path, installed=True)
    result = invoke(tmp_path, ['download','--attest-managed'], env)
    assert result.returncode == 0, result.stderr
    assert events(tmp_path)[0]['args'][-1]=='--attest-managed'


def test_unknown_mode_fails_before_any_external_action(tmp_path):
    env = launcher_environment(tmp_path, installed=False)
    result = invoke(tmp_path, ['endtoend','--unknown'], env)
    assert result.returncode == 2 and not events(tmp_path)
