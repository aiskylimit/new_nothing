import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
from pyarrow import parquet

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ("palign", "sft", "sgl", "SegmentSelectiveSFT", "cored")


def test_every_folder_has_loader_manifests_with_shared_eval_destinations():
    common = None
    for folder in FOLDERS:
        dependencies = (ROOT / f"{folder}.txt").read_text().splitlines()
        assert dependencies[:2] == ["# manager: uv", "# python: 3.11"]
        manifest = (ROOT / folder / "download.txt").read_text().splitlines()
        assert manifest[0] == "#d"
        commands = [shlex.split(line) for line in manifest if line and not line.startswith("#")]
        assert all(len(command) == 3 for command in commands)
        assert all("@PROJECT@" in command[2] for command in commands)
        assert any(command == ["--hf", "Qwen/Qwen2.5-7B-Instruct",
                               f"/mnt/local/@PROJECT@/{folder}/models/Qwen2.5-7B-Instruct"]
                   for command in commands)
        assert ["--hf", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                f"/mnt/local/@PROJECT@/{folder}/models/DeepSeek-R1-Distill-Qwen-1.5B"] in commands
        benchmarks = [command for command in commands if "/palign/datasets/" in command[2]
                      and not command[2].endswith("/P-ALIGN")]
        assert len(benchmarks) == 6
        ood = [command for command in benchmarks if command[0] == "--url"]
        assert len(ood) == 2
        assert any("/ARC-Challenge/test-00000-of-00001.parquet" in command[1]
                   and command[2].endswith("/ARC-Challenge/") for command in ood)
        assert any("/csitfun/LogiQA2.0/" in command[1]
                   and command[1].endswith("/logiqa/DATA/LOGIQA/test.txt")
                   and command[2].endswith("/LogiQA2.0/") for command in ood)
        if common is None:
            common = benchmarks
        assert benchmarks == common
    assert (ROOT / "palign.txt").read_bytes() == (ROOT / "sft.txt").read_bytes()


def test_sgl_reads_provisioned_snapshot_without_hub_or_cache(tmp_path, monkeypatch):
    import datasets

    def unexpected(*args, **kwargs):
        raise AssertionError("local snapshot attempted a Hub/cache loader")

    monkeypatch.setattr(datasets, "load_dataset", unexpected)
    data = tmp_path / "data"
    data.mkdir()
    expected = [{"question": "Q", "deepseek_thinking_trajectory": "reasoning",
                 "deepseek_attempt": "answer"}]
    parquet.write_table(pa.Table.from_pylist(expected), data / "train.parquet")
    spec = importlib.util.spec_from_file_location("sgl_local", ROOT / "sgl/src/sgl/local_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.load_source_dataset(str(tmp_path)) == expected


def test_palign_materializes_downloaded_training_and_eval_assets(tmp_path):
    assets = tmp_path / "assets"
    source = assets / "datasets/P-ALIGN/P-ALIGN.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps([{"instruction": "I", "input": "Q", "output": "R"}]))
    specs = {"AIME_2024": 30, "aime_2025": 30, "aimo-validation-amc": 83, "MATH-500": 500}
    for name, count in specs.items():
        folder = assets / "datasets" / name / "data"
        folder.mkdir(parents=True)
        parquet.write_table(pa.Table.from_pylist([
            {"problem": f"{name} {i}", "answer": str(i)} for i in range(count)
        ]), folder / "train.parquet")
    venv = tmp_path / "venv/bin"
    venv.mkdir(parents=True)
    (venv / "activate").write_text("")
    destination = tmp_path / "normalized"
    env = os.environ.copy()
    env.update(PALIGN_ASSET_ROOT=str(assets), DATA_DIR=str(destination),
               PALIGN_VENV=str(venv.parent), ASSET_ROOT="/nonexistent/inherited/old-root")
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    result = subprocess.run(["bash", str(ROOT / "palign/project_commands.sh"), "data"],
                            env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert (destination / "palign_sft_qwen2.5-7b.json").read_bytes() == source.read_bytes()
    for name, count in {"aime24": 30, "aime25": 30, "amc12": 83, "math500": 500}.items():
        assert len((destination / "raw" / f"{name}.jsonl").read_text().splitlines()) == count
