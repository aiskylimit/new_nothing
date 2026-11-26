import json

import pytest

from sgl import cli
from sgl.diagnostics import efficiency


def test_runner_records_each_stage_only_when_asked(tmp_path):
    (tmp_path / "ok.py").write_text("print('ok')\n")
    config = {"name": "arm", "run_name": "arm-t", "track": "t", "pipeline": ["prepare"],
              "run": {"log_dir": "logs"}, "stages": {"prepare": {"script": "ok.py"}}}
    assert cli.run_pipeline(config, ["prepare"], tmp_path, False, False, False) == 0
    assert not (tmp_path / "eff").exists()
    config["run"]["efficiency_dir"] = "eff"
    assert cli.run_pipeline(config, ["prepare"], tmp_path, False, False, False) == 0
    (record,) = [json.loads(line) for line in (tmp_path / "eff" / "arm-t.jsonl").read_text().splitlines()]
    assert record["stage"] == "prepare" and record["exit_code"] == 0 and record["seconds"] >= 0


def test_table_shares_prepare_and_leaves_unknown_totals_empty(tmp_path):
    def write(run, stages, pipeline):
        lines = [{"stage": s, "exit_code": code, "seconds": sec, "track": "t", "checkpoint": f"ck/{run}",
                  "results_dir": "res", "pipeline": pipeline, "peak_gpu_mib": 2048} for s, sec, code in stages]
        (tmp_path / "eff").mkdir(exist_ok=True)
        (tmp_path / "eff" / f"{run}.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))

    write("gain", [("prepare", 60, 0), ("answer_gain", 600, 0), ("train", 5, 1), ("train", 120, 0)],
          ["prepare", "answer_gain", "train", "eval"])
    write("nll", [("vanilla", 6, 0), ("train", 120, 0)], ["prepare", "vanilla", "train", "eval"])
    write("partial", [("train", 120, 0)], ["prepare", "answer_gain", "train"])
    write("zeroshot", [("eval", 30, 0)], ["eval"])
    (tmp_path / "res" / "nll").mkdir(parents=True)
    (tmp_path / "res" / "nll" / "summary.json").write_text(json.dumps(
        [{"benchmark": "math500", "pass@1": 0.5, "length": 100.0, "truncation_rate": 0.1, "n_generations": 3}]))

    efficiency.main(["--efficiency-dir", str(tmp_path / "eff"), "--root", str(tmp_path)])
    rows = {row["run"]: row for row in json.loads((tmp_path / "eff" / "table.json").read_text())}
    assert rows["gain"]["total_seconds"] == pytest.approx(780)  # the failed train attempt does not count
    assert rows["nll"]["total_seconds"] == pytest.approx(186)  # prepare taken from the same track
    assert rows["partial"]["total_seconds"] is None  # answer_gain never recorded
    assert rows["zeroshot"]["total_seconds"] == 0
    assert "100 / 10% / 50.0%" in (tmp_path / "eff" / "table.md").read_text()
