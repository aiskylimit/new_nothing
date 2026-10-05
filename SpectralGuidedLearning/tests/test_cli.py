import argparse

import pytest

from sgl import cli
from sgl.cli import StageError, args_to_argv, plan_stage, select_stages


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--opt", action=argparse.BooleanOptionalAction)
    p.add_argument("--flag", action="store_true")
    p.add_argument("--value")
    p.add_argument("--many", nargs="+")
    return p


def test_args_to_argv_maps_each_value_kind():
    argv = args_to_argv(
        {"opt": False, "flag": False, "value": 3, "many": [1, "b"], "skipped": None, "under_score": "x"},
        parser(),
    )
    assert argv == ["--no-opt", "--value", "3", "--many", "1", "b", "--under-score", "x"]
    assert args_to_argv({"opt": True, "flag": True}, parser()) == ["--opt", "--flag"]
    assert args_to_argv({"model_name": "m"}, raw_keys=True) == ["--model_name", "m"]
    assert args_to_argv({"flag": False}) == []  # no parser: False is just omitted


def test_boolean_for_an_option_that_takes_a_value_is_an_error():
    with pytest.raises(StageError, match="takes a value"):
        args_to_argv({"value": False}, parser())
    with pytest.raises(StageError, match="mapping"):
        args_to_argv({"value": {"a": 1}}, parser())


def base_config(**stage):
    return {"name": "x", "run": {"gpus": [0, 1], "log_dir": "logs"},
            "stages": {"s": {"module": "sgl.eval.compare", **stage}}, "pipeline": ["s"]}


def test_plan_validates_args_against_the_module_parser():
    plan = plan_stage(base_config(args={"results-dir": "r"}), "s")
    assert plan["cmd"][-4:] == ["-m", "sgl.eval.compare", "--results-dir", "r"]
    assert plan["env"]["CUDA_VISIBLE_DEVICES"] == "0"  # python stages: first run GPU only
    assert plan["validated"]
    with pytest.raises(StageError, match="unrecognized"):
        plan_stage(base_config(args={"no-such-flag": 1}), "s")


def test_torchrun_plan_uses_every_gpu_and_derives_gradient_accumulation():
    config = base_config(launcher="torchrun", effective_batch=32, args={"results-dir": "r"})
    config["stages"]["s"]["module"] = "sgl.nonexistent.module"  # not importable: args unchecked
    plan = plan_stage(config, "s")
    assert plan["env"]["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert plan["cmd"][1:5] == ["-m", "torch.distributed.run", "--nproc_per_node", "2"]
    assert plan["cmd"][plan["cmd"].index("--gradient-accumulation-steps") + 1] == "16"
    assert not plan["validated"]
    config["stages"]["s"]["effective_batch"] = 3
    with pytest.raises(StageError, match="not divisible"):
        plan_stage(config, "s")


def test_stage_spec_errors():
    with pytest.raises(StageError, match="exactly one"):
        plan_stage({"stages": {"s": {"module": "a", "script": "b"}}}, "s")
    with pytest.raises(StageError, match="launcher"):
        plan_stage({"stages": {"s": {"module": "a", "launcher": "mpirun"}}}, "s")
    with pytest.raises(StageError, match="not defined"):
        plan_stage({"stages": {}}, "s")


def test_select_stages():
    config = {"pipeline": ["a", "b", "c"], "stages": {"a": {}, "b": {}, "c": {}, "opt": {}}}
    assert select_stages(config, None, None) == ["a", "b", "c"]
    assert select_stages(config, None, "b") == ["b", "c"]
    assert select_stages(config, ["opt", "a"], None) == ["opt", "a"]
    with pytest.raises(StageError):
        select_stages(config, ["zzz"], None)
    with pytest.raises(StageError):
        select_stages({"pipeline": ["a"], "stages": {}}, None, None)


@pytest.mark.parametrize("family", cli.FAMILIES)
def test_every_shipped_config_validates(family, capsys):
    assert cli.family_configs(family), f"no configs for {family}"
    assert cli.main(["check"], family) == 0, capsys.readouterr().out


@pytest.mark.parametrize("family", cli.FAMILIES)
def test_every_shipped_config_dry_runs(family, capsys):
    for path in cli.family_configs(family):
        assert cli.main(["run", str(path), "--dry-run"], family) == 0, path


def test_a_config_only_runs_under_its_own_family(capsys):
    assert cli.main(["show", "r1-qwen-1.5b"], "palign") == 0
    assert cli.main(["show", str(cli.REPO_DIR / "configs/palign/r1-qwen-1.5b.yaml")], "ssft") == 2
    assert "palign run" in capsys.readouterr().err


def test_overrides_after_options_are_accepted(capsys):
    assert cli.main(["show", "r1-qwen-1.5b-palign/sft-nll", "train_seed=7"], "sgl") == 0
    assert "seed: 7" in capsys.readouterr().out
    argv = ["run", "r1-qwen-1.5b-palign/sft-nll", "--dry-run", "train_seed=7", "--stages", "train"]
    assert cli.main(argv, "sgl") == 0


def test_run_executes_stages_skips_created_outputs_and_stops_on_failure(tmp_path, capsys):
    config = {
        "name": "t", "run": {"gpus": [], "log_dir": "logs"}, "pipeline": ["ok", "done", "bad", "never"],
        "stages": {
            "ok": {"script": "ok.py"}, "done": {"script": "ok.py", "creates": "exists.txt"},
            "bad": {"script": "bad.py"}, "never": {"script": "ok.py"},
        },
    }
    (tmp_path / "ok.py").write_text("print('ran ok')\n")
    (tmp_path / "bad.py").write_text("raise SystemExit(3)\n")
    (tmp_path / "exists.txt").write_text("")
    assert cli.run_pipeline(config, config["pipeline"], tmp_path, False, False, False) == 3
    out = capsys.readouterr().out
    assert "ran ok" in out and "skip: exists.txt" in out and "stage never" not in out
    assert (tmp_path / "logs" / "t-ok.log").read_text().count("ran ok") == 1
