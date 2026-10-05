"""The configs must launch exactly what the legacy shell drivers launched for the published runs.

Runs the drivers in a scratch copy of the repo with fake `python` / `torchrun` executables that only
record their argv, and compares each recorded command, flag by flag, with what `plan_stage` builds
from the corresponding config. Path values are compared relative to the repo, numbers by value.

    python tests/regression/legacy_drivers.py
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TORCHRUN_FLAGS = ("--nproc_per_node", "--rdzv_backend", "--nnodes", "--node_rank", "--master_addr", "--master_port")
# The real interpreter by absolute path: the fake `python3` is first on PATH and would call itself.
RECORDER = """#!/usr/bin/env bash
REAL_PYTHON - "$(basename "$0")" "$@" <<'PY'
import json, os, sys
with open(os.environ["CAPTURE"], "a") as handle:
    handle.write(json.dumps({"cmd": sys.argv[1:]}) + "\\n")
PY
""".replace("REAL_PYTHON", sys.executable)


class Drivers:
    def __init__(self, root: Path):
        self.work, self.bin, self.capture, self.env = root / "repo", root / "bin", root / "capture.jsonl", root / "env"
        shutil.copytree(REPO, self.work, ignore=shutil.ignore_patterns(
            ".git", "logs", "results*", "paper", "experiments", "__pycache__"))
        self.bin.mkdir()
        for name in ("python", "python3", "torchrun"):
            (self.bin / name).write_text(RECORDER)
            (self.bin / name).chmod(0o755)
        (self.env / "bin").mkdir(parents=True)
        (self.env / "bin" / "activate").write_text("")
        (self.env / "bin" / "python").symlink_to(self.bin / "python")
        self.problems: list[str] = []

    def touch(self, relative: str) -> None:
        path = self.work / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}" if path.suffix == ".json" else "")

    def run(self, script: str, args=(), env=None) -> list[list[str]]:
        self.capture.write_text("")
        environment = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}", CAPTURE=str(self.capture),
                           VIRTUAL_ENV=str(self.env), PROJECT_ENV=str(self.env), GPUS="0", **(env or {}))
        result = subprocess.run(["bash", script, *args], cwd=self.work, env=environment, capture_output=True, text=True)
        if result.returncode:
            self.problems.append(f"legacy driver failed: {script} {list(args)}: {result.stderr[-400:]}")
        return [json.loads(line)["cmd"] for line in self.capture.read_text().splitlines()]

    def compare(self, label: str, legacy_cmd: list[str], family: str, config: str, stage: str,
                overrides=(), ignore=()) -> None:
        from sgl import cli

        plan = cli.plan_stage(cli.load_family_config(config, family, list(overrides)), stage)
        legacy_module, legacy_args = split_module(legacy_cmd[1:])
        planned_module, planned_args = split_module(plan["cmd"])
        old, new = self.flags(legacy_args), self.flags(planned_args)
        for key in (*ignore, *TORCHRUN_FLAGS):
            old.pop(key, None)
            new.pop(key, None)
        diffs = {key: (old.get(key), new.get(key)) for key in sorted(old.keys() | new.keys())
                 if not same_value(old.get(key), new.get(key))}
        if legacy_module != planned_module or diffs:
            self.problems.append(f"{label}: module {legacy_module} vs {planned_module}, differing flags {diffs}")

    def flags(self, argv: list[str]) -> dict[str, str]:
        out, index = {}, 0
        while index < len(argv):
            if argv[index].startswith("--"):
                values = []
                while index + 1 < len(argv) and not argv[index + 1].startswith("--"):
                    index += 1
                    values.append(argv[index].replace(f"{self.work}/", ""))
                out[argv[index - len(values)]] = " ".join(values)
            index += 1
        return out


def split_module(cmd: list[str]) -> tuple[str | None, list[str]]:
    """(module after the last -m, its arguments); torchrun's own flags come before the module."""
    if "-m" not in cmd:
        return None, cmd
    at = len(cmd) - 1 - cmd[::-1].index("-m")
    return cmd[at + 1], cmd[at + 2:]


def same_value(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return a is b
    try:
        return float(a) == float(b)
    except ValueError:
        return a == b


R1_ARMS = [
    ("sft-nll", "vanilla", ["--objective", "nll"]),
    ("sft-dft", "vanilla", ["--objective", "dft"]),
    ("iwc-nogate-l05", "iwc-stable-nogate-l05", []),
    ("iwc-gain-l05", "iwc-stable-gain-nogate-l05", []),
    ("iwc-shuf-l05", "iwc-stable-shuffled-nogate-l05", []),
    ("prov-nll-dft", "provenance", ["--prefix-objective", "nll", "--cont-objective", "dft"]),
    ("prov-dft-nll", "provenance", ["--prefix-objective", "dft", "--cont-objective", "nll"]),
    ("prov-nll-dft-j8", "provenance-j8",
     ["--prefix-objective", "nll", "--cont-objective", "dft", "--junction-objective", "nll"]),
]
LORA_ARMS = [
    ("sft-nll-lora", "vanilla", ["--objective", "nll"]),
    ("iwc-gain-l05-lora", "iwc-stable-gain-nogate-l05", []),
    ("iwc-nogate-l05-lora", "iwc-stable-nogate-l05", []),
]


def run_all(root: Path) -> list[str]:
    d = Drivers(root)
    track = "r1-qwen-1.5b-palign"
    data_env = {"TRACK": track, "DATASET_NAME": "references/P-ALIGN/data/palign_sft_qwen2.5-7b.json.gz",
                "ENABLE_THINKING": "false", "PALIGN_PROMPT": "true"}
    prepare, vanilla = d.run("scripts/data/data_r1-qwen-1.5b.sh", env=data_env)[:2]
    d.compare("prepare", prepare, "sgl", f"{track}/sft-nll", "prepare")
    d.compare("vanilla", vanilla, "sgl", f"{track}/sft-nll", "vanilla")
    merge_pass = d.run("scripts/capture/capture_r1-qwen-1.5b.sh", env={"TRACK": track})[-1]
    d.compare("capture", merge_pass, "sgl", f"{track}/iwc-nogate-l05", "capture", ignore=("--verify",))
    gated = d.run("scripts/masks/iwc_r1-qwen-1.5b.sh", env={
        "TRACK": track, "IWC_INTERPOLATION": "1.0", "IWC_TEMPERATURE": "2.0", "IWC_CLIP": "2.0"})[0]
    d.compare("weights iwc-stable", gated, "sgl", f"{track}/iwc-stable", "weights",
              ignore=("--variants", "--output-name", "--check-mass", "--seed"))

    for arm, variant, extra in R1_ARMS:
        d.touch(f"data/{track}/train-{variant}.jsonl")
        train = d.run("scripts/provenance/train_arm_r1-qwen-1.5b.sh", [arm, variant, *extra], env={"TRACK": track})
        d.compare(f"train {arm}", train[0], "sgl", f"{track}/{arm}", "train")
        evaluate = d.run("scripts/eval/eval_r1-qwen-1.5b.sh", [f"checkpoints/{arm}-{track}", f"{arm}-{track}"],
                         env={"ENABLE_THINKING": "false"})
        d.compare(f"eval {arm}", evaluate[0], "sgl", f"{track}/{arm}", "eval", ignore=("--lora-r",))
    evaluate = d.run("scripts/eval/eval_r1-qwen-1.5b.sh", [f"checkpoints/sft-nll-{track}", f"sft-nll-{track}-e43"],
                     env={"ENABLE_THINKING": "false", "EVAL_SEED": "43", "RESULTS_DIR": "results_evalseed"})
    d.compare("eval sft-nll seed 43", evaluate[0], "sgl", f"{track}/sft-nll", "eval",
              overrides=["eval_seed=43", "eval_suffix=-e43", "results_dir=results_evalseed"], ignore=("--lora-r",))

    for key, model in [("qwen25-7b", "Qwen/Qwen2.5-7B-Instruct"), ("qwen3-8b", "Qwen/Qwen3-8B")]:
        lora_track = f"{key}-palign"
        for arm, variant, extra in LORA_ARMS:
            d.touch(f"data/{lora_track}/train-{variant}.jsonl")
            train = d.run("scripts/iwc/train_lora.sh", [arm, variant, *extra],
                          env={"TRACK": lora_track, "MODEL_NAME": model})
            d.compare(f"train {lora_track}/{arm}", train[0], "sgl", f"{lora_track}/{arm}", "train")
            d.touch(f"checkpoints/{arm}-{lora_track}/adapter_config.json")
            evaluate = d.run("scripts/eval/eval_lora_palign.sh",
                             [f"checkpoints/{arm}-{lora_track}", f"{arm}-{lora_track}"], env={"BASE_MODEL": model})
            d.compare(f"eval {lora_track}/{arm}", evaluate[0], "sgl", f"{lora_track}/{arm}", "eval")
    return d.problems


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        problems = run_all(Path(tmp))
    print("\n".join(problems) or "all legacy commands reproduced")
