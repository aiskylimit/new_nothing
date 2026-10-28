"""Sequential, offline, local-path B200 pipeline: BI, paired OPD, eval, final report."""
import argparse
import hashlib
import json
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path
from .data import save_json

MODELS = {"llama2-tinyllama": ("Llama-2-7b-hf", "TinyLlama-1.1B-intermediate-step-1431k-3T"),
          "math-small": ("Qwen2.5-Math-7B-Instruct", "Qwen2.5-Math-1.5B-Instruct"),
          "tiny": ("Qwen2.5-1.5B-Instruct", "Qwen2.5-0.5B-Instruct")}


def preflight(out):
    import torch
    import importlib.metadata
    if not torch.cuda.is_available():
        raise RuntimeError("B200 CUDA device unavailable")
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    if "B200" not in name.upper() or capability[0] != 10:
        raise RuntimeError(f"Expected B200 / compute capability 10.x, found {name} {capability}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 unavailable on selected GPU")
    if tuple(map(int, torch.version.cuda.split(".")[:2])) < (12, 8):
        raise RuntimeError("Install the CUDA 12.8 torch wheel via setup_b200.sh")
    x = torch.randn(64, 64, device="cuda:0", dtype=torch.bfloat16, requires_grad=True)
    (x @ x).square().mean().backward()
    torch.cuda.synchronize()
    result = {"gpu": name, "capability": capability, "torch": torch.__version__,
              "cuda_runtime": torch.version.cuda, "arch_list": torch.cuda.get_arch_list(),
              "vram_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
              "python": platform.python_version(), "bf16_backward": True,
              "packages": {p: importlib.metadata.version(p) for p in
                  ("transformers", "peft", "accelerate", "datasets", "numpy")}}
    save_json(out, result)
    del x
    torch.cuda.empty_cache()
    return result


def verify_model(path):
    path = Path(path)
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"Local model missing: {path}")
    index = path / "model.safetensors.index.json"
    weights = ([path / n for n in set(json.loads(index.read_text())["weight_map"].values())]
               if index.is_file() else [path / "model.safetensors"])
    if any(not p.is_file() or not p.stat().st_size for p in weights):
        raise FileNotFoundError(f"Incomplete model weights in {path}; run download first")


def freeze_inputs(plan):
    """Prevent resumed jobs from silently using replaced local assets."""
    root = Path(plan["project_root"])
    files = list((root / "data/opd").rglob("*.jsonl")) + list((root / "raw").rglob("*.parquet"))
    records = {}
    if plan["arguments"].get("teacher_adapter"):
        from .recipes import adapter_signature
        records[plan["arguments"]["teacher_adapter"]] = {"adapter_sha256": adapter_signature(plan["arguments"]["teacher_adapter"])}
    for name in (plan["teacher"], plan["student"]):
        model = Path(name)
        files.extend(model.glob("*.json"))
        for weights in model.glob("*.safetensors"):
            info = weights.stat()
            records[str(weights)] = {"bytes": info.st_size, "mtime_ns": info.st_mtime_ns}
    for path in sorted(set(files)):
        records[str(path)] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    path = Path(plan["run_root"]) / "inputs.json"
    if path.exists() and json.loads(path.read_text()) != records:
        raise ValueError("Local model/data assets changed for this run; use a new RUN_NAME")
    save_json(path, records)


def make_plan(args):
    root = Path(args.project_root).resolve()
    run = root / "outputs" / args.run_name
    teacher_name, student_name = MODELS[args.profile]
    teacher, student = root / "models" / teacher_name, root / "models" / student_name
    jobs, cases, bi_specs = [], [], []
    reference = args.profile == "llama2-tinyllama"
    preset = "dskd-tiny" if reference else "legacy"
    methods = getattr(args, "methods", ["opd", "opd_impact"])
    eval_enabled = getattr(args, "math_eval", not reference)
    grad_accum = args.grad_accum or (32 if reference else 8)
    steps = args.steps or (0 if reference else 256)
    teacher_adapter = getattr(args, "teacher_adapter", None)
    adapter_pointer = None

    def job(stage, name, module, params, artifact):
        jobs.append({"stage": stage, "name": name, "command": [args.python_bin, "-m", module, *map(str, params)],
                     "artifact": str(artifact)})

    if reference and not teacher_adapter:
        sft = run / "teacher_sft"
        params = ["--model", teacher, "--data", root / "data/opd/dolly-dskd/train.jsonl",
                  "--out", sft, "--seed", 10]
        if getattr(args, "teacher_steps", 0):
            params += ["--steps", args.teacher_steps]
        job("teacher", "teacher-sft", "opd.sft", params, sft / "status.json")
        jobs[-1]["train_dir"] = str(sft)
        teacher_adapter = sft / "checkpoint-pending"
        adapter_pointer = sft / "latest.json"

    def bind_teacher(flag):
        if adapter_pointer:
            jobs[-1].setdefault("adapter_dependencies", {})[flag] = str(adapter_pointer)

    for domain in args.domains:
        # Frozen training calibration is always seed 42; other subsets analyze stability.
        for seed in sorted(set([42, *args.bi_seeds])):
            target = run / "bi" / f"{domain}-calib{seed}"
            bi_params = ["--adapter", teacher_adapter] if teacher_adapter else []
            job("bi", f"bi-{domain}-{seed}", "opd.bi", ["collect", "--model", teacher,
                "--data", root / "data/opd" / domain / f"calibration-seed{seed}.jsonl",
                "--out", target, "--seed", seed, "--max-length", 256 if reference else 1024,
                *bi_params], target / "summary.json")
            bind_teacher("--adapter")
            bi_specs.append(f"{domain}-calib{seed}={target / 'summary.json'}")
    if len(bi_specs) > 1:
        job("bi", "bi-comparisons", "opd.bi", ["compare", "--runs", *bi_specs,
            "--out", run / "bi/comparisons.json"], run / "bi/comparisons.json")

    def evaluation(name, model, folder, domain, method, seed=42, train_dir=None):
        if not eval_enabled:
            if train_dir is not None:
                cases.append({"name": name, "domain": domain, "method": method, "seed": seed,
                              "eval_dir": None, "train_dir": str(train_dir)})
            return
        params = ["--model", model, "--out", folder, "--local-data-root", root / "raw",
                  "--seed", seed, "--max-new-tokens", args.eval_max_new_tokens]
        if train_dir is not None:
            params += ["--adapter", train_dir / "checkpoint-pending"]
        job("eval", name, "opd.evaluate", params, folder / "evaluation_manifest.json")
        jobs[-1]["eval_dir"] = str(folder)
        if train_dir is not None:
            jobs[-1]["checkpoint_from"] = str(train_dir / "latest.json")
        cases.append({"name": name, "domain": domain, "method": method, "seed": seed,
                      "eval_dir": str(folder), "train_dir": str(train_dir) if train_dir else None})

    evaluation("base-student", student, run / "base/student/eval", "base", "student")
    evaluation("base-teacher", teacher, run / "base/teacher/eval", "base", "teacher")
    if eval_enabled and teacher_adapter:
        jobs[-1]["command"] += ["--adapter", str(teacher_adapter)]
        bind_teacher("--adapter")
    for domain in args.domains:
        for seed in args.seeds:
            for method in methods:
                train = run / domain / method / f"seed{seed}"
                params = ["--teacher", teacher, "--student", student, "--method", method,
                    "--data", root / "data/opd" / domain / "train.jsonl",
                    "--calibration", root / "data/opd" / domain / "calibration-seed42.jsonl",
                    "--out", train, "--preset", preset, "--grad-accum", grad_accum,
                    "--seed", seed, "--max-minutes", args.max_minutes]
                if steps:
                    params += ["--steps", steps]
                if not reference:
                    params += ["--kd-rate", 1, "--kd-objective", "reverse_kl", "--kd-temperature", 1]
                if teacher_adapter:
                    params += ["--teacher-adapter", teacher_adapter]
                if method == "opd_impact":
                    params += ["--bi-summary", run / "bi" / f"{domain}-calib42/summary.json"]
                name = f"{domain}-{method}-{seed}"
                job("train", name, "opd.train", params, train / "status.json")
                jobs[-1]["train_dir"] = str(train)
                bind_teacher("--teacher-adapter")
                evaluation("eval-" + name, student, train / "eval", domain, method, seed, train)
    return {"schema_version": 2, "comparison": ["OPD" if m == "opd" else "OPD + IMPACT" for m in methods],
            "math_eval_enabled": eval_enabled, "teacher_adapter": str(teacher_adapter) if teacher_adapter else None,
            "profile": args.profile, "project_root": str(root), "run_root": str(run),
            "teacher": str(teacher), "student": str(student), "arguments": vars(args),
            "cases": cases, "jobs": jobs}


def artifact_complete(job):
    path = Path(job["artifact"])
    if not path.is_file():
        return False
    if job["stage"] in ("train", "teacher"):
        status = json.loads(path.read_text())
        return status.get("state") == "completed"
    if job["stage"] == "eval":
        return len(list(Path(job["eval_dir"]).rglob("results*.json"))) == 1
    return True


def execute_job(job, ledger, logs):
    command = list(job["command"])
    if job.get("checkpoint_from"):
        latest = json.loads(Path(job["checkpoint_from"]).read_text())
        command[command.index("--adapter") + 1] = latest["checkpoint"]
    for flag, pointer in job.get("adapter_dependencies", {}).items():
        dependency = json.loads(Path(pointer).read_text())
        status = json.loads((Path(pointer).parent / "status.json").read_text())
        if status.get("state") != "completed":
            raise RuntimeError("Teacher SFT is incomplete; resume the teacher stage first")
        command[command.index(flag) + 1] = dependency["checkpoint"]
    fingerprint = hashlib.sha256(json.dumps(command).encode()).hexdigest()
    previous = ledger.get(job["name"], {})
    if previous.get("fingerprint") == fingerprint and previous.get("state") == "completed" and artifact_complete(job):
        print(f"Skip completed {job['name']}", flush=True)
        return
    if job["stage"] == "eval" and Path(job["eval_dir"]).exists():
        # Preserve old/partial predictions when the checkpoint advances on resume.
        history = Path(job["eval_dir"]).parent / "eval_history" / str(time.time_ns())
        history.parent.mkdir(parents=True, exist_ok=True)
        Path(job["eval_dir"]).rename(history)
    if job.get("train_dir"):
        latest = Path(job["train_dir"]) / "latest.json"
        if latest.is_file():
            command += ["--resume", json.loads(latest.read_text())["checkpoint"]]
    logs.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    record = {"state": "running", "fingerprint": fingerprint, "command": command}
    ledger[job["name"]] = record
    save_json(logs.parent / "jobs.json", ledger)
    print(shlex.join(command), flush=True)
    with (logs / (job["name"] + ".log")).open("a") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    record.update(state="completed" if result.returncode == 0 else "failed",
                  exit_code=result.returncode, seconds=time.monotonic() - start)
    save_json(logs.parent / "jobs.json", ledger)
    if result.returncode:
        raise RuntimeError(f"{job['name']} failed; see {logs / (job['name'] + '.log')}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["plan", "preflight", "teacher", "bi", "train", "eval", "report", "all"])
    p.add_argument("--project-root", required=True)
    p.add_argument("--run-name", default="llama2-tinyllama-opd")
    p.add_argument("--profile", choices=list(MODELS), default="llama2-tinyllama")
    p.add_argument("--domains", nargs="+", default=["dolly-dskd"])
    p.add_argument("--seeds", nargs="+", type=int, default=[10])
    p.add_argument("--bi-seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--grad-accum", type=int, default=0)
    p.add_argument("--methods", nargs="+", choices=["opd", "opd_impact"], default=["opd", "opd_impact"])
    p.add_argument("--teacher-adapter", default=None, help="Existing local Dolly teacher adapter; skips teacher SFT")
    p.add_argument("--teacher-steps", type=int, default=0, help="Optional teacher SFT smoke-test cap")
    p.add_argument("--math-eval", action="store_true", help="Explicitly opt into MATH-500/GSM8K; disabled by default")
    p.add_argument("--eval-max-new-tokens", type=int, default=1024)
    p.add_argument("--max-minutes", type=float, default=0)
    p.add_argument("--plan-out", default=None, help="For plan stage, save elsewhere while retaining target B200 paths")
    p.add_argument("--python", dest="python_bin", default=sys.executable)
    args = p.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        raise ValueError("run-name must be a directory name")
    plan = make_plan(args)
    run = Path(plan["run_root"])
    existing = Path(args.plan_out) if args.stage == "plan" and args.plan_out else run / "plan.json"
    if existing.exists():
        old = json.loads(existing.read_text())
        if old["cases"] != plan["cases"] or old["jobs"] != plan["jobs"]:
            raise ValueError("Run plan changed; use a new RUN_NAME to avoid mixing results")
    save_json(existing, plan)
    if args.stage == "plan":
        for job in plan["jobs"]:
            print(f"[{job['stage']}] {shlex.join(job['command'])}")
        return
    if args.stage != "report":
        preflight(run / "environment.json")
        verify_model(plan["teacher"])
        verify_model(plan["student"])
        # Fail before spending time on teacher SFT if actual local tokenizers differ.
        from transformers import AutoTokenizer
        from .losses import require_shared_tokenizer
        teacher_tok = AutoTokenizer.from_pretrained(plan["teacher"], local_files_only=True)
        student_tok = AutoTokenizer.from_pretrained(plan["student"], local_files_only=True)
        require_shared_tokenizer(student_tok, teacher_tok)
        save_json(run / "tokenizers.json", {"verified_shared": True, "vocab_size": len(student_tok),
                  "teacher": plan["teacher"], "student": plan["student"]})
        freeze_inputs(plan)
    if args.stage == "preflight":
        return
    ledger_path = run / "jobs.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    try:
        for job in plan["jobs"]:
            if args.stage in ("all", job["stage"]):
                execute_job(job, ledger, run / "logs")
    finally:
        if args.stage in ("all", "report", "eval"):
            from .final_results import aggregate
            aggregate(existing, run / "final_results")


if __name__ == "__main__":
    main()
