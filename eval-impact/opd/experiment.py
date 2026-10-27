"""Prepare a sequential, reviewable experiment matrix; execute only with --execute."""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from .data import save_json
from .recipes import TEACHER, STUDENT

PROFILES = {
    "llama2-tinyllama": {"teacher": TEACHER, "student": STUDENT},
    "math-small": {"teacher": "Qwen/Qwen2.5-Math-7B-Instruct", "student": "Qwen/Qwen2.5-Math-1.5B-Instruct"},
    "tiny": {"teacher": "Qwen/Qwen2.5-1.5B-Instruct", "student": "Qwen/Qwen2.5-0.5B-Instruct"},
}



def make_plan(args):
    profile = PROFILES[args.profile]
    matched = args.profile == "llama2-tinyllama"
    if matched and not args.teacher_adapter:
        raise ValueError("Use opd.b200 for automatic teacher SFT, or pass a prepared --teacher-adapter")
    args.steps = args.steps or (0 if matched else 256)
    args.grad_accum = args.grad_accum or (32 if matched else 8)
    args.bi_max_length = args.bi_max_length or (256 if matched else 1024)
    out = Path(args.out_root) / args.profile
    jobs = []
    for seed in args.seeds:
        bi_runs = []
        for domain in args.domains:
            data = Path(args.data_root) / domain
            bi = out / "bi" / f"{domain}-seed{seed}"
            cmd = [sys.executable, "-m", "opd.bi", "collect", "--model", profile["teacher"],
                   "--data", str(data / "calibration.jsonl"), "--out", str(bi),
                   "--max-length", str(args.bi_max_length), "--seed", str(seed)]
            if matched:
                cmd += ["--adapter", args.teacher_adapter]
            jobs.append({"stage": "bi", "name": f"bi-{domain}-{seed}", "command": cmd})
            bi_runs.append(f"{domain}-seed{seed}={bi / 'summary.json'}")
            for method in ("opd", "opd_impact"):
                run = out / domain / method / f"seed{seed}"
                cmd = [sys.executable, "-m", "opd.train", "--student", profile["student"],
                       "--teacher", profile["teacher"], "--method", method, "--data", str(data / "train.jsonl"),
                       "--calibration", str(data / "calibration.jsonl"), "--out", str(run),
                       "--steps", str(args.steps), "--grad-accum", str(args.grad_accum),
                       "--kd-rate", "1", "--kd-objective", "reverse_kl", "--kd-temperature", "1",
                       "--max-new-tokens", str(args.max_new_tokens), "--max-length", str(args.max_length),
                       "--seed", str(seed), "--bi-max-length", str(args.bi_max_length)]
                if matched:
                    # Replace the legacy objective/budget flags with the declared recipe.
                    start = cmd.index("--steps")
                    cmd = cmd[:start] + ["--preset", "dskd-tiny", "--grad-accum", str(args.grad_accum),
                        "--teacher-adapter", args.teacher_adapter, "--seed", str(seed),
                        "--bi-max-length", str(args.bi_max_length)]
                    if args.steps:
                        cmd += ["--steps", str(args.steps)]
                else:
                    cmd += ["--preset", "legacy"]
                if method == "opd_impact":
                    cmd += ["--bi-summary", str(bi / "summary.json")]
                jobs.append({"stage": "train", "name": f"{domain}-{method}-{seed}", "command": cmd})
                cmd = [sys.executable, "-m", "opd.evaluate", "--model", profile["student"],
                       "--adapter", str(run / "checkpoint-pending"), "--out", str(run / "eval"),
                       "--max-new-tokens", str(args.eval_max_new_tokens), "--seed", str(seed)]
                if args.math_eval:
                    jobs.append({"stage": "eval", "name": f"eval-{domain}-{method}-{seed}", "command": cmd,
                                 "checkpoint_from": str(run / "latest.json")})
        if len(bi_runs) > 1:
            jobs.append({"stage": "bi", "name": f"bi-compare-{seed}",
                         "command": [sys.executable, "-m", "opd.bi", "compare", "--runs", *bi_runs,
                                     "--out", str(out / "bi" / f"comparison-seed{seed}.json")]})
    for role in (("student", "teacher") if args.math_eval else ()):
        jobs.append({"stage": "eval", "name": f"base-{role}",
                     "command": [sys.executable, "-m", "opd.evaluate", "--model", profile[role],
                                 "--out", str(out / f"base-{role}-eval"),
                                 "--max-new-tokens", str(args.eval_max_new_tokens), "--seed", "42"]})
        if matched and role == "teacher":
            jobs[-1]["command"] += ["--adapter", args.teacher_adapter]
    return {"schema_version": 1, "profile": profile, "arguments": vars(args), "jobs": jobs,
            "budget_note": "Equal optimizer steps and rollout attempts; actual generated tokens/time logged separately.",
            "data_note": "All training seeds here use the same saved data sample sets; prepare separate data roots for calibration-subset seed experiments."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher-adapter", default=None)
    p.add_argument("--math-eval", action="store_true")
    p.add_argument("--data-root", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--profile", choices=list(PROFILES), default="llama2-tinyllama")
    p.add_argument("--domains", nargs="+", default=["dolly-dskd"])
    p.add_argument("--seeds", nargs="+", type=int, default=[10])
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--grad-accum", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--bi-max-length", type=int, default=0)
    p.add_argument("--eval-max-new-tokens", type=int, default=1024)
    p.add_argument("--plan", default="experiment-plan.json")
    p.add_argument("--stage", choices=["bi", "train", "eval", "all"], default="all")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    plan = make_plan(args)
    save_json(args.plan, plan)
    for job in plan["jobs"]:
        if args.stage in ("all", job["stage"]):
            print(f"[{job['name']}] {shlex.join(job['command'])}", flush=True)
            if args.execute:
                command = list(job["command"])
                if job.get("checkpoint_from"):
                    checkpoint = json.loads(Path(job["checkpoint_from"]).read_text())["checkpoint"]
                    command[command.index("--adapter") + 1] = checkpoint
                subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
