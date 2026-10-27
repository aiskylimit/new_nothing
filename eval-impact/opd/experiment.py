"""Prepare a sequential, reviewable experiment matrix; execute only with --execute."""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from .data import save_json

PROFILES = {
    "math-small": {"teacher": "Qwen/Qwen2.5-Math-7B-Instruct", "student": "Qwen/Qwen2.5-Math-1.5B-Instruct"},
    "tiny": {"teacher": "Qwen/Qwen2.5-1.5B-Instruct", "student": "Qwen/Qwen2.5-0.5B-Instruct"},
}



def make_plan(args):
    profile = PROFILES[args.profile]
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
                if method == "opd_impact":
                    cmd += ["--bi-summary", str(bi / "summary.json")]
                jobs.append({"stage": "train", "name": f"{domain}-{method}-{seed}", "command": cmd})
                cmd = [sys.executable, "-m", "opd.evaluate", "--model", profile["student"],
                       "--adapter", str(run / f"checkpoint-{args.steps}"), "--out", str(run / "eval"),
                       "--max-new-tokens", str(args.eval_max_new_tokens), "--seed", str(seed)]
                jobs.append({"stage": "eval", "name": f"eval-{domain}-{method}-{seed}", "command": cmd})
        if len(bi_runs) > 1:
            jobs.append({"stage": "bi", "name": f"bi-compare-{seed}",
                         "command": [sys.executable, "-m", "opd.bi", "compare", "--runs", *bi_runs,
                                     "--out", str(out / "bi" / f"comparison-seed{seed}.json")]})
    for role in ("student", "teacher"):
        jobs.append({"stage": "eval", "name": f"base-{role}",
                     "command": [sys.executable, "-m", "opd.evaluate", "--model", profile[role],
                                 "--out", str(out / f"base-{role}-eval"),
                                 "--max-new-tokens", str(args.eval_max_new_tokens), "--seed", "42"]})
    return {"schema_version": 1, "profile": profile, "arguments": vars(args), "jobs": jobs,
            "budget_note": "Equal optimizer steps and rollout attempts; actual generated tokens/time logged separately.",
            "data_note": "All training seeds here use the same saved data sample sets; prepare separate data roots for calibration-subset seed experiments."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--profile", choices=list(PROFILES), default="math-small")
    p.add_argument("--domains", nargs="+", default=["gsm8k", "mixed", "dolly"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--bi-max-length", type=int, default=1024)
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
                subprocess.run(job["command"], check=True)


if __name__ == "__main__":
    main()
