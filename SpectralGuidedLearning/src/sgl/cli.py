"""Command line of the three method families: `sgl`, `palign` and `ssft`.

Each command runs the experiment configs of its own family (configs/<family>/, `family:` key), so
a run is always launched by the name of the method it reproduces:

    sgl    show  r1-qwen-1.5b-palign/iwc-gain-l05          # resolved config (path or name under configs/sgl/)
    sgl    check                                           # validate every config of the family
    sgl    run   r1-qwen-1.5b-palign/iwc-gain-l05 --dry-run
    palign run   r1-qwen-1.5b --stages train,eval          # P-ALIGN, configs/palign/
    ssft   run   r1-qwen-1.5b run.gpus=[0,1]               # Segment-Selective SFT, configs/ssft/

A config lists its stages in `pipeline` and describes each under `stages.<name>`:

    module: sgl.training.train        # run as `python -m <module>` (or `script: path.py`)
    launcher: torchrun                # python (default) | torchrun
    args: {data-path: ..., use-lora: true}   # -> --data-path ... --use-lora
    creates: checkpoints/x            # skipped when this path exists (unless --force)
    python: /envs/other/bin/python    # interpreter for this stage (default: run.python, else this one)
    gpus: [0, 1]                      # sets CUDA_VISIBLE_DEVICES; default: all of run.gpus for
                                      # torchrun stages, the first one for python stages
    effective_batch: 32               # torchrun: gradient-accumulation-steps = 32 / (bs * #gpus)
    env: {VAR: value}
    cwd: some/dir                     # run from here (relative paths in args resolve against it)
    raw_keys: true                    # pass keys verbatim (--model_name_or_path) for reference scripts

Every stage is one subprocess whose output is also written to <run.log_dir>/<run_name>-<stage>.log.
Arguments are checked against the module's own argparse parser (`build_parser()`), so a typo in a
config fails before anything runs.
"""
import argparse
import contextlib
import importlib
import io
import os
import shlex
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from sgl.config import ConfigError, load_config

SRC_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = SRC_DIR.parent
FAMILIES = ("sgl", "palign", "ssft")


class StageError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- stage -> argv


def _option_actions(parser: argparse.ArgumentParser | None) -> dict[str, argparse.Action]:
    if parser is None:
        return {}
    return {
        option: action
        for action in parser._actions  # argparse exposes no public accessor for its actions
        for option in action.option_strings
    }


def args_to_argv(args: dict | None, parser: argparse.ArgumentParser | None = None,
                 raw_keys: bool = False) -> list[str]:
    """{key: value} -> argv. Keys may use '_' or '-' (normalised to '-' unless raw_keys).

    True -> --key, False -> --no-key for BooleanOptionalAction flags (omitted for plain
    store_true flags), None -> omitted, list -> --key v1 v2 ..., anything else -> --key value.
    Without a parser (reference scripts), False is omitted.
    """
    actions = _option_actions(parser)
    argv: list[str] = []
    for key, value in (args or {}).items():
        name = str(key).lstrip("-")
        flag = "--" + (name if raw_keys else name.replace("_", "-"))
        if value is None:
            continue
        if isinstance(value, bool):
            action = actions.get(flag)
            if action is not None and action.nargs != 0:
                # e.g. YAML reads a bare `off`/`no` as false: never drop it silently
                raise StageError(f"argument {key!r} takes a value but got the boolean {value!r} (quote it in YAML)")
            if value:
                argv.append(flag)
            elif isinstance(action, argparse.BooleanOptionalAction):
                argv.append("--no-" + flag[2:])
            continue
        if isinstance(value, (list, tuple)):
            argv.append(flag)
            argv.extend(_scalar(item) for item in value)
            continue
        if isinstance(value, dict):
            raise StageError(f"argument {key!r} is a mapping; stage args must be scalars or lists")
        argv.extend([flag, _scalar(value)])
    return argv


def _scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def module_parser(module: str) -> argparse.ArgumentParser | None:
    """The module's build_parser(), or None when it has none or cannot be imported here."""
    try:
        imported = importlib.import_module(module)
    except Exception:  # missing optional dependency, or one that refuses to load here (Unsloth without a GPU)
        return None
    build = getattr(imported, "build_parser", None)
    return build() if callable(build) else None


def validate_argv(parser: argparse.ArgumentParser, argv: list[str]) -> str | None:
    """argparse's error message for argv, or None if it parses."""
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr):
            parser.parse_args(argv)
    except SystemExit as exc:
        if exc.code:
            return stderr.getvalue().strip().splitlines()[-1] if stderr.getvalue().strip() else "invalid arguments"
    return None


# ----------------------------------------------------------------------------- planning


def _gpu_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return str(value).replace(",", " ").split()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", 0))
        return sock.getsockname()[1]


def plan_stage(config: dict, name: str, parser_cache: dict | None = None) -> dict:
    """Resolve one stage into {name, cmd, env, log, creates}. Raises StageError on a bad spec."""
    stages = config.get("stages") or {}
    if name not in stages:
        raise StageError(f"stage {name!r} is not defined under `stages`")
    spec = stages[name] or {}
    module, script = spec.get("module"), spec.get("script")
    if bool(module) == bool(script):
        raise StageError(f"stage {name!r}: give exactly one of `module` or `script`")
    launcher = spec.get("launcher", "python")
    if launcher not in ("python", "torchrun"):
        raise StageError(f"stage {name!r}: launcher must be python or torchrun, not {launcher!r}")
    run = config.get("run") or {}
    # torchrun stages take every run GPU; single-process stages the first one, unless they say otherwise
    run_gpus = _gpu_list(run.get("gpus"))
    default_gpus = run_gpus if launcher == "torchrun" else run_gpus[:1]
    gpus = _gpu_list(spec["gpus"]) if "gpus" in spec else default_gpus

    args = dict(spec.get("args") or {})
    if spec.get("effective_batch") is not None:
        if launcher != "torchrun":
            raise StageError(f"stage {name!r}: effective_batch needs launcher: torchrun")
        per_device = int(args.get("per-device-batch-size", args.get("per_device_batch_size", 1)))
        n_proc = max(len(gpus), 1)
        effective = int(spec["effective_batch"])
        if effective % (per_device * n_proc):
            raise StageError(
                f"stage {name!r}: effective_batch {effective} is not divisible by "
                f"per-device batch {per_device} x {n_proc} GPU(s)"
            )
        args["gradient-accumulation-steps"] = effective // (per_device * n_proc)

    parser = None
    if module:
        cache = parser_cache if parser_cache is not None else {}
        if module not in cache:
            cache[module] = module_parser(module)
        parser = cache[module]
    argv = [str(item) for item in spec.get("positional") or []] + args_to_argv(args, parser, bool(spec.get("raw_keys")))
    if parser is not None:
        error = validate_argv(parser, argv)
        if error:
            raise StageError(f"stage {name!r} ({module}): {error}")

    python = str(spec.get("python") or run.get("python") or sys.executable)
    target = ["-m", module] if module else [str(script)]
    if launcher == "torchrun":
        n_proc = max(len(gpus), 1)
        cmd = [  # `python -m torch.distributed.run` is torchrun, from the stage's own environment
            python, "-m", "torch.distributed.run",
            "--nproc_per_node", str(n_proc), "--nnodes", "1", "--node_rank", "0",
            "--rdzv_backend", "static", "--master_addr", "localhost", "--master_port", "{port}",
            *target, *argv,
        ]
    else:
        cmd = [python, "-u", *target, *argv]

    env = {key: str(value) for key, value in (run.get("env") or {}).items()}
    env.update({key: str(value) for key, value in (spec.get("env") or {}).items()})
    if gpus:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    if launcher == "torchrun":
        env.setdefault("DS_SKIP_CUDA_CHECK", "1")
    log_dir = Path(run.get("log_dir", "logs"))
    return {
        "name": name,
        "cmd": cmd,
        "env": env,
        "log": log_dir / f"{config.get('run_name') or config.get('name', 'run')}-{name}.log",
        "creates": spec.get("creates"),
        "validated": parser is not None,
        "launcher": launcher,
        "cwd": spec.get("cwd"),
    }


def select_stages(config: dict, only: list[str] | None, start: str | None) -> list[str]:
    """`--stages a,b` runs exactly those (pipeline or optional stages, in the given order);
    otherwise the pipeline, optionally from `start` on."""
    defined = config.get("stages") or {}
    if only:
        unknown = [name for name in only if name not in defined]
        if unknown:
            raise StageError(f"undefined stages {unknown}; defined: {sorted(defined)}")
        return list(only)
    pipeline = list(config.get("pipeline") or [])
    if not pipeline:
        raise StageError("config has an empty `pipeline`")
    missing = [name for name in pipeline if name not in defined]
    if missing:
        raise StageError(f"pipeline stages without a definition: {missing}")
    if start:
        if start not in pipeline:
            raise StageError(f"--from {start!r} is not in pipeline {pipeline}")
        pipeline = pipeline[pipeline.index(start):]
    return pipeline


# ----------------------------------------------------------------------------- execution


def child_env(extra: dict[str, str], launcher: str) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(SRC_DIR), env.get("PYTHONPATH")]))
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if launcher == "torchrun":
        # Cluster pods inject elastic rendezvous settings that make a single-node static
        # torchrun hang waiting on another worker; drop them.
        for key in [k for k in env if k.startswith(("PET_", "TORCHELASTIC_"))]:
            del env[key]
    env.update(extra)
    return env


def execute(plan: dict, cwd: Path) -> int:
    cmd = [part.replace("{port}", str(_free_port())) for part in plan["cmd"]]
    log = cwd / plan["log"]
    workdir = cwd / plan["cwd"] if plan["cwd"] else cwd
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        handle.write(f"$ {shlex.join(cmd)}\n")
        handle.flush()
        process = subprocess.Popen(
            cmd, cwd=workdir, env=child_env(plan["env"], plan["launcher"]),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                handle.write(line)
            return process.wait()
        except KeyboardInterrupt:  # never leave a training job running behind the CLI
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
            raise


def describe(plan: dict) -> str:
    env = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(plan["env"].items()))
    text = (env + " " if env else "") + shlex.join(plan["cmd"]).replace("'{port}'", "<free-port>")
    return f"(cd {shlex.quote(str(plan['cwd']))} && {text})" if plan["cwd"] else text


def run_pipeline(config: dict, stages: list[str], cwd: Path, force: bool, dry_run: bool, keep_going: bool) -> int:
    cache: dict = {}
    plans = [plan_stage(config, name, cache) for name in stages]  # validate everything first
    failures = []
    for plan in plans:
        print(f"\n=== [{config.get('name', 'run')}] stage {plan['name']} ===", flush=True)
        if plan["creates"] and not force and (cwd / plan["creates"]).exists():
            print(f"skip: {plan['creates']} exists (--force to rerun)")
            continue
        print(describe(plan), flush=True)
        if dry_run:
            continue
        code = execute(plan, cwd)
        if code:
            failures.append(plan["name"])
            print(f"stage {plan['name']} failed with exit code {code} (log: {plan['log']})", file=sys.stderr)
            if not keep_going:
                return code
    if failures:
        print(f"failed stages: {failures}", file=sys.stderr)
        return 1
    return 0


# ----------------------------------------------------------------------------- entry point


def resolve_config(name: str, family: str) -> Path:
    """A config path, or a name relative to configs/<family>/ (".yaml" optional)."""
    path = Path(name)
    if path.exists():
        return path
    for candidate in (REPO_DIR / "configs" / family / name, REPO_DIR / "configs" / family / f"{name}.yaml"):
        if candidate.is_file():
            return candidate
    raise ConfigError(f"no config {name!r} (neither a file nor under configs/{family}/)")


def family_configs(family: str) -> list[Path]:
    """Every runnable experiment of a family: configs with a pipeline, skipping building blocks."""
    root = REPO_DIR / "configs" / family
    paths = []
    for path in sorted(root.rglob("*.yaml")):
        relative = path.relative_to(root)
        if any(part.startswith("_") for part in relative.parts) or relative.parts[0] in ("methods", "tracks", "common"):
            continue
        paths.append(path)
    return paths


def load_family_config(name: str, family: str, overrides: list[str] | None = None) -> dict:
    config = load_config(resolve_config(name, family), overrides)
    declared = config.get("family", "sgl")
    if declared != family:
        raise ConfigError(f"{name} is a {declared!r} config: run it with `{declared} run`")
    return config


def build_parser(family: str = "sgl") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=family, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run the pipeline of an experiment config")
    run.add_argument("config", help=f"config path or name under configs/{family}/")
    run.add_argument("overrides", nargs="*", help="key.path=value")
    run.add_argument("--stages", help="comma-separated stages to run, in this order (pipeline or optional ones)")
    run.add_argument("--from", dest="start", help="start at this stage")
    run.add_argument("--force", action="store_true", help="rerun stages whose `creates` path exists")
    run.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    run.add_argument("--keep-going", action="store_true", help="continue after a failed stage")
    run.add_argument("--cwd", default=str(REPO_DIR),
                     help="directory relative paths resolve against (default: repo root)")

    show = commands.add_parser("show", help="print the resolved config")
    show.add_argument("config")
    show.add_argument("overrides", nargs="*")

    check = commands.add_parser("check", help="validate configs: references, stages and module arguments")
    check.add_argument("configs", nargs="*", help=f"default: every experiment under configs/{family}/")

    commands.add_parser("list", help=f"list the experiments under configs/{family}/")
    return parser


def main(argv: list[str] | None = None, family: str = "sgl") -> int:
    parser = build_parser(family)
    # overrides may sit anywhere among the options: collect the key=value leftovers
    args, extra = parser.parse_known_args(argv)
    stray = [item for item in extra if item.startswith("-") or "=" not in item]
    if stray:
        parser.error(f"unrecognized arguments: {' '.join(stray)}")
    if extra:
        if not hasattr(args, "overrides"):
            parser.error(f"{args.command} takes no overrides: {' '.join(extra)}")
        args.overrides = list(args.overrides) + extra
    try:
        if args.command == "list":
            root = REPO_DIR / "configs" / family
            for path in family_configs(family):
                print(path.relative_to(root).with_suffix(""))
            return 0
        if args.command == "show":
            config = load_family_config(args.config, family, args.overrides)
            print(yaml.safe_dump(config, sort_keys=False), end="")
            return 0
        if args.command == "check":
            failed = 0
            cache: dict = {}
            for path in args.configs or [str(p) for p in family_configs(family)]:
                try:
                    config = load_family_config(path, family)
                    select_stages(config, None, None)  # the pipeline itself must be well formed
                    plans = [plan_stage(config, name, cache) for name in config.get("stages") or {}]
                    unchecked = [plan["name"] for plan in plans if not plan["validated"]]
                    note = f" (not importable here, args unchecked: {unchecked})" if unchecked else ""
                    print(f"ok   {path}: {len(plans)} stages{note}")
                except (ConfigError, StageError) as exc:
                    failed += 1
                    print(f"FAIL {path}: {exc}")
            return 1 if failed else 0
        config = load_family_config(args.config, family, args.overrides)
        stages = select_stages(config, args.stages.split(",") if args.stages else None, args.start)
        return run_pipeline(config, stages, Path(args.cwd).resolve(), args.force, args.dry_run, args.keep_going)
    except (ConfigError, StageError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def sgl_main() -> int:
    return main(family="sgl")


def palign_main() -> int:
    return main(family="palign")


def ssft_main() -> int:
    return main(family="ssft")


if __name__ == "__main__":
    sys.exit(main())
