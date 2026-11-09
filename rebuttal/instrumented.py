"""Execute a native method without replacing its trainer, with common counters."""
import argparse
import runpy
import sys


def main():
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--module')
    group.add_argument('--script')
    p.add_argument('--unsloth-first', action='store_true')
    args, rest = p.parse_known_args()
    if args.unsloth_first:
        import unsloth  # noqa: F401
    from rebuttal.telemetry import activate, flush, install
    install()
    from transformers import AutoModelForCausalLM
    original = AutoModelForCausalLM.from_pretrained
    AutoModelForCausalLM.from_pretrained = classmethod(lambda cls, *a, **kw: activate(original(*a, **kw)))
    sys.argv = [args.module or args.script, *rest]
    try:
        if args.module:
            runpy.run_module(args.module, run_name='__main__')
        else:
            from pathlib import Path
            sys.path.insert(0, str(Path(args.script).resolve().parent))
            runpy.run_path(args.script, run_name='__main__')
    finally:
        flush()


if __name__ == '__main__':
    main()
