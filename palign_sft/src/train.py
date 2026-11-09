import time

from llamafactory.model.model_utils import checkpointing as _lf_checkpointing
from llamafactory.train.tuner import run_exp

# transformers 5.x may pass keyword arguments that LLaMA-Factory 0.9.5's
# patched gradient-checkpointing method does not accept.
_orig_gc_enable = _lf_checkpointing._gradient_checkpointing_enable


def _gradient_checkpointing_enable_compat(
    self,
    gradient_checkpointing_kwargs=None,
    use_unsloth_gc=False,
    every_n_layers=1,
    offload=False,
    **kwargs,
):
    return _orig_gc_enable(
        self,
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
        use_unsloth_gc=use_unsloth_gc,
    )


_lf_checkpointing._gradient_checkpointing_enable = _gradient_checkpointing_enable_compat


def main():
    started = time.time()
    run_exp()
    elapsed = time.time() - started
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"train finished in {elapsed:.1f}s ({hours:02d}:{minutes:02d}:{seconds:02d})")


def _mp_fn(index):
    run_exp()


if __name__ == "__main__":
    main()
