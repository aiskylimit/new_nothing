import time

from llamafactory.model.model_utils import checkpointing as _lf_checkpointing
from llamafactory.train.tuner import run_exp

# transformers 5.17 Trainer.train() passes every_n_layers/offload;
# LLaMA-Factory 0.9.5's patched enable() does not accept them.
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
    t0 = time.time()
    run_exp()
    elapsed = time.time() - t0
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    print(f"train finished in {elapsed:.1f}s ({h:02d}:{m:02d}:{s:02d})")


def _mp_fn(index):
    run_exp()


if __name__ == "__main__":
    main()
