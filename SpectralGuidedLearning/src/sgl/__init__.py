"""Step-level selection and allocation for long chain-of-thought distillation.

Every method in this package is one path through the same pipeline:

    data -> (trace transform) -> segment -> per-step signal -> select (mask) -> allocate (weights)
         -> objective -> train -> eval

    sgl.data         sources, prompts, tokenization (`python -m sgl.data.prepare`)
    sgl.segment      step segmentation and char -> token span mapping
    sgl.signals      per-step signals: spectral strength + entropy (`capture`), answer gain
    sgl.selection    binary step selection -> loss_mask (`build_masks`)
    sgl.allocation   step weights -> loss_weights (IWC / IWC-Stable, `build`)
    sgl.transforms   trace transforms and extra token fields (provenance, step transitions)
    sgl.training     masked / weighted objectives, collator, trainers (`train`, `backends.unsloth`)
    sgl.eval         benchmarks, vLLM generation, graders, result tables (`evaluate`, `compare`)
    sgl.diagnostics  pre-training checks of the weighting signals

The original P-ALIGN and Segment-Selective SFT code is kept unmodified under references/.
"""
