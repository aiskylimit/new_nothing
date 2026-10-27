> Historical notes for the earlier Qwen/math pipeline. Current default: LLaMA2 → TinyLLaMA, OPD vs OPD + IMPACT on Dolly; math evaluation disabled. Follow [README_B200.md](README_B200.md) and [DSKDV2_REFERENCE.md](DSKDV2_REFERENCE.md).

# Direct OPD validation

The current runner compares OPD with OPD + IMPACT. Earlier ETA-based runs outside validation/direct are superseded and do not validate this objective.

Current evidence is in validation/direct:

- Unit tests check analytical reverse KL, prompt masking, teacher gradient isolation, equal distributions, output-head padding, tokenizer rejection, sampled-ID preservation, and OPD + IMPACT gradients without dual-space modules.
- Existing BI, sampling and archived legacy regression tests remain available.
- Real sampled-rollout CPU training uses tiny random models with shared tokenizer, student hidden size 16/depth 2 and teacher hidden size 24/depth 3. Both methods use pure OPD; no reference CE or ETA/projectors.
- Tokenizer/config compatibility of the actual Qwen Math pair is checked without downloading model weights. See tokenizer_check.json.

The main pretrained runs and native MATH-500/GSM8K evaluation remain unexecuted because this host has no CUDA. Tiny fixture scores are execution checks only.

Sixteen unit tests passed. The direct OPD+IMPACT checkpoint after resume matched uninterrupted training bitwise for student parameters, IMPACT projection, RNG and data pointer. Baseline checkpoint contains no projector. See validation/direct/verification.json.


## Local B200 packaging

Three recent user templates were inspected (env, download manifest, project_commands). Their names/hashes are recorded under validation/b200/source_templates.json. The generated B200 templates use /mnt/local paths, uv/Python 3.11, explicit torch 2.9.0 CUDA 12.8 wheels and native lm-eval 0.4.13.
Downloaded real raw Dolly/GSM8K/MATH-500 data, prepared local train/calibration/test files, checked test sizes and prompt separation, then ran native lm-eval with network disabled on one question from each benchmark using a tiny random CPU model. The local task aliases, graders, sample logs and final aggregation completed successfully. This debug fixture is not pretrained benchmark evidence.
Checked cumulative token/time reporting across an actual resumed CPU training run. Shell syntax and local-path plans passed. The B200 CUDA preflight and pretrained GPU experiment have not run on this Mac host.

Twenty unit tests now pass, including local-path B200 planning, manifest destination checks, native-task aggregation, and eval cache invalidation when the checkpoint advances after resume.
