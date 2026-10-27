# Corrected experiment: OPD versus OPD + IMPACT

The new active runner uses direct response-masked KL on fresh student-sampled prefixes. Method names are opd and opd_impact. No DSKDv2 criterion, ETA alignment, logit projectors or pseudo-label objective is imported or optimized. The baseline has student parameters only; the augmented method adds the IMPACT hidden-state projection and auxiliary loss.

Defaults: pure reverse-KL OPD, temperature 1; Qwen2.5-Math-7B-Instruct to Qwen2.5-Math-1.5B-Instruct. A 1.5B to 0.5B Qwen profile is available. Runtime tokenizer compatibility checks prevent accidentally calling cross-tokenizer alignment plain OPD. All actual tokenizer IDs are scored; unassigned head padding is excluded.

Prepared train/calibration data, per-sample BI, all-layer ranks, Top-K/weights, bootstrap intervals, domain/subset comparison, checkpoints, resume, native evaluation commands and reports remain available. DSKDv2 source files inherited from the original ZIP remain for historical comparison; they are not used by opd.train.

See README_OPD.md for the current commands, VALIDATION.md for evidence, and REBUTTAL_NOTES.md for the measured scope. The previous ETA-based experiment plans/results are superseded.
