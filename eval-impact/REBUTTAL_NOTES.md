# Evidence for OPD + IMPACT and BI calibration questions

The comparison is direct OPD versus the same OPD objective plus IMPACT. DSKDv2 informs setup only; it is not the baseline objective.

| Question | Controlled experiment | Evidence |
|---|---|---|
| Can IMPACT augment OPD? | Same teacher/student and sampled-prefix reverse KL; with/without IMPACT | Config, rollouts, OPD/IMPACT losses, checkpoints, math eval outputs |
| BI domain sensitivity? | Same teacher/revision/context/N=256: Dolly, GSM8K, mixed | Per-example scores, all-layer CI/rank, rank correlations, Top-K overlap |
| BI subset sensitivity? | Three saved held-out subsets per domain | IDs/hashes, shared-ID count, score/rank changes, bootstrap Top-K frequency |
| Selection vs weighting? | Selection variants with uniform weights, then fixed BI set with BI/uniform weighting | Frozen scores, configs, downstream outcomes |
| K sensitivity? | K=2/4/6/8 with the same calibration file | Layer lists, weights, eval |

This direct OPD experiment uses a shared tokenizer. It establishes applicability in that setting only. Cross-tokenizer OPD requires an explicit output-distribution bridge and a separate experiment.
The frozen teacher does not change because optimizer steps pass. BI can change with its inputs, context length, prompt/response mode or checkpoint. Report these axes separately.
Stable rankings support a claim limited to tested settings; unstable rankings motivate domain-specific/mixed calibration. No universal stability claim follows.
CPU fixture metrics under validation/direct test implementation only. Prior ETA/DSKDv2 artifacts are not evidence for this corrected comparison. No pretrained accuracy is supplied.
