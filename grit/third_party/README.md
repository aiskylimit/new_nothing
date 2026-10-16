# Vendored official scorers

Official scoring code, copied from the benchmarks' own repositories so that GRIT
scores exactly like the authors. Both are MIT-licensed (see each `LICENSE`).

| Folder | Source | Commit | Files | Changes |
|---|---|---|---|---|
| `olympiadbench/` | github.com/OpenBMB/OlympiadBench | `ba5b26a7e2849940b598a9159c1190daa2b9175f` | `eval/auto_scoring_judge.py` (the scorer the README designates), `eval/scoring_examples.json` | none |
| `livecodebench/` | github.com/LiveCodeBench/LiveCodeBench | `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24` | `lcb_runner/evaluation/{testing_util,compute_code_generation_metrics,pass_k_utils}.py` | two import paths in `compute_code_generation_metrics.py` (marked `# GRIT:`) |

Glue code lives in `scripts/eval_lib.py`, and copies these parts of the official repos verbatim:
- OlympiadBench `inference/code/evaluators/evaluator.py` (prompt) and `inference/judge.py`
  (answer extraction, precision from the `error` field).
- LiveCodeBench `prompts/code_generation.py` (CodeQwenInstruct prompt, the style
  `lm_styles.py` assigns to Qwen2.5-Instruct), `utils/extraction_utils.py` (`extract_code`) and
  `benchmarks/code_generation.py` (`get_evaluation_sample`).

The only addition is a 30 s timeout around each OlympiadBench judgement (the official scorer
has none, and a sympy hang would stall the run). A timeout counts as incorrect.
`tests/test_eval.py` checks the vendored judge against the authors' 36 scoring examples.
