"""Prints the LoRA dir with the best dev score of an SFT run, e.g.
    python tools/best_lora.py results/qwen3/general_domain/sft_4B_ee_v1_maven
-> results/qwen3/general_domain/sft_4B_ee_v1_maven/<settings>/<step>

finetune.py saves the LoRA at the end of every epoch into <settings>/<global step> and writes a
"BEST dev epoch <e> | score <s>" line to <settings>/log.txt whenever the dev score improves. Save and
evaluation run at the same step, so epoch e is the (e+1)-th step dir. log.txt is appended by every rerun:
only the lines after the last "EXP at" header count. A run continued with PEFT_PATH has its own
<settings>-from<step> dir (same dev set), so every <settings> dir competes on its best score."""
import glob
import os
import re
import sys

run = sys.argv[1].rstrip("/")
logs = sorted(glob.glob(os.path.join(run, "*", "log.txt")))
assert logs, f"{run}: no <settings>/log.txt"

best = None
for log in logs:
    text = open(log, encoding="utf-8").read()
    text = text[text.rfind(" EXP at "):]
    lines = re.findall(r"^BEST dev epoch (\d+) \| score ([\d.]+)", text, re.M)
    if not lines:
        print(f"skip {log}: no BEST dev line in its last run", file=sys.stderr)
        continue
    epoch, score = int(lines[-1][0]), float(lines[-1][1])
    settings = os.path.dirname(log)
    steps = sorted((d for d in os.listdir(settings) if d.isdigit()), key=int)
    assert epoch < len(steps), f"{settings}: best epoch {epoch} but only step dirs {steps}"
    if best is None or score > best[0]:
        best = (score, os.path.join(settings, steps[epoch]))

assert best is not None, f"{run}: no finished epoch in any <settings>/log.txt"
assert os.path.exists(os.path.join(best[1], "adapter_config.json")), f"{best[1]}: no adapter_config.json"
print(best[1])
