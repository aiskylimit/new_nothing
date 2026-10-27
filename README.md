# new_nothing

## eval-impact

Local B200 pipeline for **OPD vs OPD + IMPACT**, including train/calibration data preparation, BI analysis, offline MATH-500/GSM8K evaluation and final Markdown/CSV/JSON reports.

- Code and run instructions: [eval-impact/README.md](eval-impact/README.md)
- Environment template: [impact-opd-b200.txt](impact-opd-b200.txt)
- Download manifest: [eval-impact/download.txt](eval-impact/download.txt)
- Run script: [eval-impact/project_commands.sh](eval-impact/project_commands.sh)

```bash
cd eval-impact
bash setup_b200.sh
bash project_commands.sh download
bash project_commands.sh all
```
