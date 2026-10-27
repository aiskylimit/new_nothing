# new_nothing

## eval-impact

Local B200 pipeline for **OPD vs OPD + IMPACT**, using **LLaMA2-7B → TinyLLaMA-1.1B** on Dolly, with teacher LoRA preparation, BI analysis and final training/BI reports. MATH-500/GSM8K are disabled in the current default run.

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
