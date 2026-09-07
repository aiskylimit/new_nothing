#!/usr/bin/env bash
set -e
source /mnt/local/uvenvs/vlm-distill-eval/bin/activate
bash "scripts/eval/run_one_trained_checkpoint copy.sh"
bash "scripts/eval/run_one_trained_checkpoint copy 2.sh"
bash "scripts/eval/run_one_trained_checkpoint copy 3.sh"