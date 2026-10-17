#!/usr/bin/env bash
set -euo pipefail

echo "Non-thinking evaluation is not part of the main paper reproduction matrix." >&2
echo "Call eval/evaluate_math.py with --no_thinking and local paths for a separate ablation." >&2
exit 2
