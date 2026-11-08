#!/usr/bin/env bash
# The supported local setup and run entrypoint is project_commands.sh.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${PROJECT_ROOT}/project_commands.sh"
