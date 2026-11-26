#!/usr/bin/env bash
# The answer-gain s1K driver now lives in project_commands.sh.
exec bash "$(dirname "${BASH_SOURCE[0]}")/project_commands.sh" "$@"
