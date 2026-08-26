#!/usr/bin/env bash

if [[ -z "${RUN_NAME:-}" ]]; then
  echo "_common.sh: RUN_NAME must be set before sourcing." >&2
  exit 1
fi

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
ENV_FILE="${PROJECT_DIR}/.env"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi
