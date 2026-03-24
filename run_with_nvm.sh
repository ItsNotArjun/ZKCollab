#!/usr/bin/env bash
# run_with_nvm.sh — Wrapper that loads NVM, selects Node 22, then runs run_pipeline.sh
set -euo pipefail

export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
export HARDHAT_DISABLE_TELEMETRY_PROMPT=true
# shellcheck disable=SC1091
[ -s "$NVM_DIR/nvm.sh" ] && source "$NVM_DIR/nvm.sh"

nvm use 22 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_pipeline.sh"
