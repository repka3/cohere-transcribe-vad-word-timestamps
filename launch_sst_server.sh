#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${1:-7700}"

exec .venv/bin/python -m uvicorn sst_asr_endpoint_openclaw_hermes:app --host 0.0.0.0 --port "$PORT"
