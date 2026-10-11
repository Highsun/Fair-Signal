#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$FAIR_SIGNAL_ROOT"
uv python install 3.12
uv sync --locked
uv run --frozen fair-signal doctor
