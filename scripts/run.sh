#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$FAIR_SIGNAL_ROOT"
exec uv run --frozen "$@"
