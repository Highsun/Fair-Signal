#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$FAIR_SIGNAL_ROOT"
# Only GUI commands need XQuartz; headless training/evaluation stays untouched.
FAIR_SIGNAL_GUI=0
FAIR_SIGNAL_HEADLESS=0
for FAIR_SIGNAL_ARG in "$@"; do
    case "$FAIR_SIGNAL_ARG" in
        --gui) FAIR_SIGNAL_GUI=1 ;;
        --headless) FAIR_SIGNAL_HEADLESS=1 ;;
    esac
done
if [[ "${1:-}" == python || "${1:-}" == python3 ]]; then
    case "${2:-}" in
        view_arterial.py|*/view_arterial.py|view_residual.py|*/view_residual.py) FAIR_SIGNAL_GUI=1 ;;
    esac
fi
if [[ "$(uname -s)" == "Darwin" && "$FAIR_SIGNAL_GUI" == 1 && "$FAIR_SIGNAL_HEADLESS" == 0 ]]; then
    DISPLAY="$(uv run --frozen python "$FAIR_SIGNAL_ROOT/scripts/gui_display.py")"
    export DISPLAY
fi
exec uv run --frozen "$@"
