#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$FAIR_SIGNAL_ROOT"
# Only GUI commands need XQuartz; headless training/evaluation stays untouched.
FAIR_SIGNAL_GUI=0
FAIR_SIGNAL_HEADLESS=0
for FAIR_SIGNAL_ARG in "$@"; do
    case "$FAIR_SIGNAL_ARG" in
        --gui|view_arterial.py|*/view_arterial.py) FAIR_SIGNAL_GUI=1 ;;
        --headless) FAIR_SIGNAL_HEADLESS=1 ;;
    esac
done
if [[ "$(uname -s)" == "Darwin" && "$FAIR_SIGNAL_GUI" == 1 && "$FAIR_SIGNAL_HEADLESS" == 0 ]]; then
    DISPLAY="$(uv run --frozen python "$FAIR_SIGNAL_ROOT/scripts/gui_display.py")"
    export DISPLAY
fi
exec uv run --frozen "$@"
