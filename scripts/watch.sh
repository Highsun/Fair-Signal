#!/usr/bin/env bash
# One-hour traffic demonstration; waits for Play and advances slowly by default.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
if [[ "$(uname -s)" == "Darwin" && -z "${DISPLAY:-}" ]]; then
    FAIR_SIGNAL_DISPLAY="$(launchctl getenv DISPLAY)"
    if [[ -n "$FAIR_SIGNAL_DISPLAY" ]]; then
        export DISPLAY="$FAIR_SIGNAL_DISPLAY"
    elif [[ -S /private/tmp/.X11-unix/X0 ]]; then
        # Use the existing local XQuartz session for this process only.
        export DISPLAY=:0
    else
        echo "请先打开 XQuartz，再运行本命令；首次安装可能需要注销重新登录。" >&2
        exit 1
    fi
fi
FAIR_SIGNAL_DEMO_OUTPUT="outputs/gui-1h/$(date +%Y%m%d-%H%M%S)"
exec bash "$FAIR_SIGNAL_ROOT/scripts/run.sh" fair-signal replay \
    --run outputs/pilot --method dqn_fair --seed 11 \
    --scenario changing_peak --traffic-seed 2001 --demand-seconds 3600 \
    --gui --gui-delay 200 --output "$FAIR_SIGNAL_DEMO_OUTPUT" "$@"
