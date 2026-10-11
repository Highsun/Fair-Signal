#!/usr/bin/env bash
# One-hour traffic demonstration; waits for Play and advances slowly by default.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
FAIR_SIGNAL_DEMO_OUTPUT="outputs/gui-1h/$(date +%Y%m%d-%H%M%S)"
exec bash "$FAIR_SIGNAL_ROOT/scripts/run.sh" fair-signal replay \
    --run outputs/pilot --method dqn_fair --seed 11 \
    --scenario changing_peak --traffic-seed 2001 --demand-seconds 3600 \
    --gui --gui-delay 200 --output "$FAIR_SIGNAL_DEMO_OUTPUT" "$@"
