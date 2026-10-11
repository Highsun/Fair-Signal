#!/usr/bin/env bash
# Source this file; configuration applies only to the current shell/process.
FAIR_SIGNAL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_CACHE_DIR="$FAIR_SIGNAL_ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$FAIR_SIGNAL_ROOT/.runtime/python"
export UV_PYTHON_INSTALL_BIN=0
export MPLCONFIGDIR="$FAIR_SIGNAL_ROOT/.cache/matplotlib"
export PYTHONPYCACHEPREFIX="$FAIR_SIGNAL_ROOT/.cache/pycache"
export SUMO_HOME="${SUMO_HOME:-/Library/Frameworks/EclipseSUMO.framework/EclipseSUMO/share/sumo}"
if [[ -d "$SUMO_HOME/../../bin" ]]; then
    export PATH="$(cd "$SUMO_HOME/../../bin" && pwd):$PATH"
elif [[ -d "$SUMO_HOME/bin" ]]; then
    export PATH="$SUMO_HOME/bin:$PATH"
fi
