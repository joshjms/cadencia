#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
spec="data.json"
if [[ $# -gt 0 && "$1" != --* ]]; then
    spec="$1"
    shift
fi
uv run --extra placement --extra download scripts/build_chart.py "$spec" "$@"
