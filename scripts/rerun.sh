#!/usr/bin/env bash

# This script launches rerun.

set -eo pipefail

main() {
  cd "$PIXI_PROJECT_ROOT"
  if pgrep -l rerun >/dev/null; then
    echo "Rerun is already running"
    exit 0
  else
    python -m rerun --port 8812 >/dev/null &
    sleep 2
    echo "Rerun was launched"
    exit 0
  fi
}

main
