#!/usr/bin/env bash

set -euo pipefail

TEST_TARGETS="${PYTEST_TARGETS:-tests}"

resolve_test_targets() {
  local pattern
  local file
  local targets=()

  for pattern in $TEST_TARGETS; do
    for file in $pattern; do
      if [ -e "$file" ]; then
        targets+=("$file")
      else
        echo "Warning: Skipping test since it wasn't found '$pattern'"
      fi
    done
  done

  echo "${targets[@]}"
}

main() {
  local targets
  local test_files

  IFS=' ' read -r -a test_files <<<"$(resolve_test_targets)"

  if [ ${#test_files[@]} -eq 0 ]; then
    echo "No valid test files found."
    exit 1
  fi

  echo "Running test paths: ${test_files[*]}"
  cd "$PIXI_PROJECT_ROOT"
  pytest "${test_files[@]}"
}

main
