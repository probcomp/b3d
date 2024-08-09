#!/usr/bin/env bash

set -eo pipefail

TEST_TARGETS="${TEST_TARGETS:-tests}"

resolve_test_targets() {
  local pattern
  local file
  local targets=()

  for pattern in $TEST_TARGETS; do
    for file in $pattern; do
      if [ -e "$file" ]; then
        targets+=("$file")
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
    echo "TEST_TARGETS does not contain valid test file paths: $TEST_TARGETS"
    exit 1
  fi

  echo "Running test paths: ${test_files[*]}"
  cd "$PIXI_PROJECT_ROOT"
  pytest "${test_files[@]}"
}

main
