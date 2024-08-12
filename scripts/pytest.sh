#!/usr/bin/env bash

if [[ $B3D_TEST_MODE -ne 1 ]]; then
  set -eo pipefail
fi

TEST_TARGETS="${TEST_TARGETS:-tests}"

resolve-test-targets() {
  local pattern
  local file
  local base_file
  local function_name
  local targets=()

  for pattern in $TEST_TARGETS; do
    if [[ $pattern == *::* ]]; then
      base_file="${pattern%%::*}"
      function_name="${pattern##*::}"
    else
      base_file="$pattern"
      function_name=""
    fi

    for file in $base_file; do
      if [ -e "$file" ]; then
        if [ -n "$function_name" ]; then
          targets+=("$file::$function_name")
        else
          targets+=("$file")
        fi
      fi
    done
  done

  if [ ${#targets[@]} -eq 0 ]; then
    echo "${targets[@]}"
    return 1
  else
    echo "${targets[@]}"
    return 0
  fi
}

main() {
  local targets
  local test_files

  IFS=' ' read -r -a test_files <<<"$(resolve-test-targets)"

  if [ ${#test_files[@]} -eq 0 ]; then
    echo "TEST_TARGETS does not contain valid tests: $TEST_TARGETS"
    exit 1
  fi

  echo "Running test paths: ${test_files[*]}"
  cd "$PIXI_PROJECT_ROOT"
  if ! pytest "${test_files[@]}"; then
    echo "TEST_TARGETS contains an invalid test: $TEST_TARGETS"
    return 1
  else
    return 0
  fi
}

if [[ $B3D_TEST_MODE -eq 1 ]]; then
  echo "entering test mode..."
else
  main "$@"
fi
