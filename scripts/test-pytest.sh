#!/usr/bin/env bash

# Test coverage for pytest.sh

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMP_DIR=""

suite() {
  suite_addTest test-resolve-test-targets
}

oneTimeSetUp() {
  export B3D_TEST_MODE=1
  source "$DIR/pytest.sh"
}

oneTimeTearDown() {
  export B3D_TEST_MODE=0
}

setUp() {
  echo "setUp"
}

tearDown() {
  rm -rf "$TEMP_DIR"
}

test-resolve-test-targets() {
  local status
  local test_files

  TEMP_DIR=$(mktemp -d)
  echo "test_file_1" >"$TEMP_DIR/test_file1.py"
  echo "test_file_2" >"$TEMP_DIR/test_file2.py"

  VALID_TEST_TARGETS=(
    "$TEMP_DIR/test_file1.py"
    "$TEMP_DIR/test_file2.py"
    "$TEMP_DIR/test_file1.py::test_function1"
    "$TEMP_DIR/test_file2.py::test_function2"
  )

  for t1 in "${VALID_TEST_TARGETS[@]}"; do
    for t2 in "${VALID_TEST_TARGETS[@]}"; do
      export TEST_TARGETS="$t1 $t2"
      IFS=' ' read -r -a test_files <<<"$(resolve-test-targets)"
      resolve-test-targets >/dev/null
      status=$?
      $_ASSERT_TRUE_ $status
      $_ASSERT_EQUALS_ '"${test_files[0]}"' '"$t1"'
      $_ASSERT_EQUALS_ '"${test_files[1]}"' '"$t2"'
    done
  done

  INVALID_TEST_TARGETS=(
    "$TEMP_DIR/non_existent_file.py"
    "::test_function_without_file"
    "invalid_format"
    "/non/existent/path.py"
  )

  test_files=()
  for t1 in "${INVALID_TEST_TARGETS[@]}"; do
    for t2 in "${INVALID_TEST_TARGETS[@]}"; do
      export TEST_TARGETS="$t1 $t2"
      IFS=' ' read -r -a test_files <<<"$(resolve-test-targets)"
      resolve-test-targets >/dev/null
      status=$?
      $_ASSERT_FALSE_ $status
    done
  done
}

. ./shunit2.sh
