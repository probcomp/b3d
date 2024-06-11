#!/bin/bash

# TODO: configure this properly!

# Run Python unittest discovery.
# Should be run from the root of the project.
echo "Discovering and running all task tests starting with 'tests/test_'"
python -m unittest discover -s test/task_tests/tests -p "*.py"
