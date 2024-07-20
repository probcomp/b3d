import pytest
from .registration import all_task_solver_pairs


@pytest.mark.parametrize("task,solver", [all_task_solver_pairs[0]])
def test(task, solver):
    task.run_tests(solver)
