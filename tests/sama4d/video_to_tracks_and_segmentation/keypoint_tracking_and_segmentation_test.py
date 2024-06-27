import pytest
import jax.numpy as jnp
from .registration import all_task_solver_pairs

@pytest.mark.parametrize("task,solver", all_task_solver_pairs)
def test(task, solver):
    task.run_tests(solver, distance_error_threshold=jnp.inf)