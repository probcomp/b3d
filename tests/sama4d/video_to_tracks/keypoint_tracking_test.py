import jax.numpy as jnp
import pytest

from .registration import all_task_solver_pairs


@pytest.mark.parametrize("task,solver", [all_task_solver_pairs[0]])
def test(task, solver):
    threshold = 10.0 if task.scene_name == "rotating_cheezit_box" else jnp.inf
    task.run_tests(solver, distance_error_threshold=threshold)
