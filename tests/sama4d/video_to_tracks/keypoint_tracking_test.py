import pytest
import jax.numpy as jnp
from .registration import all_task_solver_pairs

@pytest.mark.parametrize("task,solver", all_task_solver_pairs)
def test(task, solver):
    threshold = 10. if task.scene_name == "rotating_cheezit_box" else jnp.inf
    # Run the tests 
    # Viz is set to true here so that this test case catches if the viz
    # code triggers errors. Note that because we have not
    # instantiated a rerun session, the visualizations will not be saved.
    task.run_tests(solver, distance_error_threshold=threshold, viz=True)
    
    # Free GPU memory if these loaded any.
    del task
    del solver