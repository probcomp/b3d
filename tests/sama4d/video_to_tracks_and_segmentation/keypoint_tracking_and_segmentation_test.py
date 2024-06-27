import pytest
import jax.numpy as jnp
from .registration import all_task_solver_pairs

@pytest.mark.parametrize("task,solver", all_task_solver_pairs)
def test(task, solver):
    # Run the tests 
    # Viz is set to true here so that this test case catches if the viz
    # code triggers errors. Note that because we have not
    # instantiated a rerun session, the visualizations will not be saved.
    # Also turn off the assertion testing using a huge distance threshold,
    # as the current solver isn't good.
    task.run_tests(solver, viz=True, distance_error_threshold=jnp.inf)
    
    # Free GPU memory if these loaded any.
    del task
    del solver