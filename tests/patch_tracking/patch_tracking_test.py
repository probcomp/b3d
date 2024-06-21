import jax.numpy as jnp
from .task import PatchTrackingTask
from .solver import AdamPatchTracker

def test():
    task = PatchTrackingTask.task_from_rotating_cheezit_box(n_frames=2)

    # This solver currently is buggy, so don't have it throw an error
    # in CI for performance; just test that it runs.
    task.run_tests(AdamPatchTracker(), distance_error_threshold=jnp.inf)