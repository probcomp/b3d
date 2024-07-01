import jax.numpy as jnp
from tests.common.solver import Solver

class DummyTracksToSegmentationSolver(Solver):
    def solve(self, task_spec):
        # assign every keypoint to the same object
        # (called object 0)
        return jnp.zeros(
            task_spec["keypoint_tracks_2D"].shape[1],
            dtype=int
        )

    def visualize_solver_state(self, task_spec):
        pass
