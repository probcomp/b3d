from tests.common.solver import Solver
import jax.numpy as jnp

class KeypointTrackingAndSegmentationDummySolver(Solver):
    def solve(self, task_spec):
        # Dummy solution
        return {
            # All keypoints remain a their initial poses
            "inferred_keypoint_positions_2D": jnp.tile(
                task_spec["initial_keypoint_positions_2D"],
                (task_spec["video"].shape[0], 1, 1)
            ),
            # All objects assigned to the same point
            "object_assignments": jnp.zeros(
                task_spec["initial_keypoint_positions_2D"].shape[0], dtype=int
            )
            
        }
    
    def visualize_solver_state(self, task_spec):
        pass