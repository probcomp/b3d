"""
This file is WIP.
"""

from tests.common.task import Task
import jax
import jax.numpy as jnp

class KeypointTrackingAndSegmentationTask(Task):
    """

    """
    
    def score(self, solution):
        return {
            "object_count": {
                "true_n_objects": None,
                "inferred_n_objects": None,
                "error": None,
            },
            "point_tracking_3D": {
                "mean_distance_error": None,
                "n_errors_above_threshold_per_frame": None,
            },
            "object_assignment": {
                # For each frame, for every pair of two latent points,
                # consider the pair correct w.r.t. object assignment if
                # either the pair is the same object and it is labeled
                # as such, or they are not the same object, and they are labeled
                # as such
                "fraction_of_pairwise_assignments_correct_per_frame": None
            }
        }