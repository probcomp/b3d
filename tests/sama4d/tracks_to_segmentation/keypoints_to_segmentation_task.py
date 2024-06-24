"""
Note: As of June 24, 2024, this file has not yet been tested.
"""

from tests.common.task import Task
import b3d
import jax.numpy as jnp

class SegmentationFromKeypointTracksTask(Task):
    """
    The task specification consists of:
        - keypoint_tracks_2D [2D keypoint tracks]
            (T, N, 2) array of 2D keypoint center positions at each frame
            stored as (y, x) pixel coordinates
        - keypoint_visibility [keypoint visibility]
            (T, N) array of keypoint visibility at each frame
        - Xs_WC [camera pose in the world frame, per frame]
        - renderer [Renderer object containing camera intrinsics]

    The "ground truth" data consists of
        - segmentation [object segmentation]
            (T, H, W) array of object segmentation at each frame

    A "solution" to the task looks like
        - inferred_segmentation [inferred object segmentation]
            (T, H, W) array of object segmentation at each frame
    """

    def __init__(self, feature_track_data : b3d.io.FeatureTrackData):
        self.ftd = feature_track_data
        self.Xs_WC = b3d.Pose(self.ftd.camera_position, self.ftd.camera_quaternion)
        self.renderer = b3d.Renderer.from_intrinsics_object(
            b3d.camera.Intrinsics.from_array(self.ftd.camera_intrinsics)
        )

    def get_task_specification(self):
        return {
            "keypoint_tracks_2D": self.ftd.observed_keypoint_positions,
            "keypoint_visibility": self.ftd.keypoint_visibility,
            "Xs_WC": self.Xs_WC,
            "renderer": self.renderer
        }

    def score(self, solution):
        n_nonempty_true_objects = jnp.unique(self.ftd.object_assignments).shape[0]
        n_nonempty_inferred_objects = jnp.unique(solution["object_assignments"]).shape[0]

        gt_coassociation_matrix = jnp.equal(
            self.ftd.object_assignments[:, None],
            self.ftd.object_assignments[None, :]
        )
        inferred_coassociation_matrix = jnp.equal(
            solution["object_assignments"][:, None],
            solution["object_assignments"][None, :]
        )
        fraction_pairwise_assignments_correct_per_frame = jnp.mean(
            jnp.logical_or(
                jnp.logical_and(gt_coassociation_matrix, inferred_coassociation_matrix),
                jnp.logical_and(~gt_coassociation_matrix, ~inferred_coassociation_matrix)
            ),
            axis=(1, 2)
        )
        return {
            "object_count": {
                "true_n_objects": n_nonempty_true_objects,
                "inferred_n_objects": n_nonempty_inferred_objects,
                "error": jnp.abs(n_nonempty_true_objects - n_nonempty_inferred_objects),
            },
            "object_assignment": {
                "fraction_pairwise_assignments_correct_per_frame": fraction_pairwise_assignments_correct_per_frame
            }
        }
    
    def visualize_task():
        # TODO
        pass