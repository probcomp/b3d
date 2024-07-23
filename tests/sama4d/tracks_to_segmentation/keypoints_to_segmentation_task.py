from typing import Callable

import b3d
import jax.numpy as jnp

from tests.common.task import Task


class KeypointsToSegmentationTask(Task):
    """
    The task specification consists of:
        - keypoint_tracks_2D [2D keypoint tracks]
            (T, N, 2) array of 2D keypoint center positions at each frame
            stored as (y, x) pixel coordinates
        - keypoint_visibility [keypoint visibility]
            (T, N) array of keypoint visibility at each frame
        - poses_WC [`poses_WC[t]` is the camera pose in the world's coordinate frame
            at timestep `t`. `poses_WC` is a batched `b3d.Pose` object.]
        - renderer [Renderer object containing camera intrinsics]

    The "ground truth" data consists of
        - segmentation [object segmentation]
            (N,) array object segmentation, labeling each keypoint with a unique object index

    A "solution" to the task looks like
        - inferred_segmentation [inferred object segmentation]
            (N,) array object segmentation, labeling each keypoint with a unique object index
    """

    def __init__(
        self,
        feature_track_data_loader: Callable[[], b3d.io.FeatureTrackData],
        scene_name=None,
        n_frames=None,
    ):
        self.feature_track_data_loader = feature_track_data_loader
        self.n_frames = n_frames
        self.instantiated = False

        if scene_name is not None:
            self.scene_name = scene_name
            self._name = "KeypointsToSegmentationTask[" + scene_name + "]"
        else:
            self._name = "KeypointsToSegmentationTask[no scene name provided]"

    def instantiate(self):
        if self.instantiated:
            return
        self.ftd = self.feature_track_data_loader()
        if self.n_frames is not None:
            self.ftd = self.ftd.slice_time(end_frame=self.n_frames)

        self.poses_WC = self.ftd.camera_poses
        self.renderer = b3d.Renderer.from_intrinsics_object(
            b3d.camera.Intrinsics.from_array(self.ftd.camera_intrinsics)
        )
        self.instantiated = True

    def get_task_specification(self):
        self.instantiate()
        return {
            "keypoint_tracks_2D": self.ftd.observed_keypoints_positions,
            "keypoint_visibility": self.ftd.keypoint_visibility,
            "poses_WC": self.poses_WC,
            "renderer": self.renderer,
        }

    def score(self, solution_object_assignments):
        self.instantiate()

        n_nonempty_true_objects = jnp.unique(self.ftd.object_assignments).shape[0]
        n_nonempty_inferred_objects = jnp.unique(solution_object_assignments).shape[0]

        # For each pair of keypoints, determine if they are assigned to the same object
        # in the ground truth, and if they are assigned to the same object in the solver solution.
        # Score how close these pairwise co-assignment matrices for GT and Solution are to each other.
        gt_coassociation_matrix = jnp.equal(
            self.ftd.object_assignments[:, None], self.ftd.object_assignments[None, :]
        )
        inferred_coassociation_matrix = jnp.equal(
            solution_object_assignments[:, None], solution_object_assignments[None, :]
        )
        fraction_pairwise_assignments_correct = jnp.mean(
            jnp.logical_or(
                jnp.logical_and(gt_coassociation_matrix, inferred_coassociation_matrix),
                jnp.logical_and(
                    ~gt_coassociation_matrix, ~inferred_coassociation_matrix
                ),
            ),
            axis=(0, 1),
        )
        return {
            "object_count": {
                "true_n_objects": n_nonempty_true_objects,
                "inferred_n_objects": n_nonempty_inferred_objects,
                "error": jnp.abs(n_nonempty_true_objects - n_nonempty_inferred_objects),
            },
            "object_assignment": {
                "fraction_pairwise_assignments_correct": fraction_pairwise_assignments_correct
            },
        }

    def assert_passing(self, scores):
        pass
        # TODO: fill this in

    def visualize_task(self):
        self.instantiate()
        # TODO: Fill this in

    def visualize_solution(self, solution, metrics):
        pass
        # TODO: Fill this in
