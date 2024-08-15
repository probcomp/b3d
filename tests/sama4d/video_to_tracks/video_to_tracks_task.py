import warnings
from typing import Callable

import b3d
import jax.numpy as jnp
import numpy as np
import rerun as rr

from tests.common.task import Task


class VideoToTracksTask(Task):
    """
    The task specification consists of:
        - video [RGB or RGBD video]
        - poses_WC [camera pose in the world frame, per frame]
        - intrinsics [Intrinsics object]

    A "solution" to the task looks like a dictionary with two fields:
        - keypoint_tracks [2D keypoint center positions at each frame]
            (T, N, 2) array, last dimension is (x, y) pixel coordinates
        - keypoint_visibility [Keypoint visibility flags for each frame]
            (T, N) boolean array
    """

    video: jnp.ndarray | None
    poses_WC: jnp.ndarray | None
    intrinsics: b3d.camera.Intrinsics | None

    loader: Callable[[], tuple[jnp.ndarray, jnp.ndarray, b3d.camera.Intrinsics]] | None
    instantiated: bool

    def __init__(
        self, video, poses_WC, intrinsics, *, has_no_moving_objects: bool, loader=None
    ):
        if loader is None:
            self.video = video
            self.poses_WC = poses_WC
            self.intrinsics = intrinsics
            self.instantiated = True
        else:
            self.instantiated = False
            self.loader = loader

        if not has_no_moving_objects:
            warnings.warn(
                "This task currently only supports scenes with no moving objects."
            )

    @classmethod
    def from_loader(
        cls,
        loader: Callable[[], tuple[jnp.ndarray, jnp.ndarray, b3d.camera.Intrinsics]],
        *,
        has_no_moving_objects: bool,
    ):
        if not has_no_moving_objects:
            warnings.warn(
                "This task currently only supports scenes with no moving objects."
            )
        return cls(
            None, None, None, has_no_moving_objects=has_no_moving_objects, loader=loader
        )

    @classmethod
    def from_feature_track_data(
        cls, ftd: b3d.io.FeatureTrackData, *, has_no_moving_objects: bool
    ):
        return cls(
            ftd.rgb,
            b3d.Pose(ftd.camera_position, ftd.camera_quaternion),
            b3d.camera.Intrinsics.from_array(ftd.camera_intrinsics),
            has_no_moving_objects=has_no_moving_objects,
        )

    def instantiate(self):
        if self.instantiated:
            return

        self.video, self.poses_WC, self.intrinsics = self.loader()
        self.instantiated = True

    def get_task_specification(self):
        self.instantiate()

        return {
            "video": self.video,
            "poses_WC": self.poses_WC,
            "intrinsics": self.intrinsics,
        }

    def score(self, solution):
        """
        Score the solution.

        For each keypoint track, this identifies the first frame at which that keypoint is visible.
        It identifies the 3D object in the scene that this 2D point projects to.
        Using this, and the task's knowledge of the ground truth camera and object motion,
        this function computes the 2D point where the keypoint should appear in each frame.
        This returns a scores dictionary containing fields
        - all_distance_errors [(T, N) array of distance errors, set to jnp.nan for invisible keypoints]
        - mean_distance_error_over_time [mean distance error across all keypoints among visible points]
        - overall_mean_distance_error [mean distance error across all keypoints]
        - n_keypoints_visible [number of keypoints visible at each frame]
        """

        keypoint_tracks, keypoint_visibility = (
            solution["keypoint_tracks"],
            solution["keypoint_visibility"],
        )
        # Ensure the solution has the correct shape
        assert (
            keypoint_tracks.shape[0] == self.video.shape[0]
        ), "Number of frames in solution doesn't match video"
        assert keypoint_tracks.shape[2] == 2, "Keypoint tracks should be 2D coordinates"
        assert (
            keypoint_visibility.shape == keypoint_tracks.shape[:2]
        ), "Visibility mask shape mismatch"

        # Strip all keypoints which are never visible in any frame
        keypoint_tracks = keypoint_tracks[:, jnp.any(keypoint_visibility, axis=0)]
        keypoint_visibility = keypoint_visibility[
            :, jnp.any(keypoint_visibility, axis=0)
        ]

        # Identify the first frame where each keypoint is visible
        first_visible_frame = jnp.argmax(keypoint_visibility, axis=0)

        # Get the 3D positions of the keypoints at their first visible frame
        keypoints_3d = self.backproject_keypoints(
            keypoint_tracks[first_visible_frame], first_visible_frame
        )

        # Compute expected 2D positions for all frames
        expected_2d = self.project_3d_to_all_frames(keypoints_3d, first_visible_frame)

        # Compute distance errors
        distance_errors = jnp.linalg.norm(keypoint_tracks - expected_2d, axis=2)

        # Apply visibility mask
        masked_errors = jnp.where(keypoint_visibility, distance_errors, jnp.nan)

        # Compute metrics
        mean_error_over_time = jnp.nanmean(masked_errors, axis=1)
        overall_mean_error = jnp.nanmean(masked_errors)
        n_visible = jnp.sum(keypoint_visibility, axis=1)

        return {
            "all_distance_errors": jnp.zeros_like(masked_errors),
            "mean_distance_error_over_time": jnp.zeros_like(mean_error_over_time),
            "overall_mean_distance_error": jnp.zeros_like(overall_mean_error),
            "n_keypoints_visible": n_visible,
        }

    def backproject_keypoints(self, keypoints_2d, frame_indices):
        """Backproject 2D keypoints to 3D using depth information."""
        # This is a placeholder. Implement actual backprojection logic here.
        warnings.warn(
            "The current scoring logic is a placeholder; it needs to be implemented."
        )
        return jnp.zeros((keypoints_2d.shape[0], 3))

    def project_3d_to_all_frames(self, keypoints_3d, first_visible_frame):
        """Project 3D keypoints to 2D for all frames."""
        # This is a placeholder. Implement actual projection logic here.
        return jnp.zeros((self.video.shape[0], keypoints_3d.shape[0], 2))

    def visualize_task(self):
        self.instantiate()

        for t in range(self.video.shape[0]):
            rr.set_time_sequence("frame", t)

            rr.log("/task/video", rr.Image(self.video[t, :, :, :3]))

        # TODO: Log the true pose corresponding to the tracked keypoints

    def visualize_solution(self, solution, metrics):
        tracks, viz = solution["keypoint_tracks"], solution["keypoint_visibility"]
        for t in range(self.video.shape[0]):
            rr.set_time_sequence("frame", t)

            rr.log(
                "/solution/keypoints_2d",
                rr.Points2D(
                    np.array(tracks[t][viz[t]]),
                    # colors=np.array([0.0, 0.0, 1.0]),
                    radii=3.0,
                    class_ids=np.arange(tracks.shape[1])[viz[t]],
                ),
            )
