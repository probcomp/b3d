from typing import Callable

import jax.numpy as jnp
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

import b3d
from tests.common.task import Task


class KeypointTrackingTask(Task):
    """
    The task specification consists of:
        - video [RGB or RGBD video]
        - poses_WC [camera pose in the world frame, per frame]
        - initial_keypoint_positions_2D [2D keypoint center positions at frame 0]
            (N, 2) array of 2D keypoint center positions at frame 0
            stored as [x, y]) pixel coordinates
        - renderer [Renderer object containing camera intrincis]

    The "ground truth" data consists of
        - keypoint_positions_2D [2D keypoint center positions at each frame]
            (T, N, 2) array

    A "solution" to the task looks like
        - inferred_keypoint_positions_2D [3D keypoint center positions at each frame]
            (T, N, 2) array

    Indexing in the `N` dimension in any of these arrays will index to the same keypoint.

    The task is scored by comparing the inferred_keypoint_positions_3D to the keypoint_positions_3D.
    """

    def __init__(
        self,
        feature_track_data_loader: Callable[[], b3d.io.FeatureTrackData],
        n_frames: int = None,
        scene_name=None,
        # By default, ensure the feature_track_data has all keypoints visible at frame 0
        # and that the 2D keypoint positions are not too close to each other at frame 0.
        preprocessing_fn=(
            lambda ftd: ftd.remove_points_invisible_at_frame0().sparsify_points_to_minimum_2D_distance_at_frame0(
                # max of video Height // 80, 5
                max(ftd.rgbd_images.shape[1] // 80, 5)
            )
        ),
    ):
        self.feature_track_data_loader = feature_track_data_loader
        self.n_frames = n_frames
        self.preprocessing_fn = preprocessing_fn
        self.instantiated = False

        if scene_name is not None:
            self.scene_name = scene_name
            self._name = "KeypointTrackingTask[" + scene_name + "]"
        else:
            self._name = "KeypointTrackingTask[no scene name provided]"

    # Actually load in the feature track data and process it.
    # This lazy loading mechanism lets us construct and pass around the Task
    # object without waiting to load a big file from disk.
    def instantiate(self):
        if self.instantiated:
            return

        feature_track_data = self.feature_track_data_loader()
        if self.n_frames is not None:
            feature_track_data = feature_track_data.slice_time(end_frame=self.n_frames)
        self.ftd = self.preprocessing_fn(feature_track_data)
        self.poses_WC = b3d.Pose(self.ftd.camera_position, self.ftd.camera_quaternion)
        self.renderer = b3d.Renderer.from_intrinsics_object(
            b3d.camera.Intrinsics.from_array(self.ftd.camera_intrinsics)
        )
        self.instantiated = True

    @property
    def name(self):
        return self._name

    def get_task_specification(self):
        self.instantiate()

        return {
            "video": self.video,
            "poses_WC": self.poses_WC,
            "initial_keypoint_positions_2D": self.keypoint_positions_2D[0],
            "renderer": self.renderer,
        }

    @property
    def keypoint_positions_2D(self):
        return self.ftd.observed_keypoints_positions

    @property
    def video(self):
        return self.ftd.rgbd_images

    def score(
        self,
        inferred_keypoint_positions_2D,
        distance_error_threshold=3.0,  # pixels
    ):
        return {
            "mean_distance_error": safe_mean(
                jnp.linalg.norm(
                    inferred_keypoint_positions_2D - self.keypoint_positions_2D, axis=-1
                )
            ),
            "n_errors_above_threshold_per_frame": jnp.sum(
                jnp.linalg.norm(
                    inferred_keypoint_positions_2D - self.keypoint_positions_2D, axis=-1
                )
                > distance_error_threshold,
                axis=-1,
            ),
        }

    def assert_passing(self, metrics, **kwargs):
        n_tracks = self.keypoint_positions_2D.shape[1]
        assert jnp.all(metrics["n_errors_above_threshold_per_frame"] < n_tracks * 0.1)

    def visualize_task(self, *, viz_keypoints=True):
        self.instantiate()

        # Log initial frame of video, and the 2D keypoints
        rr.log(
            "/task/frame0", rr.Image(np.array(self.video[0, :, :, :3])), timeless=True
        )

        rr.log(
            "/task/initial_keypoint_positions_2D",
            rr.Points2D(
                np.array(self.keypoint_positions_2D[0]),
                colors=np.array([0.0, 1.0, 0.0]),
            ),
            timeless=True,
        )

        for i in range(self.video.shape[0]):
            rr.set_time_sequence("frame", i)

            # If 3D keypoints are available, visualize these
            if self.ftd.latent_keypoint_positions is not None and viz_keypoints:
                rr.log(
                    "/aux_info_unavailable_to_solver/keypoint_positions_3D",
                    rr.Points3D(
                        np.array(self.ftd.latent_keypoint_positions[i]),
                        colors=np.array([0.0, 1.0, 0.0]),
                        radii=0.003,
                    ),
                )

            # Visualize the camera, observed RGB image, and the 2D keypoints
            renderer = self.renderer
            rr.log(
                "/task/camera",
                rr.Pinhole(
                    focal_length=[float(renderer.fx), float(renderer.fy)],
                    width=renderer.width,
                    height=renderer.height,
                    principal_point=np.array([renderer.cx, renderer.cy]),
                ),
            )
            X_WC = self.poses_WC[i]
            rr.log(
                "/task/camera",
                rr.Transform3D(translation=X_WC.pos, mat3x3=X_WC.rot.as_matrix()),
            )
            rr.log(
                "/task/camera/rgb_observed", rr.Image(np.array(self.video[i, :, :, :3]))
            )
            if viz_keypoints:
                rr.log(
                    "/groundtruth_solution/keypoints_2d",
                    rr.Points2D(
                        np.array(self.keypoint_positions_2D[i]),
                        colors=np.array([0.0, 1.0, 0.0]),
                        radii=4.0,
                    ),
                )
                rr.log(
                    "/task/rgb_observed", rr.Image(np.array(self.video[i, :, :, :3]))
                )

            # If depth is available, visualize it
            if self.ftd.has_depth_channel():
                rr.log(
                    "/task/camera/depth_observed",
                    rr.DepthImage(np.array(self.video[i, :, :, 3])),
                )
                # If video is RGBD, get the point cloud and visualize it in the 3D viewer
                r = self.renderer
                xyzs_C = b3d.utils.xyz_from_depth_vectorized(
                    self.video[i, :, :, 3], r.fx, r.fy, r.cx, r.cy
                )
                rgbs = np.array(self.video[i, :, :, :3])
                xyzs_W = X_WC.apply(xyzs_C)
                rr.log(
                    "/task/observed_pointcloud",
                    rr.Points3D(
                        positions=np.array(xyzs_W).reshape(-1, 3),
                        colors=rgbs.reshape(-1, 3),
                        radii=0.001
                        * jnp.ones(np.array(xyzs_W).reshape(-1, 3).shape[0]),
                    ),
                )

    def visualize_solution(self, solution, metrics):
        for i in range(self.video.shape[0]):
            rr.set_time_sequence("frame", i)
            rr.log(
                "/solution/keypoints_2d",
                rr.Points2D(
                    np.array(solution[i]),
                    colors=np.array([0.0, 0.0, 1.0]),
                    radii=3.0,
                ),
            )

    @classmethod
    def rr_blueprint(cls):
        return rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(),
                rrb.Vertical(
                    rrb.Spatial2DView(
                        contents=[
                            "groundtruth_solution/keypoints_2d",
                            "solution/keypoints_2d",
                        ]
                    ),
                    rrb.Spatial2DView(
                        contents=[
                            "groundtruth_solution/keypoints_2d",
                            "solution/keypoints_2d",
                            "task/rgb_observed",
                        ]
                    ),
                ),
            )
        )


def safe_mean(x):
    """Returns the mean of the non-inf and non-nan values of x."""
    return jnp.mean(x[jnp.isfinite(x)])
