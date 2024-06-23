from tests.sama4d.feature_track_data_task import FeatureTrackDataTask
import jax.numpy as jnp

class KeypointTrackingTask(FeatureTrackDataTask):
    """
    The task specification consists of:
        - video [RGB or RGBD video]
        - Xs_WC [camera pose in the world frame, per frame]
        - initial_keypoint_positions_2D [2D keypoint center positions at frame 0]
            (N, 2) array of 2D keypoint center positions at frame 0
            stored as (y, x) pixel coordinates
        - renderer [Renderer object containing camera intrincis]

    The "ground truth" data consists of
        - keypoint_positions_3D [3D keypoint center positions at each frame]
            (T, N, 3) array

    A "solution" to the task looks like
        - inferred_keypoint_positions_3D [3D keypoint center positions at each frame]
            (T, N, 3) array

    Indexing in the `N` dimension in any of these arrays will index to the same keypoint.

    The task is scored by comparing the inferred_keypoint_positions_3D to the keypoint_positions_3D.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def score(self,
        inferred_keypoint_positions_3D,
        distance_error_threshold=0.1
    ):
        return {
            "mean_distance_error": jnp.mean(
                jnp.linalg.norm(inferred_keypoint_positions_3D - self.keypoint_positions_3D, axis=-1)
            ),
            "n_errors_above_threshold_per_frame": jnp.sum(
                jnp.linalg.norm(inferred_keypoint_positions_3D - self.keypoint_positions_3D, axis=-1) > distance_error_threshold,
                axis=-1
            )
        }

    def assert_passing(self, metrics, **kwargs):
        n_tracks = self.keypoint_positions_3D.shape[1]
        assert jnp.all(metrics["n_errors_above_threshold_per_frame"] < n_tracks * 0.1)