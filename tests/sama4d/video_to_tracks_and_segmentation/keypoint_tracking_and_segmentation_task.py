from tests.sama4d.tracks_to_segmentation.keypoints_to_segmentation_task import (
    KeypointsToSegmentationTask,
)
from tests.sama4d.video_to_tracks.keypoint_tracking_task import KeypointTrackingTask
<<<<<<< HEAD
from tests.sama4d.tracks_to_segmentation.keypoints_to_segmentation_task import (
    KeypointsToSegmentationTask,
)
=======
>>>>>>> main


# Extend the KeypointTrackingTask class to expect an object segmentation,
# and score the segmentation.
class KeypointTrackingAndSegmentationTask(KeypointTrackingTask):
    """
    The task specification consists of:
        - video [RGB or RGBD video]
        - poses_WC [camera pose in the world frame, per frame]
        - initial_keypoint_positions_2D [2D keypoint center positions at frame 0]
            (N, 2) array of 2D keypoint center positions at frame 0
            stored as (x, y) pixel coordinates
        - renderer [Renderer object containing camera intrincis]

    The "ground truth" data consists of
        - keypoint_positions_2D [2D keypoint center positions at each frame]
            (T, N, 2) array
        object_assignments [Object assignments for each keypoint]
            (N,) array of integer object indices

    A "solution" to the task looks is a dict with two fields:
        - inferred_keypoint_positions_2D [3D keypoint center positions at each frame]
            (T, N, 2) array
        - object_assignments [Object assignments for each keypoint]
            (N,) array of integer object indices
    """

    # Init and get_task_specification are inherited from KeypointTrackingTask

    def score(self, solution, **kwargs):
        # Score the tracking and object association separately, using the logic
        # in the `KeypointTrackingTask` and `SegmentationFromKeypointTracksTask` classes
        return {
            "point_tracking_2D": super().score(
                solution["inferred_keypoint_positions_2D"], **kwargs
            ),
            "object_association": KeypointsToSegmentationTask(lambda: self.ftd).score(
                solution["object_assignments"]
            ),
        }

    def assert_passing(self, metrics, **kwargs):
        super().assert_passing(metrics["point_tracking_2D"])
        KeypointsToSegmentationTask(lambda: self.ftd).assert_passing(
            metrics["object_association"]
        )

    def visualize_task(self):
        # Use the parent viz to visualize the video, but not the keypoints.
        super().visualize_task(viz_keypoints=False)

        # TODO: now, visualize the keypoints, using colors to denote object assignments

    def visualize_solution(self, solution, metrics):
        pass
        # TODO: implement this
