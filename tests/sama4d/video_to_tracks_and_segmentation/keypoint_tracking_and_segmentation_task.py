"""
Note: As of June 24, 2024, this file has not yet been tested.
"""
from tests.sama4d.video_to_tracks.keypoint_tracking_task import KeypointTrackingTask
from tests.sama4d.tracks_to_segmentation.keypoints_to_segmentation_task import SegmentationFromKeypointTracksTask

# Extend the KeypointTrackingTask class to expect an object segmentation,
# and score the segmentation.
class KeypointTrackingAndSegmentationTask(KeypointTrackingTask):
    """
    The task specification consists of:
        - video [RGB or RGBD video]
        - Xs_WC [camera pose in the world frame, per frame]
        - initial_keypoint_positions_2D [2D keypoint center positions at frame 0]
            (N, 2) array of 2D keypoint center positions at frame 0
            stored as (y, x) pixel coordinates
        - renderer [Renderer object containing camera intrincis]

    The "ground truth" data consists of
        - keypoint_positions_2D [2D keypoint center positions at each frame]
            (T, N, 2) array

    A "solution" to the task looks is a dict with two fields:
        - inferred_keypoint_positions_2D [3D keypoint center positions at each frame]
            (T, N, 2) array
        - object_assignments [Object assignments for each keypoint]
            (T, N) array
    """
    
    def score(self, solution):
        # Score the tracking and object association separately, using the logic
        # in the `KeypointTrackingTask` and `SegmentationFromKeypointTracksTask` classes
        return {
            "point_tracking_2D": super().score(solution["inferred_keypoint_positions_2D"]),
            "object_association": SegmentationFromKeypointTracksTask(self.ftd).score(solution["object_assignments"])
        }
    
    def visualize_task(self):
        super().visualize_task(viz_keypoints=False)

        # TODO: visualize the keypoints, with colors for object assignments