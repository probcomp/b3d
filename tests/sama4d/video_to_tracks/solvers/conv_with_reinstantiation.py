import jax
from b3d.chisight.patch_tracking_2d.patch_tracker import PatchTracker2D

from tests.common.solver import Solver

### Solver for VideoToTracksTask ###


class PatchTracker2DSolver(Solver):
    tracker: PatchTracker2D

    def __init__(self, **kwargs):
        self.tracker = PatchTracker2D(**kwargs)

    def solve(self, task_specification):
        video = task_specification["video"]
        (keypoint_tracks, keypoint_visibility) = (
            self.tracker.run_and_get_tracks_separated_by_dimension(
                jax.random.PRNGKey(1235134), video
            )
        )

        return {
            "keypoint_tracks": keypoint_tracks,
            "keypoint_visibility": keypoint_visibility,
        }
