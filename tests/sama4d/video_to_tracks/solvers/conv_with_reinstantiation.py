import jax
from b3d.chisight.patch_tracking_2d.patch_tracker import (
    PatchTrackerParams,
    TrackerState,
)

from tests.common.solver import Solver

### Solver for VideoToTracksTask ###


class KeypointTracker2DWithReinitialization(Solver):
    params: PatchTrackerParams

    def __init__(self, **kwargs):
        self.params = PatchTrackerParams(**kwargs)

    def solve(self, task_specification):
        video = task_specification["video"]

        pre_frame0_state = TrackerState.pre_init_state(self.params)

        def step(state, key_and_frame):
            (key, frame) = key_and_frame
            new_state = state.update(key, frame, self.params)
            return (new_state, new_state.get_tracks_and_visibility())

        keys = jax.random.split(jax.random.PRNGKey(816527), video.shape[0])
        _, (keypoint_tracks, keypoint_visibility) = jax.lax.scan(
            step, pre_frame0_state, (keys, video)
        )

        return {
            "keypoint_tracks": keypoint_tracks,
            "keypoint_visibility": keypoint_visibility,
        }
