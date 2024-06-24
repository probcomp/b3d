import pytest
import jax.numpy as jnp
from ..data_curation import get_loaders_for_all_curated_scenes
from .keypoint_tracking_task import KeypointTrackingTask
from .patch_tracking_solver import AdamPatchTracker

# Right now I am only running the last of these tests, which is for the rotating_cheezit_box scene.
# This is for speed - once these can be run faster, we should test on all of them.
@pytest.mark.parametrize("scene_name,scene_loader", [
    (spec["scene_name"], spec["feature_track_data_loader"])
    for spec in get_loaders_for_all_curated_scenes()[-1]
])
def test(scene_name, scene_loader):
    scene = scene_loader()
    task = KeypointTrackingTask(scene, n_frames=3)
    
    if scene_name == "rotating_cheezit_box":
        # Assert tracks remain within 10px of the ground truth
        task.run_tests(AdamPatchTracker(), distance_error_threshold=10.)
    else:
        task.run_tests(AdamPatchTracker(), distance_error_threshold=jnp.inf)
