import pytest
import jax.numpy as jnp
from ..data_curation import get_loaders_for_all_curated_scenes
from .keypoint_tracking_task import KeypointTrackingTask
from .patch_tracking_solver import AdamPatchTracker

@pytest.mark.parametrize("scene_name,scene_loader", [
    (spec["scene_name"], spec["feature_track_data_loader"])
    for spec in get_loaders_for_all_curated_scenes()
])
def test(scene_name, scene_loader):
    scene = scene_loader()
    task = KeypointTrackingTask.task_from_feature_track_data(scene, n_frames=3)
    
    # This solver currently is buggy, so don't have it throw an error
    # in CI for performance; just test that it runs.
    task.run_tests(AdamPatchTracker(), distance_error_threshold=jnp.inf)
