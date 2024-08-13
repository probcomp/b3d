"""
This file registers a default set of tasks and solvers for the video to keypoint tracks task class.
"""

import b3d
import jax.numpy as jnp

from ..data_curation import get_loaders_for_all_curated_scenes
from .keypoint_tracking_task import KeypointTrackingTask
from .solvers.particle_system_patch_tracking_solver import (
    AdamPatchTracker_UsingSingleframeParticleSystemTraces,
)

all_tasks = [
    KeypointTrackingTask(
        spec["feature_track_data_loader"], scene_name=spec["scene_name"], n_frames=3
    )
    for spec in get_loaders_for_all_curated_scenes()
]
all_solvers = [AdamPatchTracker_UsingSingleframeParticleSystemTraces()]
# Once we fix the memory leak, could be good to test against both of these...
# [AdamPatchTracker_UsingDenseOnlyTraces(), AdamPatchTracker_UsingSingleframeParticleSystemTraces()]

all_task_solver_pairs = [(task, solver) for task in all_tasks for solver in all_solvers]

####
# I will also define a few tasks and solvers here which I won't register
# with the testing system yet.
####


def get_curated_single_patch_tracking_tasks():
    """
    Returns a list of `KeypointTrackingTask` objects which have been curated so that
    the keypoints should be trackable via patch tracking.
    """
    good_scene_specs = [
        {
            "file": "pan_through_plantroom.npz",
            "time0": 10,
            "n_frames": 20,
            "good_keypoint_indices": [68, 84, 89, 213, 515, 682],
        },
        {
            "file": "pan_around_blocks.npz",
            "time0": 30,
            "n_frames": 20,
            "good_keypoint_indices": [6, 7, 16, 58, 82, 91, 101, 188, 652, 675, 695],
        },
        {
            "file": "pan_around_frog.npz",
            "time0": 30,
            "n_frames": 20,
            "good_keypoint_indices": [20, 50, 150, 200, 500, 520],
        },
    ]

    task_specs = [
        {
            "file": s["file"],
            "time0": s["time0"],
            "n_frames": s["n_frames"],
            "keypoint_index": i,
        }
        for s in good_scene_specs
        for i in s["good_keypoint_indices"]
    ]

    def load_ftd_from_task_spec(spec):
        # TODO: change this URL after uploading these data files to the b3d bucket
        ftd_full = b3d.io.FeatureTrackData.load(
            b3d.get_assets_path() / "mydata" / spec["file"]
        )
        return (
            ftd_full.slice_time(spec["time0"])
            .remove_points_invisible_at_frame0()
            .slice_keypoints(jnp.array([spec["keypoint_index"]]))
        )

    return [
        KeypointTrackingTask(
            (lambda spec: (lambda: load_ftd_from_task_spec(spec)))(spec),
            scene_name=spec["file"],
            n_frames=spec["n_frames"],
        )
        for spec in task_specs
    ]
