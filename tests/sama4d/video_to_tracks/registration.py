"""
This file registers a default set of tasks and solvers for the video to keypoint tracks task class.
"""

from ..data_curation import get_loaders_for_all_curated_scenes
from .keypoint_tracking_task import KeypointTrackingTask
from .patch_tracking_solver import AdamPatchTracker

all_tasks = [
    KeypointTrackingTask(spec["feature_track_data_loader"], scene_name=spec["scene_name"], n_frames=3)
    for spec in get_loaders_for_all_curated_scenes()
]
all_solvers = [AdamPatchTracker()]

all_task_solver_pairs = [(task, solver) for task in all_tasks for solver in all_solvers]