"""
This file registers a default set of keypoint to segmentation tasks and solvers.
"""

from ..data_curation import get_loaders_for_all_curated_scenes
from .keypoints_to_segmentation_task import KeypointsToSegmentationTask
from .dummy_solver import DummyTracksToSegmentationSolver

all_tasks = [
    KeypointsToSegmentationTask(
        spec["feature_track_data_loader"], scene_name=spec["scene_name"], n_frames=3
    )
    for spec in get_loaders_for_all_curated_scenes()
]
all_solvers = [DummyTracksToSegmentationSolver()]

all_task_solver_pairs = [(task, solver) for task in all_tasks for solver in all_solvers]
