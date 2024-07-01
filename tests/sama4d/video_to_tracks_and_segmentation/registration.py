"""
This file registers a default set of tasks and solvers for the video to keypoint tracks & segmentation task class.
"""

from ..data_curation import get_loaders_for_all_curated_scenes
from .keypoint_tracking_and_segmentation_task import KeypointTrackingAndSegmentationTask
from .dummy_solver import KeypointTrackingAndSegmentationDummySolver

all_tasks = [
    KeypointTrackingAndSegmentationTask(spec["feature_track_data_loader"], scene_name=spec["scene_name"], n_frames=3)
    for spec in get_loaders_for_all_curated_scenes()
]
all_solvers = [KeypointTrackingAndSegmentationDummySolver()]

all_task_solver_pairs = [(task, solver) for task in all_tasks for solver in all_solvers]
