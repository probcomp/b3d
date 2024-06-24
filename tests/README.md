# b3d `tests/`

This directory currently contains assertion tests for the `b3d` package,
and some code for other evaluation of `b3d` functionality (which can
terminate in assertion testing, computation of metrics, or production
of visualizations).

Any file with `_test` or `test_` in the name will get run by the assertion
testing system (which uses `pytest`, and is triggered when running `./run_tests.sh`).

## Running the code
All code in `tests/` expects to be run in a Python session triggered from the top-level directory of the `b3d` repository.
In notebooks, this can be accomplished by adding
```python
import b3d
import sys
sys.path.append(b3d.get_root_path())
```
to the top of the notebook.  Then imports like
```python
from tests.common.task import Task
```
will work.

## Current directry structure
- Top level python files: various assertion tests
- `common`: code to be imported by other testing/assertion files.  Currently contains:
  - `task.py`: lightweight base class defining tasks.  Useful when running a piece of inference functionality across a number of test scenes and visualizing or quantitatively evaluating the results.
  A Task defines a "task specification" (input format given to
  the task solver), a proceure for computing metrics scoring a solution
  to the task, and optionally, a visualization function for the task.
  - `solver.py`: very lightweight base class defining a solver for the Tasks.
- `dense_model_unit_tests`: some tests for the dense model and inference.
- `sama4d`: integration tests for major parts of the SAMA4D system.
  - `data_curation.py`: this file collects high-quality test scenes to use for evaluation.
  - `feature_track_data_task.py`: subclass of `Task` wrapping a `b3d.io.FeatureTrackData` and providing a default visualization. Downstream tasks should define different solution formats and scoring procedures.
  - `keypoint_tracking/`: Task, patch-tracking solver, assertion testing, and visualization notebooks, for tracking keypoints from RGB(D) video, given the initial 2D keypoint positions.
  - `keypoint_tracking_and_segmentation/`: [WIP] Task, solver, assertion testing, and visualization notebooks, for tracking keypoints from RGB(D) video, given the initial 2D keypoint positions, and segmenting the tracked keypoints into independently moving objects.  (This is a first test for the integrated SAMA4D system.)

Please feel free to make PRs proposing modifications to this structure, as needed.

## TODOs

Some TODOs:
- Finish a version of the `keypoint_tracking_and_segmentation/` task.
- Add a test for keypoint tracking and segmentation from 2D keypoint tracks, not video.  (This may require modifying `feature_track_data_task.py` to make it less specialized to the current subclasses.)
- Add a cotracker solver for `keypoint_tracking/`.

Possible TODOs, to discuss after a first integrated SAMA4D test has been run, and the genjax 0.4 port is complete:
- Refactor the repo, to move code other than assertion testing currently stored under `tests/` to a new directory `b3d/eval/`.