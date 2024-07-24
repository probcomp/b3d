# This script can be run to visualize all tasks and solvers in the sama4d dataset.

import sys

<<<<<<< HEAD
=======
import b3d

>>>>>>> main
sys.path.append(b3d.get_root_path())

import rerun as rr

from tests.sama4d.tracks_to_segmentation.registration import (
    all_task_solver_pairs as pairs_2,
)
from tests.sama4d.video_to_tracks.registration import all_task_solver_pairs as pairs_1
<<<<<<< HEAD
from tests.sama4d.tracks_to_segmentation.registration import (
    all_task_solver_pairs as pairs_2,
)
=======
>>>>>>> main
from tests.sama4d.video_to_tracks_and_segmentation.registration import (
    all_task_solver_pairs as pairs_3,
)

# For now, only run the first two tasks and solvers from each group.
# Due to a memory leak in `b3d.Renderer`, if we run more tasks,
# the GPU runs out of memory.
# Fixing this is a priority for the week of July 1, 2024.
for groupname, pairs in [
<<<<<<< HEAD
    ("Video To Tracks", pairs_1[:2]),
    ("Tracks To Segmentation", pairs_2[:2]),
    ("Video To Tracks And Segmentation", pairs_3[:2]),
=======
    ("Video To Tracks", [pairs_1[0], pairs_1[-1]]),
    ("Tracks To Segmentation", [pairs_2[0], pairs_2[-1]]),
    ("Video To Tracks And Segmentation", [pairs_3[0], pairs_3[-1]]),
>>>>>>> main
]:
    print(f"****************{groupname}****************")
    for task, solver in pairs:
        rr.init(f"sama4d_TASK:{task.name}_SOLVER:{solver.name}")
        rr.connect("127.0.0.1:8812")
        metrics = task.run_solver_and_make_all_visualizations(solver)

        print(f"----Metrics for solver {solver.name} on task {task.name}----")
        for key, value in metrics.items():
            print(f"\t{key}:\t {value}")
