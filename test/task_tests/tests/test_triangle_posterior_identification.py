import jax.numpy as jnp
import b3d
import test.task_tests.common as common
import test.task_tests.tasks.triangle_posterior_identification as tpi
from test.task_tests.solvers.triangle_posterior_identification_solver import dummy_solver
import importlib

importlib.reload(tpi)
TPIT = tpi.TrianglePosteriorIdentificationTask

TASK_CONFIGS = [
    {
        "name": "monochrome-background--no-motion--oneframe",
        "task_args": {
            "scene_background": TPIT.get_monochrome_room_background(jnp.array([0., 0., 1.])),
            "foreground_triangle": {
                "color": jnp.array([1., 1., 1.]),
                "vertices": jnp.array([
                    [5, 3, 7], [6, 1.5, 7], [6, 1.5, 8]   
                ], dtype=float)
            },
            "triangle_path": b3d.Pose.identity()[None, ...],
            "camera_path": b3d.camera_from_position_and_target(
                position=jnp.array([5., -0.5, 6.]),
                target=jnp.array([5, 3, 7.])
            )[None, ...]
        }
    }
]

# task = TPIT(**TASK_CONFIGS[0]["task_args"])
# task.visualize_scene()

SOLVER_CONFIGS = [
    {
        "name": "dummy_solver",
        "solver": dummy_solver,
        "solver_args": {}
    }
]

tasks, solvers = common.register_tasks_and_solvers(
    TPIT, TASK_CONFIGS, SOLVER_CONFIGS
)

TEST_CONFIGS = [{
        "testcase_name": f"Task::{TASK_CONFIGS[ti]['name']};;--;;Solver::{SOLVER_CONFIGS[si]['name']}",
        "solver": solvers[si], "task": tasks[ti]
    } for ti in range(len(TASK_CONFIGS)) for si in range(len(SOLVER_CONFIGS))
]
class TrianglePosteriorIdentificationTest(common.TestCase):
    @common.named_parameters(TEST_CONFIGS)
    def test(self, solver, task, *args, **kwargs):
        task.run_tests(self, solver, *args, **kwargs)

# tester = TrianglePosteriorIdentificationTest()
# tester.setUpClass()
# tester.setUp()
# tester.test(solvers[0], tasks[0])