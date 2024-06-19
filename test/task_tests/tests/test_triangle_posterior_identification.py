import jax.numpy as jnp
import test.task_tests.common as common
import test.task_tests.tasks.triangle_posterior_identification as tpi
from test.task_tests.solvers.triangle_posterior_identification.dummy_solver import dummy_solver
import test.task_tests.solvers.triangle_posterior_identification.gt_initialized_importance_sampling_solver as gtiis

TPIT = tpi.TrianglePosteriorIdentificationTask

TASK_CONFIGS = [
    {
        "name": f"{bc_name}-background--{tc_name}-triangle--oneframe",
        "task_args": TPIT.generate_unichromatic_background_single_frame_taskargs(
            bc, tc,
            f"{bc_name}-background--{tc_name}-triangle--oneframe"
        )
    }
    for bc_name, bc in [("white", jnp.array([1., 1., 1.])), ("black", jnp.array([0., 0., 0.]))]
    for tc_name, tc in [("red", jnp.array([1., 0., 0.])), ("green", jnp.array([0., 1., 0.]))]
]

SOLVER_CONFIGS = [
    {
        "name": "dummy_solver",
        "solver": dummy_solver,
        "solver_args": {}
    },
    {
        "name": "importance_sampling_solver_w_gt_knowledge",
        "solver": gtiis.importance_solver,
        "solver_args": {}
    }
]

tasks, solvers = common.register_tasks_and_solvers(
    tpi.TrianglePosteriorIdentificationTask, TASK_CONFIGS, SOLVER_CONFIGS
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
