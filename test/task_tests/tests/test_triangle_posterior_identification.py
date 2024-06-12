import jax.numpy as jnp
import b3d
import test.task_tests.common as common
import test.task_tests.tasks.triangle_posterior_identification as tpi
from test.task_tests.solvers.triangle_posterior_identification.dummy_solver import dummy_solver
import test.task_tests.solvers.triangle_posterior_identification.gt_initialized_importance_sampling_solver as gtiis
import importlib
import jax

importlib.reload(tpi)
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

# task = TPIT(**TASK_CONFIGS[0]["task_args"])
# task.visualize_scene()

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

### 

# import rerun as rr

# task = tasks[0]
# task_input, baseline = task.get_test_pair()
# task.visualize_scene()


# task.score_grid_approximation(task_input, baseline, solvers[0](task_input)["grid"])




# partition, _ = task._get_grid_partition()
# pts_C = partition[:, None] * jnp.array([[0., 0., 1.]])
# X_WC = task.camera_path[0]
# pts_W = X_WC.apply(pts_C)
# posterior_approximation = jnp.ones_like(partition[:-1]) / len(partition[:-1])
# rr.log(
#     "/3D/grid_posterior_approximation/camera_z_partition",
#     rr.LineStrips3D(
#         [jnp.array([pts_W[i], pts_W[i+1]]) for i in range(len(pts_W) - 1)],
#         colors=[[a, a, a] for a in jnp.clip(posterior_approximation, 0., 1.)]
#     )
# )



###
import test.task_tests.solvers.triangle_posterior_identification.model as m
key = jax.random.PRNGKey(0)

importlib.reload(gtiis)
importlib.reload(m)
task = tasks[0]
key, = jax.random.split(key, 1)

task = tasks[0]
task_input, baseline = task.get_test_pair()

renderer = task_input["renderer"]
# model = gtiis.model_factory(
#     renderer, gtiis.get_likelihood(renderer), gtiis.RENDERER_HYPERPARAMS
# )
# trace, weight = gtiis.importance_sample_with_depth_in_partition(
#     key, task_input, model, -1., 2.0
# )

# dist = gtiis.importance_solver(task_input)["grid"](task._get_grid_partition()[0])

task.visualize_scene()
task.score_grid_approximation(task_input, baseline, gtiis.importance_solver(task_input)["grid"])

# task.maybe_initialize_rerun()
# m.rr_log_trace(trace, task.renderer)
# task.visualize_scene()