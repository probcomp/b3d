import jax
import jax.numpy as jnp
from .task import TrianglePosteriorGridApproximationTask
from .solver.importance import ImportanceSolver
import pytest

task_specs = [
    (background_color, triangle_color)
    for background_color in [jnp.array([1.0, 1.0, 1.0]), jnp.array([0.0, 0.0, 0.0])]
    for triangle_color in [jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 0.0])]
]


# Only run one test for now, to prevent issues due to the current
# memory leak in the renderer.
@pytest.mark.parametrize("task_spec", task_specs[:1])
def test(task_spec):
    task = TrianglePosteriorGridApproximationTask.default_scene_using_colors(*task_spec)
    task.run_tests(ImportanceSolver(), divergence_from_uniform_threshold=jnp.inf)
