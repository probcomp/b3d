import jax
import jax.numpy as jnp
from .task import TrianglePosteriorGridApproximationTask
from .solver.importance import ImportanceSolver
import pytest

tasks = [
    TrianglePosteriorGridApproximationTask.default_scene_using_colors(
        background_color, triangle_color
    )
    for background_color in [jnp.array([1., 1., 1.]), jnp.array([0., 0., 0.])]
    for triangle_color in [jnp.array([1., 0., 0.]), jnp.array([0., 1., 0.])]
]

jax.clear_caches()
@pytest.mark.parametrize("task", tasks)
def test(task):
    task.run_tests(
        ImportanceSolver(),
        divergence_from_uniform_threshold=jnp.inf
    )