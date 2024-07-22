import jax.numpy as jnp
from .task import TrianglePosteriorGridApproximationTask
from .solver.importance import (
    SimpleLikelihoodImportanceSolver,
    DiffrendImportanceSolver,
)
from b3d.chisight.dense.differentiable_renderer import DifferentiableRendererHyperparams
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

    # This test should pass with the simple depth likelihood.
    task.run_tests(SimpleLikelihoodImportanceSolver())

    # I also expect this to pass with the differentiable likelihood
    # when the sigma parameter is extremely low.
    task.run_tests(
        DiffrendImportanceSolver(DifferentiableRendererHyperparams(3, 1e-15, 1e-2, -1))
    )

    # As of July 22, 2024, I don't expect the test to pass
    # using the differentiable renderer with
    # the hyperparameters I tested this renderer with for
    # gradient-based patch tracking.
