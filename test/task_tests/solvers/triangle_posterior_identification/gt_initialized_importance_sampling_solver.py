import jax
import jax.numpy as jnp
import b3d
import b3d.likelihoods
import genjax
from .model import model_factory, get_likelihood

RENDERER_HYPERPARAMS = b3d.differentiable_renderer.DifferentiableRendererHyperparams(3, 1e-5, 1e-2, -1)

def importance_sample_with_depth_in_partition(
    key, triangle, task_input, model, mindepth, maxdepth
):
    """
    triangle will be a 3x3 array [v1, v2, v3] (not normalized! at true pose!)

    P(depth | image, depth \in [mindepth, maxdepth) )
    w ~~ P( image, depth \in [mindepth, maxdepth) )
    """
    proposed_depth = genjax.uniform.sample(key, mindepth, maxdepth)
    log_q = genjax.uniform.logpdf(proposed_depth, mindepth, maxdepth)

    og_triangle_depth = triangle[0, 1]
    triangle_size = proposed_depth / og_triangle_depth

    trace, log_p_score = model.importance(
        key,
        genjax.choice_map({
            "triangle_xyz": jnp.array([
                triangle[0, 0], proposed_depth, triangle[0, 2]
            ]),
            "triangle_size": triangle_size,
            "observed_rgbs": genjax.vector_choice_map(genjax.choice_map({
                "observed_rgb": task_input["video"]
            }))
        }),
        (
            task_input["background_mesh"],
            task_input["triangle"]["vertices"],
            task_input["triangle"]["color"],
            task_input["camera_path"]
        )
    )

    return trace, log_p_score - log_q

def importance_solver(task_input):
    def grid_solver(partition):
        renderer = task_input["renderer"]
        model = model_factory(renderer, get_likelihood(renderer), RENDERER_HYPERPARAMS)
    
        joint_scores = jax.vmap(importance_sample_with_depth_in_partition)(partition)
        return joint_scores / jnp.sum(joint_scores)
    
    return {
        "laplace": {},
        "grid": grid_solver,
        "mala": {},
        "multi_initialized_mala": {}
    }