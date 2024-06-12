import jax
import jax.numpy as jnp
import b3d
import b3d.likelihoods
import genjax
from .model import model_factory, get_likelihood, rr_log_trace

RENDERER_HYPERPARAMS = b3d.differentiable_renderer.DifferentiableRendererHyperparams(3, 1e-5, 1e-2, -1)

def importance_sample_with_depth_in_partition(
    key, task_input, model, mindepth, maxdepth
):
    """
    triangle will be a 3x3 array [v1, v2, v3] (not normalized! at true pose!)

    P(depth | image, depth \in [mindepth, maxdepth) )
    w ~~ P( image, depth \in [mindepth, maxdepth) )
    """
    k1, k2, k3 = jax.random.split(key, 3)
    proposed_depth = genjax.uniform.sample(k1, mindepth, maxdepth)
    log_q = genjax.uniform.logpdf(proposed_depth, mindepth, maxdepth)

    X_WC = task_input["camera_path"][0]
    triangle_W = task_input["cheating_info"]["triangle_vertices"]
    triangle_C = X_WC.inv().apply(triangle_W)

    og_triangle_depth = triangle_C[0, 2]
    size_at_camera = 
    triangle_size = (proposed_depth - og_triangle_depth
    mean_pos_C = jnp.array([triangle_C[0, 0], triangle_C[0, 1], proposed_depth])
    exact_position_C = genjax.normal.sample(k2, mean_pos_C, 1e-8)
    log_q += genjax.normal.logpdf(exact_position_C, mean_pos_C, 1e-8)

    exact_position_W = X_WC.apply(exact_position_C)

    trace, log_p_score = model.importance(
        key,
        genjax.choice_map({
            "triangle_xyz": exact_position_W,
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
        key = jax.random.PRNGKey(1)
        renderer = task_input["renderer"]
        model = model_factory(renderer, get_likelihood(renderer), RENDERER_HYPERPARAMS)
    
        partition_starts = partition[:-1]
        partition_ends = partition[1:]
        
        @jax.jit
        def get_tr_and_score(k, s, e):
            tr, score = importance_sample_with_depth_in_partition(k, task_input, model, s, e)
            return (tr, score)
        
        trs_and_scores = [
            get_tr_and_score(k, s, e) for (k, s, e) in zip(jax.random.split(key, len(partition_starts)), partition_starts, partition_ends)
        ]
        trs = [ts[0] for ts in trs_and_scores]
        joint_scores = jnp.array([ts[1] for ts in trs_and_scores])

        for i in range(len(trs)):
            if i % 20 == 0:
                rr_log_trace(trs[i], task_input["renderer"], prefix=f"trace_{i}")

        return jnp.exp(joint_scores - jax.scipy.special.logsumexp(joint_scores))
    
    return {
        "laplace": {},
        "grid": grid_solver,
        "mala": {},
        "multi_initialized_mala": {}
    }