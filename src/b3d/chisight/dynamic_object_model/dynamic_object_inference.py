from functools import partial

import jax
import jax.numpy as jnp
from genjax import ChoiceMapBuilder as C
from genjax import Diff, Pytree
from genjax import UpdateProblemBuilder as U

import b3d
from b3d import Pose

from .dynamic_object_model import info_from_trace

### Inference moves ###


@jax.jit
def score_with_give_pose_and_then_color_update(
    trace, address, pose, outlier_probability_sweep
):
    trace = b3d.update_choices(trace, address, pose)

    info = info_from_trace(trace)

    color_delta = info["observed_rgbd_masked"][..., :3] - trace.get_choices()["colors"]
    max_color_shift = trace.get_args()[0]["max_color_shift"].const
    color_delta_clipped = jnp.clip(color_delta, -max_color_shift, max_color_shift)

    trace = b3d.update_choices(
        trace,
        Pytree.const(("colors",)),
        trace.get_choices()["colors"] + color_delta_clipped,
    )

    trace = grid_move_on_outlier_probability(
        trace, Pytree.const(("color_outlier_probability",)), outlier_probability_sweep
    )
    trace = grid_move_on_outlier_probability(
        trace, Pytree.const(("depth_outlier_probability",)), outlier_probability_sweep
    )

    return trace.get_score()


@partial(jax.jit, static_argnames=("number",))
def gaussian_vmf_enumerative_move_with_other_updates(
    trace, key, address, variance, concentration, number, outlier_probability_sweep
):
    keys = jax.random.split(key, number)
    poses = Pose.concatenate_poses(
        [
            Pose.sample_gaussian_vmf_pose_vmap(
                keys, trace.get_choices()["pose"], 0.02, 2000.0
            ),
            trace.get_choices()["pose"][None, ...],
        ]
    )

    scores = jax.vmap(
        score_with_give_pose_and_then_color_update, in_axes=(None, None, 0, None)
    )(trace, Pytree.const(("pose",)), poses, outlier_probability_sweep)

    key = b3d.split_key(keys[-1])
    sampled_pose = poses[scores.argmax()]
    trace = b3d.update_choices(trace, Pytree.const(("pose",)), sampled_pose)
    return trace, key


@jax.jit
def grid_move_on_outlier_probability(trace, address, sweep):
    current_setting = trace.get_choices()[address.const]

    potential_values = sweep[..., None] * jnp.ones_like(current_setting)

    get_vertex_scores_vmap = jax.vmap(
        lambda x: info_from_trace(b3d.update_choices(trace, address, x))["scores"]
    )
    scores = get_vertex_scores_vmap(potential_values)
    best_setting = sweep[jnp.argmax(scores, axis=0)]
    return b3d.update_choices(trace, address, best_setting)


@jax.jit
def update_address_with_sweep(trace, address, sweep):
    scores = b3d.enumerate_choices_get_scores(trace, address, sweep)
    best_setting = sweep[jnp.argmax(scores)]
    return b3d.update_choices(trace, address, best_setting)


@jax.jit
def advance_time(key, trace, observed_rgbd):
    """
    Advance to the next timestep, setting the new latent state to the
    same thing as the previous latent state, and setting the new
    observed RGBD value.

    Returns a trace where previous_state (stored in the arguments)
    and new_state (sampled in the choices and returned) are identical.
    """
    hyperparams, _ = trace.get_args()
    previous_state = trace.get_retval()["new_state"]
    trace, _, _, _ = trace.update(
        key,
        U.g(
            (Diff.no_change(hyperparams), Diff.unknown_change(previous_state)),
            C.kw(rgbd=observed_rgbd),
        ),
    )
    return trace


### Inference step loop ###

# def georges_proposed_inference_step(trace):
#     trace = advance_time(trace)

#     for i in range(4):
#         trace = outlier_prob_sweep(trace)
#         trace = color_variance_sweep(trace)
#         trace = depth_variance_sweep(trace)

#     # TODO


def inference_step(trace, key, observed_rgbd):
    trace = advance_time(key, trace, observed_rgbd)

    outlier_probability_sweep = jnp.array([0.01, 0.5, 1.0])
    num_grid_points = 20000
    for _ in range(2):
        trace, key = gaussian_vmf_enumerative_move_with_other_updates(
            trace,
            key,
            Pytree.const(("pose",)),
            0.04,
            200.0,
            num_grid_points,
            outlier_probability_sweep,
        )
        trace, key = gaussian_vmf_enumerative_move_with_other_updates(
            trace,
            key,
            Pytree.const(("pose",)),
            0.01,
            500.0,
            num_grid_points,
            outlier_probability_sweep,
        )
        trace, key = gaussian_vmf_enumerative_move_with_other_updates(
            trace,
            key,
            Pytree.const(("pose",)),
            0.005,
            1000.0,
            num_grid_points,
            outlier_probability_sweep,
        )
        trace, key = gaussian_vmf_enumerative_move_with_other_updates(
            trace,
            key,
            Pytree.const(("pose",)),
            0.001,
            2000.0,
            num_grid_points,
            outlier_probability_sweep,
        )

    trace = grid_move_on_outlier_probability(
        trace, Pytree.const(("color_outlier_probability",)), outlier_probability_sweep
    )
    trace = grid_move_on_outlier_probability(
        trace, Pytree.const(("depth_outlier_probability",)), outlier_probability_sweep
    )

    trace = update_address_with_sweep(
        trace,
        Pytree.const(("color_variance",)),
        jnp.array([0.005, 0.01, 0.05, 0.1, 0.2]),
    )
    trace = update_address_with_sweep(
        trace,
        Pytree.const(("depth_variance",)),
        jnp.array([0.0005, 0.001, 0.0025, 0.005, 0.01]),
    )

    info = info_from_trace(trace)
    color_delta = info["observed_rgbd_masked"][..., :3] - trace.get_choices()["colors"]
    max_color_shift = trace.get_args()[0]["max_color_shift"].const
    color_delta_clipped = jnp.clip(color_delta, -max_color_shift, max_color_shift)
    trace = b3d.update_choices(
        trace,
        Pytree.const(("colors",)),
        trace.get_choices()["colors"] + color_delta_clipped,
    )
    return trace


#### New Inference Programs
def propose_depth_outlier_probability(trace, key, depth_outlier_probability_sweep):
    # depth_outlier_probability_sweep is (k.) shape array

    current_depth_outlier_probabilities = trace.get_choices()[
        "depth_outlier_probability"
    ]  # (num_vertices,)

    depth_outlier_probability_sweep_full = (
        depth_outlier_probability_sweep[..., None]
        * jnp.ones_like(current_depth_outlier_probabilities)
    )  # (k, num_vertices) where the values in row i are all depth_outlier_probability_sweep[i]

    # Function takes in depth outlier probability array of shape (num_vertices,) and gives scores for each vertex (num_vertices,)
    get_per_vertex_likelihoods_with_new_depth_outlier_probabilities = (
        lambda depth_outlier_probs: info_from_trace(
            b3d.update_choices(
                trace, Pytree.const(("depth_outlier_probability",)), depth_outlier_probs
            )
        )["scores"]
    )

    # Vmap over the depth_outlier_probability_sweep_full array to get scores for each vertex for each depth_outlier_probability in the sweep
    likelihood_scores_per_sweep_point_and_vertex = jax.vmap(
        get_per_vertex_likelihoods_with_new_depth_outlier_probabilities
    )(depth_outlier_probability_sweep_full)  # (k, num_vertices)
    # P(pixel / nothing  |  depth outlier prob, latent point)

    transition_scores_per_sweep_point_and_vertex = jax.vmap(
        jax.vmap(depth_outlier_transition_kernel.logpdf, in_axes=(0, 0)),
        in_axes=(0, None),
    )(
        depth_outlier_probability_sweep_full, current_depth_outlier_probabilities
    )  # (k, num_vertices)

    scores_per_sweep_point_and_vertex = (
        likelihood_scores_per_sweep_point_and_vertex
        + transition_scores_per_sweep_point_and_vertex
    )
    normalized_log_probabilities = jax.nn.log_softmax(
        scores_per_sweep_point_and_vertex, axis=0
    )

    sampled_indices = jax.random.categorical(key, normalized_log_probabilities, axis=0)
    sampled_depth_outlier_probabilities = depth_outlier_probability_sweep[
        sampled_indices
    ]
    log_q_depth_outlier_probability = normalized_log_probabilities[
        sampled_indices, jnp.arange(normalized_log_probabilities.shape[1])
    ].sum()

    return sampled_depth_outlier_probabilities, log_q_depth_outlier_probability


def propose_pose(trace, key, pose_sample_variance, pose_sample_concentration):
    pose = Pose.sample_gaussian_vmf_pose_vmap(
        key,
        trace.get_choices()["pose"],
        pose_sample_variance,
        pose_sample_concentration,
    )
    log_q_pose = Pose.logpdf_gaussian_vmf_pose_vmap(
        pose,
        trace.get_choices()["pose"],
        pose_sample_variance,
        pose_sample_concentration,
    )
    return pose, log_q_pose


# def propose_color_and_color_outlier_probability_for_point(
#     trace, key, color_outlier_probability_sweep
# ):
#     # color_outlier_probability_sweep is (k,) shape array

#     current_color_outlier_probability = trace.get_choices()["color_outlier_probability"]
#     current_colors = trace.get_choices()["colors"]

#     num_vertices = current_colors.shape[0]
#     k = color_outlier_probability_sweep.shape[0]

#     color_outlier_probabilities_sweep_full = jnp.zeros(
#         (
#             k,
#             num_vertices,
#         )
#     )
#     colors_sweep = jnp.zeros((k, num_vertices, 3))  # TODO: propose this intelligent

#     # Function takes in color and color outlier probabilities array of shapes (num_vertices,3) and (num_vertices,) respectively
#     # and gives scores for each vertex (num_vertices,)

#     def get_per_vertex_likelihoods_with_new_color_and_color_outlier_probabilities(
#         color, color_outlier_probability
#     ):
#         return info_from_trace(
#             b3d.update_choices(
#                 trace,
#                 Pytree.const(
#                     (
#                         "color",
#                         "color_outlier_probability",
#                     )
#                 ),
#                 color,
#                 color_outlier_probability,
#             )
#         )["scores"]

#     # Vmap over the depth_outlier_probability_sweep_full array to get scores for each vertex for each depth_outlier_probability in the sweep
#     likelihood_scores_per_sweep_point_and_vertex = jax.vmap(
#         get_per_vertex_likelihoods_with_new_color_and_color_outlier_probabilities
#     )(colors_sweep, color_outlier_probabilities_sweep_full)  # (k, num_vertices)
#     return likelihood_scores_per_sweep_point_and_vertex
#     # color_outlier_transition_scores_per_sweep_point_and_vertex = jax.vmap(
#     #     jax.vmap(color_outlier_transition_kernel.logpdf, in_axes=(0, 0)),
#     #     in_axes=(0, None),
#     # )(
#     #     color_outlier_probabilities_sweep_full, current_color_outlier_probability
#     # )  # (k, num_vertices)

#     # color_transition_scores_per_sweep_point_and_vertex = jax.vmap(
#     #     jax.vmap(color_transition_kernel.logpdf, in_axes=(0, 0)), in_axes=(0, None)
#     # )(colors_sweep, current_colors)  # (k, num_vertices)

#     # scores_per_sweep_point_and_vertex = (
#     #     likelihood_scores_per_sweep_point_and_vertex
#     #     + color_outlier_transition_scores_per_sweep_point_and_vertex
#     #     + color_transition_scores_per_sweep_point_and_vertex
#     # )
#     # normalized_log_probabilities = jax.nn.log_softmax(
#     #     scores_per_sweep_point_and_vertex, axis=0
#     # )

#     # sampled_indices = jax.random.categorical(key, normalized_log_probabilities, axis=0)

#     # sampled_colors = colors_sweep[sampled_indices]
#     # sampled_color_outlier_probabilities = color_outlier_probability_sweep[
#     #     sampled_indices
#     # ]
#     # log_q_color_color_outlier_probability = normalized_log_probabilities[
#     #     sampled_indices, jnp.arange(normalized_log_probabilities.shape[1])
#     # ].sum()

#     # return (
#     #     sampled_colors,
#     #     sampled_color_outlier_probabilities,
#     #     log_q_color_color_outlier_probability,
#     # )


# def propose_update(trace, key, pose_sample_variance, pose_sample_concentration):
#     total_log_q = 0.0

#     # Update pose
#     pose, log_q_pose = propose_pose(
#         trace, key, pose_sample_variance, pose_sample_concentration
#     )
#     trace = b3d.update_choices(trace, Pytree.const(("pose",)), pose)
#     total_log_q += log_q_pose

#     # Update depth outlier probability
#     depth_outlier_probability_sweep = jnp.array([0.01, 0.5, 1.0])
#     depth_outlier_probability, log_q_depth_outlier_probability = (
#         propose_depth_outlier_probability(trace, key, depth_outlier_probability_sweep)
#     )
#     trace = b3d.update_choices(
#         trace, Pytree.const(("depth_outlier_probability",)), depth_outlier_probability
#     )
#     total_log_q += log_q_depth_outlier_probability
