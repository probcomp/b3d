import jax
import jax.numpy as jnp
import jax.random
from genjax import ChoiceMapBuilder as C
from genjax import Diff
from genjax import UpdateProblemBuilder as U

from b3d import Pose

from .dynamic_object_model import (
    info_from_trace,
    make_color_outlier_probabilities_choicemap,
    make_colors_choicemap,
    make_depth_outlier_probabilities_choicemap,
)


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


#### New Inference Programs


# Propose new depth outlier probabilities
@jax.jit
def propose_depth_outlier_probability(trace, key):
    depth_outlier_probability_sweep = jnp.array([0.01, 0.5, 1.0])  # (k, )

    current_depth_outlier_probabilities = trace.get_args()[1][
        "depth_outlier_probabilities"
    ]  # (num_vertices, )

    depth_outlier_probability_sweep_full = (
        depth_outlier_probability_sweep[..., None]
        * jnp.ones_like(current_depth_outlier_probabilities)
    )  # (k, num_vertices) where the values in row i are all depth_outlier_probability_sweep[i]

    # Function takes in depth outlier probability array of shape (num_vertices,) and gives scores for each vertex (num_vertices,)

    def get_per_vertex_likelihoods_with_new_depth_outlier_probabilities(
        depth_outlier_probabilities,
    ):
        return info_from_trace(
            trace.update(
                key,
                make_depth_outlier_probabilities_choicemap(depth_outlier_probabilities),
            )[0]
        )["scores"]

    # Vmap over the depth_outlier_probability_sweep_full array to get scores for each vertex for each depth_outlier_probability in the sweep
    likelihood_scores_per_sweep_point_and_vertex = jax.vmap(
        get_per_vertex_likelihoods_with_new_depth_outlier_probabilities
    )(depth_outlier_probability_sweep_full)  # (k, num_vertices)
    # P(pixel / nothing  |  depth outlier prob, latent point)

    depth_outlier_probability_transition_kernel = trace.get_args()[0][
        "depth_outlier_probability_transition_kernel"
    ]
    vectorized_outlier_probability_transition_kernel_logpdf = jnp.vectorize(
        depth_outlier_probability_transition_kernel.logpdf, signature="(),()->()"
    )

    transition_scores_per_sweep_point_and_vertex = (
        vectorized_outlier_probability_transition_kernel_logpdf(
            depth_outlier_probability_sweep_full,  # (k, num_vertices)
            current_depth_outlier_probabilities,  # (num_vertices, )
        )
    )  # (k, num_vertices)

    scores_per_sweep_point_and_vertex = (
        likelihood_scores_per_sweep_point_and_vertex
        + transition_scores_per_sweep_point_and_vertex
    )
    normalized_log_probabilities = jax.nn.log_softmax(
        scores_per_sweep_point_and_vertex, axis=0
    )

    sampled_indices = jax.random.categorical(key, normalized_log_probabilities, axis=0)
    # sampled_indices = jnp.argmax(scores_per_sweep_point_and_vertex, axis=0)

    sampled_depth_outlier_probabilities = depth_outlier_probability_sweep[
        sampled_indices
    ]

    # normalized_log_probabilities is (k, num_vertices)
    log_q_depth_outlier_probability = normalized_log_probabilities[
        sampled_indices, jnp.arange(normalized_log_probabilities.shape[1])
    ].sum()

    return (
        sampled_depth_outlier_probabilities,
        log_q_depth_outlier_probability,
        scores_per_sweep_point_and_vertex,
    )


# @jax.jit
# def propose_pose(trace, key, pose_sample_variance, pose_sample_concentration):
#     pose = Pose.sample_gaussian_vmf_pose(
#         key,
#         trace.get_choices()["pose"],
#         pose_sample_variance,
#         pose_sample_concentration,
#     )
#     log_q_pose = Pose.logpdf_gaussian_vmf_pose(
#         pose,
#         trace.get_choices()["pose"],
#         pose_sample_variance,
#         pose_sample_concentration,
#     )
#     return pose, log_q_pose


@jax.jit
def propose_color_and_color_outlier_probability(trace, key, outlier_probability_sweep):
    # color_outlier_probability_sweep is (k,) shape array

    current_color_outlier_probabilities = trace.get_args()[1][
        "color_outlier_probabilities"
    ]
    current_colors = trace.get_args()[1]["colors"]

    # num_vertices = current_colors.shape[0]
    # num_outlier_grid_points = outlier_probability_sweep.shape[0]
    color_outlier_probabilities_sweep = (
        outlier_probability_sweep[..., None]  # (num_outlier_grid_points, 1)
        * jnp.ones_like(current_color_outlier_probabilities)  # (num_vertices,)
    )  # (num_outlier_grid_points, num_vertices)

    # We will grid over color values, using a grid that mixes the old and observed
    # colors in a set of exact proportions.
    # We regard these as coming from uniform proposals where we sample the RGB
    # values uniformly between the mixed R, G, and B values with mixtures between
    # [0., .125], [.125, .5], [.5, .875], [.875, 1.].
    # So the q scores will be .125^3, .375^3, .375^3, .125^3.
    # TODO: we really ought to add a small amount of proposal probability mass
    # onto the points at the end, to capture the fact that the posterior could allow
    # colors outside the considered interpolation window.
    color_interpolations_per_proposal = jnp.array([0.0, 0.5, 1.0])
    # num_color_grid_points = len(color_interpolations_per_proposal)

    observed_colors = info_from_trace(trace)["observed_rgbd_masked"][
        ..., :3
    ]  # (num_vertices, 3)
    color_sweep = observed_colors[None, ...] * color_interpolations_per_proposal[
        :, None, None
    ] + current_colors[None, ...] * (
        1 - color_interpolations_per_proposal[:, None, None]
    )  # (num_color_grid_points, num_vertices, 3)

    # Function takes in color and color outlier probabilities array of shapes (num_vertices,3) and (num_vertices,) respectively
    # and gives scores for each vertex (num_vertices,)
    def get_per_vertex_likelihoods_with_new_color_and_color_outlier_probabilities(
        colors, color_outlier_probabilities
    ):
        return info_from_trace(
            trace.update(
                key,
                make_color_outlier_probabilities_choicemap(color_outlier_probabilities)
                ^ make_colors_choicemap(colors),
            )[0]
        )["scores"]

    vmap_version = jax.vmap(
        jax.vmap(
            get_per_vertex_likelihoods_with_new_color_and_color_outlier_probabilities,
            in_axes=(None, 0),
        ),
        in_axes=(0, None),
    )

    # Vmap over the depth_outlier_probability_sweep_full array to get scores for each vertex for each depth_outlier_probability in the sweep
    likelihood_scores_per_sweep_point_and_vertex = vmap_version(
        color_sweep, color_outlier_probabilities_sweep
    )  # (num_color_grid_points, num_outlier_grid_points, num_vertices)

    # Color outlier transition scores
    color_outlier_probability_transition_kernel = trace.get_args()[0][
        "color_outlier_probability_transition_kernel"
    ]
    vectorized_outlier_probability_transition_kernel_logpdf = jnp.vectorize(
        color_outlier_probability_transition_kernel.logpdf, signature="(),()->()"
    )
    color_outlier_transition_scores_per_sweep_point_and_vertex = vectorized_outlier_probability_transition_kernel_logpdf(
        color_outlier_probabilities_sweep,  # (num_outlier_grid_points, num_vertices)
        current_color_outlier_probabilities,
    )  # (num_outlier_grid_points, num_vertices)

    color_transition_kernel = trace.get_args()[0]["color_transition_kernel"]
    vectorized_color_transition_kernel_logpdf = jnp.vectorize(
        color_transition_kernel.logpdf, signature="(3),(3)->()"
    )
    color_transition_scores_per_sweep_point_and_vertex = (
        vectorized_color_transition_kernel_logpdf(color_sweep, current_colors)
    )  # (num_color_grid_points, num_vertices)

    scores_per_sweep_point_and_vertex = (
        likelihood_scores_per_sweep_point_and_vertex  # (num_color_grid_points, num_outlier_grid_points, num_vertices)
        + color_outlier_transition_scores_per_sweep_point_and_vertex[None, ...]
        + color_transition_scores_per_sweep_point_and_vertex[:, None, ...]
    )  # (num_color_grid_points, num_outlier_grid_points, num_vertices)

    unraveled_scores = scores_per_sweep_point_and_vertex.reshape(
        -1, scores_per_sweep_point_and_vertex.shape[-1]
    )
    normalized_log_probabilities = jax.nn.log_softmax(unraveled_scores, axis=0)
    sampled_indices = jax.random.categorical(key, normalized_log_probabilities, axis=0)

    color_sweep_indices, color_outlier_sweep_indices = jnp.unravel_index(
        sampled_indices, scores_per_sweep_point_and_vertex.shape[:2]
    )

    # color_sweep is (num_outlier_grid_points, num_vertices, 3)
    # outlier_probability_sweep is (num_outlier_grid_points,)
    # color_outlier_probabilities_sweep is (num_outlier_grid_points, num_vertices)
    sampled_colors = color_sweep[color_sweep_indices, jnp.arange(color_sweep.shape[1])]
    sampled_color_outlier_probabilities = outlier_probability_sweep[
        color_outlier_sweep_indices
    ]

    log_q_color_and_color_outlier_probability = normalized_log_probabilities[
        sampled_indices, jnp.arange(normalized_log_probabilities.shape[1])
    ].sum()

    # log_q = estimate of q(all these colors, all these outliers ; inputs)
    # Only source of real randomness = sampling indices.  Captured in log_q_color_and_color_outlier_probability.
    # But we also want to be careful with the continuous values...
    # (1) outlier probs.  --> change the model to have discrete grid.  [Do later.]
    # (2) colors.  --> 1/q()
    #     uniform(old r, 2/3 oldr + 1/3 newr) 0  | uniform(0, 0.1)
    #     uniform(1/3, 2/3) # .5                 | uniform(.1, .9)
    #     uniform(2/3, 1) # 1                    | uniform(.9, 1)
    #
    # q(c1) * q(c2) * q(c3)
    # but we just output c2
    # q(the c values we output, marginalizing over the other choices)
    # -> just output q(c2)

    # We will treat this like the case where each sweep is uniform, so the q scores
    # are each (oldr - obsr)/3 * (oldg - obsg)/3 * (oldb - obsb)/3.
    q_prob_per_vertex = (
        1.0 / ((jnp.abs(current_colors - observed_colors) / 3) + 0.001)
    ).prod(-1)
    log_q_for_the_color_proposal = jnp.log(q_prob_per_vertex).sum()

    return (
        sampled_colors,
        sampled_color_outlier_probabilities,
        log_q_color_and_color_outlier_probability + log_q_for_the_color_proposal,
        scores_per_sweep_point_and_vertex,
    )


@jax.jit
def propose_depth_variance(trace, key, depth_variance_sweep):
    scores = jax.vmap(
        lambda variance: trace.update(key, C["depth_variance"].set(variance))[
            0
        ].get_score()
    )(depth_variance_sweep)
    scores = jax.nn.log_softmax(scores)
    sampled_index = jax.random.categorical(key, scores)
    sampled_depth_variance = depth_variance_sweep[sampled_index]
    log_q_depth_variance = scores[sampled_index]
    return sampled_depth_variance, log_q_depth_variance


@jax.jit
def propose_color_variance(trace, key, color_variance_sweep):
    scores = jax.vmap(
        lambda variance: trace.update(key, C["color_variance"].set(variance))[
            0
        ].get_score()
    )(color_variance_sweep)
    scores = jax.nn.log_softmax(scores)
    sampled_index = jax.random.categorical(key, scores)
    sampled_color_variance = color_variance_sweep[sampled_index]
    log_q_color_variance = scores[sampled_index]
    return sampled_color_variance, log_q_color_variance


@jax.jit
def propose_update(trace, key, pose):
    total_log_q = 0.0

    # Update pose
    # pose, log_q_pose = propose_pose(
    #     trace, key, pose_sample_variance, pose_sample_concentration
    # )
    trace = trace.update(key, C["pose"].set(pose))[0]
    # total_log_q += log_q_pose

    # Update depth outlier probability
    sampled_depth_outlier_probability, log_q_depth_outlier_probability, _ = (
        propose_depth_outlier_probability(trace, key)
    )
    trace = trace.update(
        key,
        make_depth_outlier_probabilities_choicemap(sampled_depth_outlier_probability),
    )[0]
    total_log_q += log_q_depth_outlier_probability

    # Update color and color outlier probability
    color_outlier_probability_sweep = jnp.array([0.01, 0.5, 1.0])
    colors, color_outlier_probability, log_q_color_color_outlier_probability, _ = (
        propose_color_and_color_outlier_probability(
            trace, key, color_outlier_probability_sweep
        )
    )
    trace = trace.update(
        key,
        make_colors_choicemap(colors)
        ^ make_color_outlier_probabilities_choicemap(color_outlier_probability),
    )[0]
    total_log_q += log_q_color_color_outlier_probability

    # # Update depth variance
    # depth_variance_sweep = jnp.array([0.0005, 0.001, 0.0025, 0.005, 0.01])
    # depth_variance, log_q_depth_variance = propose_depth_variance(
    #     trace, key, depth_variance_sweep
    # )
    # trace = trace.update(key, C["depth_variance"].set(depth_variance))[0]
    # total_log_q += log_q_depth_variance

    # # Update color variance
    # color_variance_sweep = jnp.array([0.005, 0.01, 0.05, 0.1, 0.2])
    # color_variance, log_q_color_variance = propose_color_variance(
    #     trace, key, color_variance_sweep
    # )
    # trace.update(key, C["color_variance"].set(color_variance))[0]
    # total_log_q += log_q_color_variance
    return trace, total_log_q


@jax.jit
def propose_update_get_score(trace, key, pose):
    new_trace, log_q = propose_update(trace, key, pose)
    # score is an estimate of P(data, pose | previous state)
    return new_trace.get_score() - log_q


propose_update_get_score_vmap = jax.jit(
    jax.vmap(propose_update_get_score, in_axes=(None, None, 0))
)


def inference_step_without_advance(trace, key):
    number = 20000

    var, conc = 0.02, 2000.0

    for _ in range(10):
        key = jax.random.split(key, 2)[-1]
        keys = jax.random.split(key, number)
        poses = Pose.concatenate_poses(
            [
                Pose.sample_gaussian_vmf_pose_vmap(
                    keys, trace.get_choices()["pose"], var, conc
                ),
                trace.get_choices()["pose"][None, ...],
            ]
        )
        pose_scores = Pose.logpdf_gaussian_vmf_pose_vmap(
            poses, trace.get_choices()["pose"], var, conc
        )
        scores = propose_update_get_score_vmap(trace, key, poses)
        scores = (
            scores - pose_scores
        )  # After this, scores are fair estimates of P(data | previous state)
        #                               and can be used to resample the choice sets.
        index = jax.random.categorical(key, scores)
        trace = propose_update(trace, key, poses[index])[0]

    return trace


def inference_step(trace, key, observed_rgbd):
    trace = advance_time(key, trace, observed_rgbd)
    trace = inference_step_without_advance(trace, key)
    return trace


# ## Old Inference moves ###
# @jax.jit
# def update_colors(trace):
#     key = jax.random.PRNGKey(0)
#     info = info_from_trace(trace)

#     color_delta = (
#         info["observed_rgbd_masked"][..., :3] - trace.get_choices()["colors", ...]
#     )
#     max_color_shift = trace.get_args()[0]["max_color_shift"]
#     color_delta_clipped = jnp.clip(color_delta, -max_color_shift, max_color_shift)

#     trace = trace.update(
#         key,
#         make_colors_choicemap(trace.get_choices()["colors", ...] + color_delta_clipped),
#     )[0]
#     return trace


# @jax.jit
# def score_with_give_pose_and_then_color_update(trace, pose, outlier_probability_sweep):
#     key = jax.random.PRNGKey(0)
#     trace = trace.update(key, C["pose"].set(pose))[0]

#     trace = update_colors(trace)

#     trace = grid_move_on_color_outlier_probability(trace, outlier_probability_sweep)
#     trace = grid_move_on_depth_outlier_probability(trace, outlier_probability_sweep)

#     return trace.get_score()


# @jax.jit
# def grid_move_on_depth_outlier_probability(trace, sweep):
#     current_setting = trace.get_choices()["depth_outlier_probabilities", ...]
#     potential_values = sweep[..., None] * jnp.ones_like(current_setting)
#     key = jax.random.PRNGKey(0)
#     get_vertex_scores_vmap = jax.vmap(
#         lambda x: info_from_trace(
#             trace.update(key, make_depth_outlier_probabilities_choicemap(x))[0]
#         )["scores"]
#     )
#     scores = get_vertex_scores_vmap(potential_values)
#     best_setting = sweep[jnp.argmax(scores, axis=0)]
#     return trace.update(key, make_depth_outlier_probabilities_choicemap(best_setting))[
#         0
#     ]


# @jax.jit
# def grid_move_on_color_outlier_probability(trace, sweep):
#     current_setting = trace.get_choices()["color_outlier_probabilities", ...]

#     potential_values = sweep[..., None] * jnp.ones_like(current_setting)
#     key = jax.random.PRNGKey(0)
#     get_vertex_scores_vmap = jax.vmap(
#         lambda x: info_from_trace(
#             trace.update(key, make_color_outlier_probabilities_choicemap(x))[0]
#         )["scores"]
#     )
#     scores = get_vertex_scores_vmap(potential_values)
#     best_setting = sweep[jnp.argmax(scores, axis=0)]
#     return trace.update(key, make_color_outlier_probabilities_choicemap(best_setting))[
#         0
#     ]


# @jax.jit
# def update_address_with_sweep(trace, address, sweep):
#     scores = b3d.enumerate_choices_get_scores(trace, address, sweep)
#     best_setting = sweep[jnp.argmax(scores)]
#     return b3d.update_choices(trace, address, best_setting)


# @partial(jax.jit, static_argnames=("number",))
# def gaussian_vmf_enumerative_move_with_other_updates(
#     trace, key, address, varianc, conc, number, outlier_probability_sweep
# ):
#     keys = jax.random.split(key, number)
#     poses = Pose.concatenate_poses(
#         [
#             Pose.sample_gaussian_vmf_pose_vmap(
#                 keys, trace.get_choices()["pose"], 0.02, 2000.0
#             ),
#             trace.get_choices()["pose"][None, ...],
#         ]
#     )

#     scores = jax.vmap(
#         score_with_give_pose_and_then_color_update, in_axes=(None, 0, None)
#     )(trace, poses, outlier_probability_sweep)

#     key = b3d.split_key(keys[-1])
#     sampled_pose = poses[scores.argmax()]
#     trace = trace.update(key, C["pose"].set(sampled_pose))[0]
#     return trace, key


# def inference_step_old(trace, key, observed_rgbd):
#     trace = advance_time(key, trace, observed_rgbd)

#     outlier_probability_sweep = jnp.array([0.01, 0.5, 1.0])
#     num_grid_points = 20000
#     for _ in range(2):
#         trace, key = gaussian_vmf_enumerative_move_with_other_updates(
#             trace,
#             key,
#             Pytree.const(("pose",)),
#             0.04,
#             200.0,
#             num_grid_points,
#             outlier_probability_sweep,
#         )
#         trace, key = gaussian_vmf_enumerative_move_with_other_updates(
#             trace,
#             key,
#             Pytree.const(("pose",)),
#             0.01,
#             500.0,
#             num_grid_points,
#             outlier_probability_sweep,
#         )
#         trace, key = gaussian_vmf_enumerative_move_with_other_updates(
#             trace,
#             key,
#             Pytree.const(("pose",)),
#             0.005,
#             1000.0,
#             num_grid_points,
#             outlier_probability_sweep,
#         )
#         trace, key = gaussian_vmf_enumerative_move_with_other_updates(
#             trace,
#             key,
#             Pytree.const(("pose",)),
#             0.001,
#             2000.0,
#             num_grid_points,
#             outlier_probability_sweep,
#         )

#     trace = grid_move_on_color_outlier_probability(trace, outlier_probability_sweep)
#     trace = grid_move_on_depth_outlier_probability(trace, outlier_probability_sweep)

#     trace = update_address_with_sweep(
#         trace,
#         Pytree.const(("color_variance",)),
#         jnp.array([0.005, 0.01, 0.05, 0.1, 0.2]),
#     )
#     trace = update_address_with_sweep(
#         trace,
#         Pytree.const(("depth_variance",)),
#         jnp.array([0.0005, 0.001, 0.0025, 0.005, 0.01]),
#     )

#     trace = update_colors(trace)

#     trace = grid_move_on_color_outlier_probability(trace, outlier_probability_sweep)
#     trace = grid_move_on_depth_outlier_probability(trace, outlier_probability_sweep)
#     return trace
