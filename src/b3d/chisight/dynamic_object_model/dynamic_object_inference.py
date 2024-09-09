# import jax
# import jax.numpy as jnp
# import jax.random
# from genjax import ChoiceMapBuilder as C
# from genjax import Diff
# from genjax import UpdateProblemBuilder as U

# from b3d import Pose

# from .dynamic_object_model import (
#     info_from_trace,
#     make_colors_choicemap,
# )


# @jax.jit
# def advance_time(key, trace, observed_rgbd):
#     """
#     Advance to the next timestep, setting the new latent state to the
#     same thing as the previous latent state, and setting the new
#     observed RGBD value.

#     Returns a trace where previous_state (stored in the arguments)
#     and new_state (sampled in the choices and returned) are identical.
#     """
#     hyperparams, _ = trace.get_args()
#     previous_state = trace.get_retval()["new_state"]
#     trace, _, _, _ = trace.update(
#         key,
#         U.g(
#             (Diff.no_change(hyperparams), Diff.unknown_change(previous_state)),
#             C.kw(rgbd=observed_rgbd),
#         ),
#     )
#     return trace


# #### New Inference Programs


# @jax.jit
# def propose_color_and_visibility_probability(trace, key, outlier_probability_sweep):
#     # color_outlier_probability_sweep is (k,) shape array

#     hyperparams, previous_state = trace.get_args()
#     previous_visibility = previous_state["visibility"]
#     current_colors = trace.get_args()[1]["colors"]

#     # num_vertices = current_colors.shape[0]
#     # num_outlier_grid_points = outlier_probability_sweep.shape[0]
#     color_outlier_probabilities_sweep = (
#         outlier_probability_sweep[..., None]  # (num_outlier_grid_points, 1)
#         * jnp.ones_like(current_is_visible_probabilities)  # (num_vertices,)
#     )  # (num_outlier_grid_points, num_vertices)

#     # We will grid over color values, using a grid that mixes the old and observed
#     # colors in a set of exact proportions.
#     # We regard these as coming from uniform proposals where we sample the RGB
#     # values uniformly between the mixed R, G, and B values with mixtures between
#     # [0., .125], [.125, .5], [.5, .875], [.875, 1.].
#     # So the q scores will be .125^3, .375^3, .375^3, .125^3.
#     # TODO: we really ought to add a small amount of proposal probability mass
#     # onto the points at the end, to capture the fact that the posterior could allow
#     # colors outside the considered interpolation window.
#     color_interpolations_per_proposal = jnp.array([0.0, 0.5, 1.0])
#     # num_color_grid_points = len(color_interpolations_per_proposal)

#     observed_colors = info_from_trace(trace)["observed_rgbd_masked"][
#         ..., :3
#     ]  # (num_vertices, 3)
#     color_sweep = observed_colors[None, ...] * color_interpolations_per_proposal[
#         :, None, None
#     ] + current_colors[None, ...] * (
#         1 - color_interpolations_per_proposal[:, None, None]
#     )  # (num_color_grid_points, num_vertices, 3)

#     # Function takes in color and color outlier probabilities array of shapes (num_vertices,3) and (num_vertices,) respectively
#     # and gives scores for each vertex (num_vertices,)
#     def get_per_vertex_likelihoods_with_new_color_and_color_outlier_probabilities(
#         colors, color_outlier_probabilities
#     ):
#         return info_from_trace(
#             trace.update(
#                 key,
#                 make_is_visible_probabilities_choicemap(color_outlier_probabilities)
#                 ^ make_colors_choicemap(colors),
#             )[0]
#         )["scores"]

#     vmap_version = jax.vmap(
#         jax.vmap(
#             get_per_vertex_likelihoods_with_new_color_and_color_outlier_probabilities,
#             in_axes=(None, 0),
#         ),
#         in_axes=(0, None),
#     )

#     # Vmap over the depth_outlier_probability_sweep_full array to get scores for each vertex for each depth_outlier_probability in the sweep
#     likelihood_scores_per_sweep_point_and_vertex = vmap_version(
#         color_sweep, color_outlier_probabilities_sweep
#     )  # (num_color_grid_points, num_outlier_grid_points, num_vertices)

#     # Color outlier transition scores
#     color_outlier_transition_scores_per_sweep_point_and_vertex = vectorized_outlier_probability_transition_kernel_logpdf(
#         color_outlier_probabilities_sweep,  # (num_outlier_grid_points, num_vertices)
#         current_is_visible_probabilities,
#         trace.get_args()[0]["is_visible_transition_scale"],
#     )  # (num_outlier_grid_points, num_vertices)

#     color_transition_scores_per_sweep_point_and_vertex = (
#         vectorized_color_transition_kernel_logpdf(
#             color_sweep, current_colors, trace.get_args()[0]["color_shift_scale"]
#         )
#     )  # (num_color_grid_points, num_vertices)

#     scores_per_sweep_point_and_vertex = (
#         likelihood_scores_per_sweep_point_and_vertex  # (num_color_grid_points, num_outlier_grid_points, num_vertices)
#         + color_outlier_transition_scores_per_sweep_point_and_vertex[None, ...]
#         + color_transition_scores_per_sweep_point_and_vertex[:, None, ...]
#     )  # (num_color_grid_points, num_outlier_grid_points, num_vertices)

#     unraveled_scores = scores_per_sweep_point_and_vertex.reshape(
#         -1, scores_per_sweep_point_and_vertex.shape[-1]
#     )
#     normalized_log_probabilities = jax.nn.log_softmax(unraveled_scores, axis=0)
#     sampled_indices = jax.random.categorical(key, normalized_log_probabilities, axis=0)

#     color_sweep_indices, color_outlier_sweep_indices = jnp.unravel_index(
#         sampled_indices, scores_per_sweep_point_and_vertex.shape[:2]
#     )

#     # color_sweep is (num_outlier_grid_points, num_vertices, 3)
#     # outlier_probability_sweep is (num_outlier_grid_points,)
#     # color_outlier_probabilities_sweep is (num_outlier_grid_points, num_vertices)
#     sampled_colors = color_sweep[color_sweep_indices, jnp.arange(color_sweep.shape[1])]
#     sampled_color_outlier_probabilities = outlier_probability_sweep[
#         color_outlier_sweep_indices
#     ]

#     log_q_color_and_color_outlier_probability = normalized_log_probabilities[
#         sampled_indices, jnp.arange(normalized_log_probabilities.shape[1])
#     ].sum()

#     # log_q = estimate of q(all these colors, all these outliers ; inputs)
#     # Only source of real randomness = sampling indices.  Captured in log_q_color_and_color_outlier_probability.
#     # But we also want to be careful with the continuous values...
#     # (1) outlier probs.  --> change the model to have discrete grid.  [Do later.]
#     # (2) colors.  --> 1/q()
#     #     uniform(old r, 2/3 oldr + 1/3 newr) 0  | uniform(0, 0.1)
#     #     uniform(1/3, 2/3) # .5                 | uniform(.1, .9)
#     #     uniform(2/3, 1) # 1                    | uniform(.9, 1)
#     #
#     # q(c1) * q(c2) * q(c3)
#     # but we just output c2
#     # q(the c values we output, marginalizing over the other choices)
#     # -> just output q(c2)

#     # We will treat this like the case where each sweep is uniform, so the q scores
#     # are each (oldr - obsr)/3 * (oldg - obsg)/3 * (oldb - obsb)/3.

#     hyperparams = trace.get_args()[0]
#     color_shift_scale = hyperparams["color_shift_scale"]
#     color_scale = trace.get_choices()["color_scale"]

#     d = 1 / (1 / color_shift_scale + 1 / color_scale)

#     q_prob_per_vertex = (
#         1.0 / ((jnp.abs(current_colors - observed_colors) / 3) + 4 * d)
#     ).prod(-1)
#     log_q_for_the_color_proposal = jnp.log(q_prob_per_vertex).sum()

#     return (
#         sampled_colors,
#         sampled_color_outlier_probabilities,
#         log_q_color_and_color_outlier_probability + log_q_for_the_color_proposal,
#         scores_per_sweep_point_and_vertex,
#     )


# @jax.jit
# def propose_update(trace, key, pose):
#     total_log_q = 0.0

#     # Update pose
#     # pose, log_q_pose = propose_pose(
#     #     trace, key, pose_sample_variance, pose_sample_concentration
#     # )
#     trace = trace.update(key, C["pose"].set(pose))[0]

#     # Update color and color outlier probability
#     color_outlier_probability_sweep = jnp.array([0.01, 0.5, 1.0])
#     colors, color_outlier_probability, log_q_color_color_outlier_probability, _ = (
#         propose_color_and_color_outlier_probability(
#             trace, key, color_outlier_probability_sweep
#         )
#     )
#     trace = trace.update(
#         key,
#         make_colors_choicemap(colors)
#         ^ make_color_outlier_probabilities_choicemap(color_outlier_probability),
#     )[0]
#     total_log_q += log_q_color_color_outlier_probability

#     # # Update depth variance
#     # depth_variance_sweep = jnp.array([0.0005, 0.001, 0.0025, 0.005, 0.01])
#     # depth_variance, log_q_depth_variance = propose_depth_variance(
#     #     trace, key, depth_variance_sweep
#     # )
#     # trace = trace.update(key, C["depth_variance"].set(depth_variance))[0]
#     # total_log_q += log_q_depth_variance

#     # # Update color variance
#     # color_variance_sweep = jnp.array([0.005, 0.01, 0.05, 0.1, 0.2])
#     # color_variance, log_q_color_variance = propose_color_variance(
#     #     trace, key, color_variance_sweep
#     # )
#     # trace.update(key, C["color_variance"].set(color_variance))[0]
#     # total_log_q += log_q_color_variance
#     return trace, total_log_q


# @jax.jit
# def propose_update_get_score(trace, key, pose):
#     new_trace, log_q = propose_update(trace, key, pose)
#     # score is an estimate of P(data, pose | previous state)
#     return new_trace.get_score() - log_q


# propose_update_get_score_vmap = jax.jit(
#     jax.vmap(propose_update_get_score, in_axes=(None, None, 0))
# )


# def inference_step_without_advance(trace, key):
#     number = 20000

#     var, conc = 0.02, 2000.0

#     for _ in range(10):
#         key = jax.random.split(key, 2)[-1]
#         keys = jax.random.split(key, number)
#         poses = Pose.concatenate_poses(
#             [
#                 Pose.sample_gaussian_vmf_pose_vmap(
#                     keys, trace.get_choices()["pose"], var, conc
#                 ),
#                 trace.get_choices()["pose"][None, ...],
#             ]
#         )
#         pose_scores = Pose.logpdf_gaussian_vmf_pose_vmap(
#             poses, trace.get_choices()["pose"], var, conc
#         )
#         scores = propose_update_get_score_vmap(trace, key, poses)
#         scores = (
#             scores - pose_scores
#         )  # After this, scores are fair estimates of P(data | previous state)
#         #                               and can be used to resample the choice sets.
#         index = jax.random.categorical(key, scores)
#         trace = propose_update(trace, key, poses[index])[0]

#     return trace


# def inference_step(trace, key, observed_rgbd):
#     trace = advance_time(key, trace, observed_rgbd)
#     trace = inference_step_without_advance(trace, key)
#     return trace
