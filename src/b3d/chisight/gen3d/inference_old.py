import jax
import jax.numpy as jnp
import jax.random
from genjax import ChoiceMapBuilder as C

import b3d
from b3d import Pose

from .model import (
    make_colors_choicemap,
    make_depth_nonreturn_prob_choicemap,
    make_visibility_prob_choicemap,
)


@jax.jit
def attribute_proposal_only_color_and_visibility(
    key,
    observed_rgbd_for_point,
    latent_rgbd_for_point,
    previous_color,
    previous_visibility_prob,
    previous_dnrp,
    color_scale,
    depth_scale,
    hyperparams,
    inference_hyperparams,
):
    image_kernel = hyperparams["image_kernel"]
    vertex_rgbd_kernel = image_kernel.get_rgbd_vertex_kernel()

    # color_outlier_probability_sweep is (k,) shape array
    depth_nonreturn_prob_kernel = hyperparams["depth_nonreturn_prob_kernel"]
    dnrp_values = depth_nonreturn_prob_kernel.support

    def likelihood_scorer(dnrp):
        return vertex_rgbd_kernel.logpdf(
            observed_rgbd_for_point,
            latent_rgbd_for_point,
            color_scale,
            depth_scale,
            previous_visibility_prob,
            dnrp,
            hyperparams["intrinsics"],
        )

    dnrp = dnrp_values[jnp.argmax(jax.vmap(likelihood_scorer)(dnrp_values))]

    # color_outlier_probability_sweep is (k,) shape array
    visibility_values = hyperparams["visibility_prob_kernel"].support
    visibility_prob_kernel = hyperparams["visibility_prob_kernel"]

    visbility_transition_scores = jax.vmap(
        visibility_prob_kernel.logpdf, in_axes=(0, None)
    )(visibility_values, previous_visibility_prob)

    color_interpolations_per_proposal = jnp.array([0.0, 0.5, 1.0])
    observed_color = observed_rgbd_for_point[:3]
    color_sweep = (
        color_interpolations_per_proposal[..., None] * observed_color
        + (1.0 - color_interpolations_per_proposal[..., None]) * previous_color
    )

    color_kernel = hyperparams["color_kernel"]
    color_transition_scores = jax.vmap(color_kernel.logpdf, in_axes=(0, None))(
        color_sweep, previous_color
    )

    def likelihood_scorer(color, visibility_prob):
        latent_rgbd_adjusted = latent_rgbd_for_point.at[:3].set(color)
        return vertex_rgbd_kernel.logpdf(
            observed_rgbd_for_point,
            latent_rgbd_adjusted,
            color_scale,
            depth_scale,
            visibility_prob,
            dnrp,
            hyperparams["intrinsics"],
        )

    vmap_version = jax.vmap(
        jax.vmap(
            likelihood_scorer,
            in_axes=(None, 0),
        ),
        in_axes=(0, None),
    )

    likelihood_scores_per_sweep_point_and_vertex = vmap_version(
        color_sweep, visibility_values
    )

    scores_color_and_visibility = (
        likelihood_scores_per_sweep_point_and_vertex  # (num_color_grid_points, num_outlier_grid_points)
        + color_transition_scores[:, None, ...]
        + visbility_transition_scores[None, ...]
    )  # (num_color_grid_points, num_outlier_grid_points, num_vertices)

    idx_color, idx_visibility = jnp.unravel_index(
        jnp.argmax(scores_color_and_visibility.reshape(-1)),
        scores_color_and_visibility.shape,
    )
    return {
        "colors": color_sweep[idx_color],
        "visibility_prob": visibility_values[idx_visibility],
        "depth_nonreturn_prob": dnrp,
        "scores": scores_color_and_visibility,
    }


@jax.jit
def update_vertex_attributes(key, trace, inference_hyperparams):
    hyperparams, previous_state = trace.get_args()

    latent_rgbd_per_point, observed_rgbd_per_point = (
        b3d.chisight.gen3d.image_kernel.get_latent_and_observed_correspondences(
            trace.get_retval()["new_state"],
            trace.get_args()[0],
            trace.get_choices()["rgbd"],
        )
    )

    previous_state = trace.get_args()[1]
    previous_color = previous_state["colors"]
    previous_visibility_prob = previous_state["visibility_prob"]
    previous_dnrp = previous_state["depth_nonreturn_prob"]
    color_scale = previous_state["color_scale"]
    depth_scale = previous_state["depth_scale"]

    keys = jax.random.split(key, len(observed_rgbd_per_point))

    sample = jax.vmap(
        attribute_proposal_only_color_and_visibility,
        in_axes=(0, 0, 0, 0, 0, 0, None, None, None, None),
    )(
        keys,
        observed_rgbd_per_point,
        latent_rgbd_per_point,
        previous_color,
        previous_visibility_prob,
        previous_dnrp,
        color_scale,
        depth_scale,
        hyperparams,
        inference_hyperparams,
    )
    trace = trace.update(
        key,
        make_colors_choicemap(sample["colors"])
        ^ make_visibility_prob_choicemap(sample["visibility_prob"])
        ^ make_depth_nonreturn_prob_choicemap(sample["depth_nonreturn_prob"]),
    )[0]
    return trace, {}


def update_all(key, trace, pose, inference_hyperparams):
    trace = trace.update(key, C["pose"].set(pose))[0]
    trace, _ = update_vertex_attributes(key, trace, inference_hyperparams)
    return trace


def update_all_get_score(key, trace, pose, inference_hyperparams):
    trace = update_all(key, trace, pose, inference_hyperparams)
    return trace.get_score()


update_all_get_score_vmap = jax.jit(
    jax.vmap(update_all_get_score, in_axes=(0, None, 0, None))
)


def inference_step(trace, key, inference_hyperparams):
    number = 20000
    current_pose = trace.get_choices()["pose"]
    var_conc = [(0.04, 1000.0), (0.02, 1500.0), (0.005, 2000.0)]
    for var, conc in var_conc:
        key = jax.random.split(key, 2)[-1]
        keys = jax.random.split(key, number)
        poses = Pose.concatenate_poses(
            [
                Pose.sample_gaussian_vmf_pose_vmap(keys[:-1], current_pose, var, conc),
                current_pose[None, ...],
            ]
        )
        pose_scores = Pose.logpdf_gaussian_vmf_pose_vmap(
            poses, trace.get_choices()["pose"], var, conc
        )
        scores = update_all_get_score_vmap(keys, trace, poses, inference_hyperparams)
        scores_pose_q_correction = (
            scores - pose_scores
        )  # After this, scores are fair estimates of P(data | previous state)
        #                               and can be used to resample the choice sets.
        current_pose = poses[jnp.argmax(scores)]
    trace = update_all(key, trace, current_pose, inference_hyperparams)
    return trace, scores, scores_pose_q_correction
