import jax
import jax.numpy as jnp
import jax.random
from genjax import ChoiceMapBuilder as C
from genjax import Diff
from genjax import UpdateProblemBuilder as U
from jax.random import split

import b3d

from .model import (
    get_hypers,
    get_n_vertices,
    get_new_state,
    get_observed_rgbd,
    get_prev_state,
)
from .projection import PixelsPointsAssociation


def propose_pose(key, advanced_trace, inference_hyperparams):
    previous_pose = get_prev_state(advanced_trace)["pose"]
    hp = inference_hyperparams["pose_proposal"]
    pose = b3d.sample_gaussian_vmf_pose(key, previous_pose, hp["std"], hp["conc"])
    log_q = b3d.logpdf_gaussian_vmf_pose_vmap(
        pose, previous_pose, hp["std"], hp["conc"]
    )
    return pose, log_q


def propose_other_latents_given_pose(key, pose, advanced_trace, inference_hyperparams):
    k1, k2, k3, k4, k5, k6 = split(key, 6)

    trace_with_pose = update_field(k1, advanced_trace, "pose", pose)

    depth_nonreturn_probs, log_q_dnrps = propose_depth_nonreturn_probs(
        k2, trace_with_pose
    )
    colors, visibility_probs, log_q_cvp = propose_colors_and_visibility_probs(
        k3, trace_with_pose
    )
    depth_scale, log_q_ds = propose_depth_scale(k4, trace_with_pose)
    color_scale, log_q_cs = propose_color_scale(k5, trace_with_pose)

    proposed_trace = update_fields(
        k6,
        trace_with_pose,
        [
            "depth_nonreturn_probs",
            "colors",
            "visibility_probs",
            "depth_scale",
            "color_scale",
        ],
        [depth_nonreturn_probs, colors, visibility_probs, depth_scale, color_scale],
    )
    log_q = log_q_dnrps + log_q_cvp + log_q_ds + log_q_cs

    return proposed_trace, log_q


def propose_depth_nonreturn_probs(key, trace):
    observed_depths_per_points = PixelsPointsAssociation.from_hyperparams_and_pose(
        get_hypers(trace), get_new_state(trace)["pose"]
    ).get_point_depths(get_observed_rgbd(trace))

    depth_nonreturn_probs, per_vertex_log_qs = jax.vmap(
        propose_vertex_depth_nonreturn_prob, in_axes=(0, 0, 0, None, None, None)
    )(
        split(key, get_n_vertices(trace)),
        jnp.arange(get_n_vertices(trace)),
        observed_depths_per_points,
        get_prev_state(trace),
        get_new_state(trace),
        get_hypers(trace),
    )

    return depth_nonreturn_probs, per_vertex_log_qs.sum()


def propose_colors_and_visibility_probs(key, trace):
    observed_rgbds_per_points = PixelsPointsAssociation.from_hyperparams_and_pose(
        get_hypers(trace), get_new_state(trace)["pose"]
    ).get_point_rgbds(get_observed_rgbd(trace))

    colors, visibility_probs, per_vertex_log_qs = jax.vmap(
        propose_vertex_color_and_visibility_prob, in_axes=(0, 0, 0, None, None, None)
    )(
        split(key, get_n_vertices(trace)),
        jnp.arange(get_n_vertices(trace)),
        observed_rgbds_per_points,
        get_prev_state(trace),
        get_new_state(trace),
        get_hypers(trace),
    )

    return colors, visibility_probs, per_vertex_log_qs.sum()


def propose_vertex_depth_nonreturn_prob(
    key, vertex_index, observed_depth, previous_state, new_state, hyperparams
):
    previous_dnrp = previous_state["depth_nonreturn_prob"][vertex_index]
    visibility_prob = new_state["visibility_prob"][vertex_index]
    latent_depth = new_state["pose"].apply(hyperparams["vertices"][vertex_index])[2]

    def score_dnrp_value(dnrp_value):
        transition_score = hyperparams["depth_nonreturn_prob_kernel"].logpdf(
            dnrp_value, previous_dnrp
        )
        likelihood_score = hyperparams["depth_pixel_kernel"].logpdf(
            observed_depth, latent_depth, visibility_prob, dnrp_value
        )
        return transition_score + likelihood_score

    support = hyperparams["depth_nonreturn_prob_kernel"].support
    log_pscores = jax.vmap(score_dnrp_value)(support)
    log_normalized_scores = log_pscores - jax.scipy.special.logsumexp(log_pscores)
    index = jax.random.categorical(key, log_normalized_scores)

    return support[index], log_normalized_scores[index]


def propose_vertex_color_and_visibility_prob(
    key, vertex_index, observed_rgbd, previous_state, new_state, hyperparams
):
    # Placeholder
    return (
        previous_state["colors"][vertex_index],
        previous_state["visibility_prob"][vertex_index],
        0.0,
    )


def propose_depth_scale(key, trace):
    # Placeholder
    return get_prev_state(trace)["depth_scale"], 0.0


def propose_color_scale(key, trace):
    # Placeholder
    return get_prev_state(trace)["color_scale"], 0.0


### Utils ###
def update_field(key, trace, fieldname, value):
    return update_fields(key, trace, [fieldname], [value])


def update_fields(key, trace, fieldnames, values):
    hyperparams, previous_state = trace.get_args()
    trace, _, _, _ = trace.update(
        key,
        U.g(
            (Diff.no_change(hyperparams), Diff.unknown_change(previous_state)),
            C.kw(**dict(zip(fieldnames, values))),
        ),
    )
    return trace
