import jax
import jax.numpy as jnp
import jax.random
from genjax import ChoiceMapBuilder as C
from genjax import Diff
from genjax import UpdateProblemBuilder as U
from jax.random import split

from b3d import Pose
from b3d.modeling_utils import renormalized_color_laplace

from ..image_kernel import PixelsPointsAssociation
from ..model import (
    get_hypers,
    get_n_vertices,
    get_new_state,
    get_observed_rgbd,
    get_prev_state,
)
from .utils import update_field, update_fields, update_vmapped_fields, split, normalize_log_scores
from .point_attribute_proposals import propose_all_pointlevel_attributes

def propose_pose(key, advanced_trace, inference_hyperparams):
    """
    Propose a random pose near the previous timestep's pose.
    Returns (proposed_pose, log_proposal_density).
    """
    previous_pose = get_new_state(advanced_trace)["pose"]
    ih = inference_hyperparams
    pose = Pose.sample_gaussian_vmf_pose(
        key, previous_pose, ih.pose_proposal_std, ih.pose_proposal_conc
    )
    log_q = Pose.logpdf_gaussian_vmf_pose(
        pose, previous_pose, ih.pose_proposal_std, ih.pose_proposal_conc
    )
    return pose, log_q


def get_pose_proposal_density(pose, advanced_trace, inference_hyperparams):
    """
    Returns the log proposal density of the given pose, conditional upon the previous pose.
    """
    previous_pose = get_prev_state(advanced_trace)["pose"]
    ih = inference_hyperparams
    return Pose.logpdf_gaussian_vmf_pose(
        pose, previous_pose, ih.pose_proposal_std, ih.pose_proposal_conc
    )


def propose_other_latents_given_pose(key, advanced_trace, pose, inference_hyperparams):
    """
    Proposes all latents other than the pose, conditional upon the pose and observed RGBD
    in `advanced_trace`.
    Returns (proposed_trace, log_q) where `propose_trace` is the new trace with the
    proposed latents (and the same pose and observed rgbd as in the given trace).
    `log_q` is (a fair estimate of) the log proposal density.
    """
    k1, k2, k3, k4 = split(key, 4)

    trace = update_field(k1, advanced_trace, "pose", pose)

    k2a, k2b = split(k2)
    (
        colors,
        visibility_probs,
        depth_nonreturn_probs,
        log_q_point_attributes,
        point_proposal_metadata,
    ) = propose_all_pointlevel_attributes(k2a, trace, inference_hyperparams)
    trace = update_vmapped_fields(
        k2b,
        trace,
        ["colors", "visibility_prob", "depth_nonreturn_prob"],
        [colors, visibility_probs, depth_nonreturn_probs],
    )

    k3a, k3b = split(k3)
    depth_scale, log_q_ds = propose_depth_scale(k3a, trace)
    trace = update_field(k3b, trace, "depth_scale", depth_scale)

    k4a, k4b = split(k4)
    color_scale, log_q_cs = propose_color_scale(k4a, trace)
    trace = update_field(k4b, trace, "color_scale", color_scale)

    log_q = log_q_point_attributes + log_q_ds + log_q_cs
    return (
        trace,
        log_q,
        {
            "point_attribute_proposal_metadata": point_proposal_metadata,
            "log_q_point_attributes": log_q_point_attributes,
        },
    )


def propose_depth_scale(key, trace):
    """
    Propose a new global depth scale, conditioned upon the other values in `trace`.
    Returns (depth_scale, log_q) where `depth_scale` is the proposed value and
    `log_q` is (a fair estimate of) the log proposal density.
    """
    k1, k2 = split(key, 2)

    def score_depth_scale(k, depth_scale):
        newtr = update_field(k, trace, "depth_scale", depth_scale)
        return newtr.get_score()

    support = get_hypers(trace)["depth_scale_kernel"].support
    scores = jax.vmap(score_depth_scale, in_axes=(0, 0))(
        split(k1, len(support)), support
    )

    normalized_scores = normalize_log_scores(scores)
    index = jax.random.categorical(k2, normalized_scores)

    return support[index], normalized_scores[index]


def propose_color_scale(key, trace):
    """
    Propose a new global color scale, conditioned upon the other values in `trace`.
    Returns (color_scale, log_q) where `color_scale` is the proposed value and
    `log_q` is (a fair estimate of) the log proposal density.
    """
    k1, k2 = split(key, 2)

    def score_color_scale(k, color_scale):
        newtr = update_field(k, trace, "color_scale", color_scale)
        return newtr.get_score()

    support = get_hypers(trace)["color_scale_kernel"].support
    scores = jax.vmap(score_color_scale, in_axes=(0, 0))(
        split(k1, len(support)), support
    )

    normalized_scores = normalize_log_scores(scores)
    index = jax.random.categorical(k2, normalized_scores)

    return support[index], normalized_scores[index]

