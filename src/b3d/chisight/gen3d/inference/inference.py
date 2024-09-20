import jax
import jax.numpy as jnp
import jax.random
from genjax import ChoiceMapBuilder as C
from genjax import Diff
from genjax import UpdateProblemBuilder as U
from jax.random import split

import b3d
from b3d import Pose
from b3d.chisight.gen3d.hyperparams import InferenceHyperparams
from b3d.chisight.gen3d.model import (
    dynamic_object_generative_model,
    get_hypers,
    get_new_state,
    get_prev_state,
    make_colors_choicemap,
    make_depth_nonreturn_prob_choicemap,
    make_visibility_prob_choicemap,
)

from .point_attribute_proposals import propose_all_pointlevel_attributes
from .utils import logmeanexp, normalize_log_scores, update_field, update_vmapped_fields


@jax.jit
def c2f_step(
    key,
    trace,
    pose_proposal_args,
    inference_hyperparams,
    use_gt_pose,
    gt_pose=b3d.Pose.identity(),
):
    k1, k2, k3 = split(key, 3)

    # Propose the poses
    pose_generation_keys = split(k1, inference_hyperparams.n_poses)
    proposed_poses, log_q_poses = jax.vmap(propose_pose, in_axes=(0, None, None))(
        pose_generation_keys, trace, pose_proposal_args
    )
    proposed_poses, log_q_poses = maybe_swap_in_gt_pose(
        proposed_poses, log_q_poses, trace, use_gt_pose, gt_pose, pose_proposal_args
    )

    # Generate the remaining latents to get pose scores
    def propose_other_latents_given_pose_and_get_scores(
        key, proposed_pose, trace, inference_hyperparams
    ):
        proposed_trace, log_q, _ = propose_other_latents_given_pose(
            key, trace, proposed_pose, inference_hyperparams
        )
        return proposed_trace.get_score(), log_q

    param_generation_keys = split(k2, inference_hyperparams.n_poses)
    p_scores, log_q_nonpose_latents = jax.lax.map(
        lambda x: propose_other_latents_given_pose_and_get_scores(
            x[0], x[1], trace, inference_hyperparams
        ),
        (param_generation_keys, proposed_poses),
    )

    # Scoring + resampling
    weights = jnp.where(
        inference_hyperparams.include_q_scores_at_top_level,
        p_scores - log_q_poses - log_q_nonpose_latents,
        p_scores,
    )

    chosen_index = jax.random.categorical(k3, weights)
    resampled_trace, _, _ = propose_other_latents_given_pose(
        param_generation_keys[chosen_index],
        trace,
        proposed_poses[chosen_index],
        inference_hyperparams,
    )
    return (
        resampled_trace,
        logmeanexp(weights),
        param_generation_keys,
        proposed_poses,
        weights,
    )


def inference_step(
    key,
    trace,
    observed_rgbd,
    inference_hyperparams: InferenceHyperparams,
    *,
    gt_pose=b3d.Pose.identity(),
    use_gt_pose=False,
    do_advance_time=True,
    get_all_metadata=False,
    get_all_weights=False,
):
    if do_advance_time:
        key, subkey = split(key)
        trace = advance_time(subkey, trace, observed_rgbd)

    for pose_proposal_args in inference_hyperparams.pose_proposal_args:
        key, subkey = split(key)
        trace, weight, keys_to_regenerate_traces, all_poses, all_weights = c2f_step(
            subkey,
            trace,
            pose_proposal_args,
            inference_hyperparams,
            use_gt_pose,
            gt_pose=gt_pose,
        )

    if get_all_metadata:
        metadata = {}  # TODO: someone add in getting thee metadata you need
        # Please make it relatively clean!
        return (
            trace,
            weight,
            all_weights,
            all_poses,
            keys_to_regenerate_traces,
            metadata,
        )
    elif get_all_weights:
        return (trace, weight, all_weights, all_poses, keys_to_regenerate_traces)
    else:
        return (trace, weight)


def get_trace_generated_during_inference(key, trace, pose, inference_hyperparams):
    return propose_other_latents_given_pose(key, trace, pose, inference_hyperparams)[0]


def maybe_swap_in_gt_pose(
    proposed_poses, log_q_poses, trace, use_gt_pose, gt_pose, pose_proposal_args
):
    proposed_poses = jax.tree.map(
        lambda x, y: x.at[0].set(jnp.where(use_gt_pose, y, x[0])),
        proposed_poses,
        gt_pose,
    )

    log_q_poses = log_q_poses.at[0].set(
        jnp.where(
            use_gt_pose,
            get_pose_proposal_density(gt_pose, trace, pose_proposal_args),
            log_q_poses[0],
        )
    )

    return proposed_poses, log_q_poses


@jax.jit
def advance_time(key, trace, observed_rgbd):
    """
    Advance to the next timestep, setting the new latent state to the
    same thing as the previous latent state, and setting the new
    observed RGBD value.

    Returns a trace where previous_state (stored in the arguments)
    and new_state (sampled in the choices and returned) are identical.
    """
    trace, _, _, _ = trace.update(
        key,
        U.g(
            (
                Diff.no_change(get_hypers(trace)),
                Diff.unknown_change(get_new_state(trace)),
            ),
            C.kw(rgbd=observed_rgbd),
        ),
    )
    return trace


def get_initial_trace(
    key, hyperparams, initial_state, initial_observed_rgbd, get_weight=False
):
    """
    Get the initial trace, given the initial state.
    The previous state and current state in the trace will be `initial_state`.
    """
    choicemap = (
        C.d(
            {
                "pose": initial_state["pose"],
                "color_scale": initial_state["color_scale"],
                "depth_scale": initial_state["depth_scale"],
                "rgbd": initial_observed_rgbd,
            }
        )
        ^ make_visibility_prob_choicemap(initial_state["visibility_prob"])
        ^ make_colors_choicemap(initial_state["colors"])
        ^ make_depth_nonreturn_prob_choicemap(initial_state["depth_nonreturn_prob"])
    )
    trace, weight = dynamic_object_generative_model.importance(
        key, choicemap, (hyperparams, initial_state)
    )
    if get_weight:
        return trace, weight
    else:
        return trace


### Inference moves ###


def propose_pose(key, advanced_trace, args):
    """
    Propose a random pose near the previous timestep's pose.
    Returns (proposed_pose, log_proposal_density).
    """
    std, conc = args
    previous_pose = get_new_state(advanced_trace)["pose"]
    pose = Pose.sample_gaussian_vmf_pose(key, previous_pose, std, conc)
    log_q = Pose.logpdf_gaussian_vmf_pose(pose, previous_pose, std, conc)
    return pose, log_q


def get_pose_proposal_density(pose, advanced_trace, args):
    """
    Returns the log proposal density of the given pose, conditional upon the previous pose.
    """
    std, conc = args
    previous_pose = get_prev_state(advanced_trace)["pose"]
    return Pose.logpdf_gaussian_vmf_pose(pose, previous_pose, std, conc)


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
