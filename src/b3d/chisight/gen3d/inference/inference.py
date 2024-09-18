from functools import partial, wraps

import jax
import jax.numpy as jnp
import jax.random
from genjax import ChoiceMapBuilder as C
from genjax import Diff, Pytree
from genjax import UpdateProblemBuilder as U
from jax.random import split
from tqdm import tqdm

import b3d
from b3d.chisight.gen3d.model import (
    dynamic_object_generative_model,
    get_hypers,
    get_new_state,
    make_colors_choicemap,
    make_depth_nonreturn_prob_choicemap,
    make_visibility_prob_choicemap,
)

from b3d.chisight.gen3d.hyperparams import InferenceHyperparams

from .toplevel_proposals import (
    get_pose_proposal_density,
    propose_other_latents_given_pose,
    propose_pose,
)

def batched_vmap(f, batch_size)
    test_poses_batches = test_poses.split(10)

#                jnp.array_split(self.pos, n), jnp.array_split(self.quat, n)

    scores = jnp.concatenate(
        [
            b3d.enumerate_choices_get_scores(trace, Pytree.const((address,)), poses)
            for poses in test_poses_batches
        ]
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
):
    if do_advance_time:
        key, subkey = split(key)
        trace = advance_time(subkey, trace, inference_hyperparams)

    k1, k2, k3 = split(key, 3)

    # Propose the poses
    pose_generation_keys = split(k1, inference_hyperparams.n_poses)
    proposed_poses, log_q_poses = jax.vmap(propose_pose, in_axes=(0, None, None))(
        pose_generation_keys, trace, inference_hyperparams
    )
    proposed_poses, log_q_poses = maybe_swap_in_gt_pose(
        proposed_poses, log_q_poses, use_gt_pose, gt_pose, inference_hyperparams
    )

    # Generate the remaining latents
    def propose_other_latents_given_pose_and_get_scores(
        key, proposed_pose, trace, inference_hyperparams
    ):
        proposed_trace, log_q, _, = propose_other_latents_given_pose(
            key, trace, proposed_pose, inference_hyperparams
        )
        return proposed_trace.get_score(), log_q

    param_generation_keys = split(k2, inference_hyperparams.n_poses)

    p_scores, log_q_nonpose_latents = jax.lax.map(
        lambda x: propose_other_latents_given_pose_and_get_scores(
            x[0], x[1], trace, inference_hyperparams
        )
    )((param_generation_keys, proposed_poses))

    # Scoring + resampling
    weights = jnp.where(
        inference_hyperparams.include_q_scores_at_top_level,
        p_scores - log_q_poses - log_q_nonpose_latents,
        p_scores,
    )

    chosen_index = jax.random.categorical(k3, weights)
    resampled_trace = jax.tree.map(lambda x: x[chosen_index], proposed_traces)
    weight = logmeanexp(weights)
    
    return (
        resampled_trace,
        all_keys,
        all_poses,
        all_weights,
        metadata
    )

def maybe_swap_in_gt_pose(proposed_poses, log_q_poses, use_gt_pose, gt_pose, inference_hyperparams)r:
    proposed_poses = jax.tree.map(
        lambda x, y: x.at[0].set(jnp.where(use_gt_pose, y, x[0])),
        proposed_poses,
        gt_pose,
    )

    log_q_poses = log_q_poses.at[0].set(
        jnp.where(
            use_gt_pose,
            get_pose_proposal_density(gt_pose, trace, inference_hyperparams),
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

def get_initial_trace(key, hyperparams, initial_state, initial_observed_rgbd):
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
    trace, _ = dynamic_object_generative_model.importance(
        key, choicemap, (hyperparams, initial_state)
    )
    return trace