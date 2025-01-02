from functools import partial

import jax
import jax.numpy as jnp
import jax.random
from genjax import ChoiceMapBuilder as C
from genjax import Diff
from genjax import UpdateProblemBuilder as U
from jax.random import split

import b3d
from b3d import Pose
from b3d.chisight.dense.dense_model import (
    get_hypers,
    get_new_state,
    get_prev_state,
)
from b3d.chisight.gen3d.hyperparams import InferenceHyperparams

from .utils import logmeanexp, update_field


@partial(jax.jit, static_argnames=("do_advance_time"))
def inference_step(
    key,
    trace,
    observed_rgbd,
    inference_hyperparams: InferenceHyperparams,
    addresses,
    posterior_across_frames,
    do_advance_time=True,
    include_previous_pose=True,
    sample=False,
    k=50,
):
    if do_advance_time:
        key, subkey = split(key)
        trace = advance_time(subkey, trace, observed_rgbd)

    @jax.jit
    def c2f_step(
        key,
        trace,
        pose_proposal_args,
        address,
    ):
        addr = address.unwrap()
        k1, k2, k3 = split(key, 3)

        # Propose the poses
        pose_generation_keys = split(k1, inference_hyperparams.n_poses)
        proposed_poses, log_q_poses = jax.vmap(
            propose_pose, in_axes=(0, None, None, None)
        )(pose_generation_keys, trace, addr, pose_proposal_args)
        proposed_poses, log_q_poses = maybe_swap_in_previous_pose(
            proposed_poses, log_q_poses, trace, addr, include_previous_pose, pose_proposal_args
        )

        def update_and_get_scores(key, proposed_pose, trace, addr):
            key, subkey = split(key)
            updated_trace = update_field(subkey, trace, addr, proposed_pose)
            return updated_trace, updated_trace.get_score()

        param_generation_keys = split(k2, inference_hyperparams.n_poses)
        # _, p_scores = jax.lax.map(
        #     lambda x: update_and_get_scores(x[0], x[1], trace, addr),
        #     (param_generation_keys, proposed_poses),
        # )
        _, p_scores = jax.vmap(update_and_get_scores, in_axes=(0, 0, None, None))(
            param_generation_keys, proposed_poses, trace, addr
        )

        # Scoring + resampling
        weights = jnp.where(
            inference_hyperparams.include_q_scores_at_top_level,
            p_scores - log_q_poses,
            p_scores,
        )

        if sample:
            chosen_index = jax.random.categorical(k3, weights)
        else:
            chosen_index = weights.argmax()
        resampled_trace, _ = update_and_get_scores(
            param_generation_keys[chosen_index],
            proposed_poses[chosen_index],
            trace,
            addr,
        )
        return (
            resampled_trace,
            logmeanexp(weights),
            param_generation_keys,
            proposed_poses,
            weights,
        )

    this_frame_posterior = {}
    for addr in addresses:
        for pose_proposal_args in inference_hyperparams.pose_proposal_args:
            key, subkey = split(key)
            trace, _, _, proposed_poses, weights = c2f_step(
                subkey,
                trace,
                pose_proposal_args,
                addr,
            )
        top_k_indices = jnp.argsort(weights)[-k:][::-1]
        top_scores = [weights[idx] for idx in top_k_indices]
        posterior_poses = [proposed_poses[idx] for idx in top_k_indices]
        this_frame_posterior[int(addr.unwrap().split('_')[-1])] = ([(score, posterior_pose) for (posterior_pose, score) in zip(posterior_poses, top_scores)], trace.get_choices()[addr.unwrap()],
        )
    posterior_across_frames["pose"].append(this_frame_posterior)

    return (trace, posterior_across_frames)


def maybe_swap_in_previous_pose(
    proposed_poses, log_q_poses, trace, addr, include_previous_pose, pose_proposal_args
):
    previous_pose, log_q = assess_previous_pose(trace, addr, pose_proposal_args)
    proposed_poses = jax.tree.map(
        lambda x, y: x.at[0].set(jnp.where(include_previous_pose, y, x[0])),
        proposed_poses,
        previous_pose,
    )

    log_q_poses = log_q_poses.at[0].set(
        jnp.where(
            include_previous_pose,
            log_q,
            log_q_poses[0],
        )
    )

    return proposed_poses, log_q_poses


def assess_previous_pose(advanced_trace, addr, args):
    """
    Returns the log proposal density of the given pose, conditional upon the previous pose.
    """
    std, conc = args
    previous_pose = get_prev_state(advanced_trace)[addr]
    log_q = Pose.logpdf_gaussian_vmf_pose(previous_pose, previous_pose, std, conc)
    return previous_pose, log_q



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
    key,
    renderer,
    likelihood_func,
    hyperparams,
    initial_state,
    initial_observed_rgbd,
    get_weight=False,
):
    """
    Get the initial trace, given the initial state.
    The previous state and current state in the trace will be `initial_state`.
    """
    choicemap = C.d(
        {
            "camera_pose": hyperparams["camera_pose"],
            "color_noise_variance": hyperparams["color_noise_variance"],
            "depth_noise_variance": hyperparams["color_noise_variance"],
            "outlier_probability": hyperparams["outlier_probability"],
            "rgbd": initial_observed_rgbd,
        }
        | initial_state
    )
    b3d.reload(b3d.chisight.dense.dense_model)
    dynamic_object_generative_model, _ = (
        b3d.chisight.dense.dense_model.make_dense_multiobject_dynamics_model(
            renderer, likelihood_func
        )
    )
    trace, weight = dynamic_object_generative_model.importance(
        key, choicemap, (hyperparams, initial_state | {'t': -1})
    )
    if get_weight:
        return trace, weight
    else:
        return trace


### Inference moves ###


def propose_pose(key, advanced_trace, addr, args):
    """
    Propose a random pose near the previous timestep's pose.
    Returns (proposed_pose, log_proposal_density).
    """
    std, conc = args
    previous_pose = get_new_state(advanced_trace)[addr]
    pose = Pose.sample_gaussian_vmf_pose(key, previous_pose, std, conc)
    log_q = Pose.logpdf_gaussian_vmf_pose(pose, previous_pose, std, conc)
    return pose, log_q
