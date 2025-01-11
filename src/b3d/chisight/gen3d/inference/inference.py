import time
import itertools
# import jax.scipy.stats as ss

import jax
import jax.numpy as jnp
import jax.random
from genjax import ChoiceMapBuilder as C
from genjax import Diff
from genjax import UpdateProblemBuilder as U
from jax.random import split

from b3d import Pose
from b3d.chisight.dense.dense_model import (
    get_hypers,
    get_new_state,
    get_prev_state,
)
from b3d.chisight.gen3d.hyperparams import InferenceHyperparams

from .utils import logmeanexp, update_field


@jax.jit
def inference_step(
    key,
    trace,
    observed_rgbd,
    inference_hyperparams: InferenceHyperparams,
    addresses,
    include_previous_pose=True,
    k=50,
):
    key, subkey = split(key)
    trace = advance_time(subkey, trace, observed_rgbd)

    @jax.jit
    def c2f_step(
        key,
        trace,
        pose_proposal_args,
        addr,
    ):
        addr = addr.unwrap()
        k1, k2, k3 = split(key, 3)

        # Propose the poses
        pose_generation_keys = split(k1, inference_hyperparams.n_poses)
        proposed_poses, log_q_poses = jax.vmap(
            propose_pose, in_axes=(0, None, None, None)
        )(pose_generation_keys, trace, addr, pose_proposal_args)
        # jax.debug.print("rank before: {v}", v=ss.rankdata(log_q_poses))
        # jax.debug.print("score before: {v}", v=log_q_poses)
        
        proposed_poses, log_q_poses = maybe_swap_in_previous_pose(
            proposed_poses, log_q_poses, trace, addr, include_previous_pose, pose_proposal_args
        )
        # jax.debug.print("rank after: {v}", v=ss.rankdata(log_q_poses))
        # jax.debug.print("score after: {v}", v=log_q_poses)

        def update_and_get_scores(key, proposed_pose, trace, addr):
            key, subkey = split(key)
            updated_trace = update_field(subkey, trace, addr, proposed_pose)
            return updated_trace, updated_trace.get_score()

        param_generation_keys = split(k2, inference_hyperparams.n_poses)
        _, p_scores = jax.vmap(update_and_get_scores, in_axes=(0, 0, None, None))(
            param_generation_keys, proposed_poses, trace, addr
        )

        # Scoring + resampling
        weights = jnp.where(
            inference_hyperparams.include_q_scores_at_top_level,
            p_scores - log_q_poses,
            p_scores,
        )

        # chosen_index = jax.random.categorical(k3, weights)
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
            proposed_poses[chosen_index],
            proposed_poses,
            weights,
        )

    this_frame_posterior = {}
    for i, (addr, pose_proposal_args) in enumerate(itertools.product(addresses, inference_hyperparams.pose_proposal_args)):
        key, subkey = split(key)
        this_iteration_start_time = time.time()
        trace, _, best_pose, proposed_poses, weights = c2f_step(
            subkey,
            trace,
            pose_proposal_args,
            addr,
        )
        if i%len(inference_hyperparams.pose_proposal_args) == 0:
            top_k_indices = jnp.argsort(weights)[-k:][::-1]
            top_scores = [weights[idx] for idx in top_k_indices]
            posterior_poses = [proposed_poses[idx] for idx in top_k_indices]
            this_frame_posterior[int(addr.unwrap().split('_')[-1])] = [[(score, posterior_pose) for (posterior_pose, score) in zip(posterior_poses, top_scores)]]
        elif (i+1)%len(inference_hyperparams.pose_proposal_args) == 0:
            this_frame_posterior[int(addr.unwrap().split('_')[-1])].append(best_pose)
        this_iteration_end_time = time.time()
        print(f"\t\t\t c_2_f step time: {this_iteration_end_time - this_iteration_start_time}")
    
    return (trace, this_frame_posterior)


def maybe_swap_in_previous_pose(
    proposed_poses, log_q_poses, trace, addr, include_previous_pose, pose_proposal_args
):
    previous_pose = get_prev_state(trace)[addr]
    log_q = assess_previous_pose(trace, addr, previous_pose, pose_proposal_args)
    chosen_index = log_q_poses.argmin()
    proposed_poses = jax.tree.map(
        lambda x, y: x.at[chosen_index].set(jnp.where(include_previous_pose, y, x[chosen_index])),
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


def assess_previous_pose(advanced_trace, addr, previous_pose, args):
    """
    Returns the log proposal density of the given pose, conditional upon the previous pose.
    """
    std, conc = args
    new_pose = get_new_state(advanced_trace)[addr]
    log_q = Pose.logpdf_gaussian_vmf_pose_approx(previous_pose, new_pose, std, conc)
    return log_q



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
    importance_jit,
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
    
    trace, weight = importance_jit(
        key, choicemap, (hyperparams, dict([(k, v) for k, v in initial_state.items() if not k.startswith('object_scale')]) | {'t': -1})
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
    pose = Pose.sample_gaussian_vmf_pose_approx(key, previous_pose, std, conc)
    log_q = Pose.logpdf_gaussian_vmf_pose_approx(pose, previous_pose, std, conc)
    return pose, log_q
