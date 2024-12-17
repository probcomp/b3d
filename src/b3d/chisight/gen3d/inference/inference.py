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
)
from b3d.chisight.gen3d.hyperparams import InferenceHyperparams

from .utils import logmeanexp, update_field


@jax.jit
def c2f_step(
    key,
    trace,
    pose_proposal_args,
    inference_hyperparams,
    addr,
):
    k1, k2, k3 = split(key, 3)
    addr = addr.unwrap()

    # Propose the poses
    pose_generation_keys = split(k1, inference_hyperparams.n_poses)
    proposed_poses, log_q_poses = jax.vmap(propose_pose, in_axes=(0, None, None, None))(
        pose_generation_keys, trace, addr, pose_proposal_args
    )

    param_generation_keys = split(k2, inference_hyperparams.n_poses)

    def update_and_get_scores(key, proposed_pose, trace, addr):
        key, subkey = split(key)
        updated_trace = update_field(subkey, trace, addr, proposed_pose)
        return updated_trace, updated_trace.get_score()

    _, p_scores = jax.lax.map(
        lambda x: update_and_get_scores(x[0], x[1], trace, addr),
        (param_generation_keys, proposed_poses),
    )

    # Scoring + resampling
    weights = jnp.where(
        inference_hyperparams.include_q_scores_at_top_level,
        p_scores - log_q_poses,
        p_scores,
    )

    chosen_index = jax.random.categorical(k3, weights)
    resampled_trace, _ = update_and_get_scores(
        param_generation_keys[chosen_index], proposed_poses[chosen_index], trace, addr
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
    addresses,
    do_advance_time=True,
):
    if do_advance_time:
        key, subkey = split(key)
        trace = advance_time(subkey, trace, observed_rgbd)

    for addr in addresses:
        for pose_proposal_args in inference_hyperparams.pose_proposal_args:
            key, subkey = split(key)
            trace, weight, _, _, _ = c2f_step(
                subkey,
                trace,
                pose_proposal_args,
                inference_hyperparams,
                addr,
            )

    return (trace, weight)


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
    dynamic_object_generative_model = (
        b3d.chisight.dense.dense_model.make_dense_multiobject_dynamics_model(
            renderer, likelihood_func
        )
    )
    trace, weight = dynamic_object_generative_model.importance(
        key, choicemap, (hyperparams, initial_state)
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
