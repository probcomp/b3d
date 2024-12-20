from functools import partial
from jax.scipy.spatial.transform import Rotation

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

from .utils import logmeanexp, update_vmapped_fields


@partial(jax.jit, static_argnames=("do_advance_time"))
def inference_step(
    key,
    trace,
    observed_rgbd,
    inference_hyperparams: InferenceHyperparams,
    addresses,
    do_advance_time=True,
    include_previous_pose=True,
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
        proposed_poses, proposed_vels, proposed_ang_vels, log_q_poses = jax.vmap(
            propose_pose, in_axes=(0, None, None, None)
        )(pose_generation_keys, trace, addr, pose_proposal_args)
        proposed_poses, proposed_vels, proposed_ang_vels, log_q_poses = maybe_swap_in_previous_pose(
            proposed_poses, proposed_vels, proposed_ang_vels, log_q_poses, trace, addr, include_previous_pose, pose_proposal_args
        )

        def update_and_get_scores(key, proposed_pose, proposed_vel, proposed_ang_vel, trace, addr):
            key, subkey = split(key)
            updated_trace = update_vmapped_fields(
                subkey,
                trace,
                [addr, addr.replace("pose", "vel"), addr.replace("pose", "ang_vel")],
                [proposed_pose, proposed_vel, proposed_ang_vel],
            )
            return updated_trace, updated_trace.get_score()

        param_generation_keys = split(k2, inference_hyperparams.n_poses)
        # _, p_scores = jax.lax.map(
        #     lambda x: update_and_get_scores(x[0], x[1], trace, addr),
        #     (param_generation_keys, proposed_poses),
        # )
        _, p_scores = jax.vmap(update_and_get_scores, in_axes=(0, 0, 0, 0, None, None))(
            param_generation_keys, proposed_poses, proposed_vels, proposed_ang_vels, trace, addr
        )

        # Scoring + resampling
        weights = jnp.where(
            inference_hyperparams.include_q_scores_at_top_level,
            p_scores - log_q_poses,
            p_scores,
        )

        chosen_index = jax.random.categorical(k3, weights)
        resampled_trace, _ = update_and_get_scores(
            param_generation_keys[chosen_index],
            proposed_poses[chosen_index],
            proposed_vels[chosen_index],
            proposed_ang_vels[chosen_index],
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

    for addr in addresses:
        for pose_proposal_args in inference_hyperparams.pose_proposal_args:
            key, subkey = split(key)
            # with jax.checking_leaks():
            trace, weight, _, _, _ = c2f_step(
                subkey,
                trace,
                pose_proposal_args,
                addr,
            )

    return (trace, weight)


def maybe_swap_in_previous_pose(
    proposed_poses, proposed_vels, proposed_ang_vels, log_q_poses, trace, addr, include_previous_pose, pose_proposal_args
):
    previous_pose, log_q = assess_previous_pose(trace, addr, pose_proposal_args)
    proposed_poses = jax.tree.map(
        lambda x, y: x.at[0].set(jnp.where(include_previous_pose, y, x[0])),
        proposed_poses,
        previous_pose,
    )
    proposed_vels = jax.tree.map(
        lambda x, y: x.at[0].set(jnp.where(include_previous_pose, y, x[0])),
        proposed_vels,
        jnp.zeros(3),
    )
    proposed_ang_vels = jax.tree.map(
        lambda x, y: x.at[0].set(jnp.where(include_previous_pose, y, x[0])),
        proposed_ang_vels,
        jnp.zeros(3),
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
    previous_pose = get_new_state(advanced_trace)[addr]
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
    # jax.debug.print("advance time prev state: {v}", v=trace.get_args()[1])
    # jax.debug.print("advance time choices: {v}", v=trace.get_sample())
    # jax.debug.print("advance time new state: {v} \n", v=trace.get_retval()['new_state'])
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
    def compute_angular_velocity(p1, p2):
        """
        Compute angular velocity in radians per second from two quaternions.

        Parameters:
            q1 (array-like): Quaternion at the earlier time [w, x, y, z].
            q2 (array-like): Quaternion at the later time [w, x, y, z].
        Returns:
            angular_velocity (numpy array): Angular velocity vector (radians per second).
        """
        # Convert quaternions to scipy Rotation objects
        rot1 = p1.rot()
        rot2 = p2.rot()

        # Compute the relative rotation
        relative_rotation = rot2 * rot1.inv()

        # Convert the relative rotation to angle-axis representation
        angle = relative_rotation.magnitude()  # Rotation angle in radians
        axis = (
            relative_rotation.as_rotvec() / angle if angle != 0 else jnp.zeros(3)
        )  # Rotation axis

        # Compute angular velocity
        angular_velocity = (axis * angle)
        return angular_velocity

    std, conc = args
    previous_pose = get_new_state(advanced_trace)[addr]
    pose = Pose.sample_gaussian_vmf_pose(key, previous_pose, std, conc)
    log_q = Pose.logpdf_gaussian_vmf_pose(pose, previous_pose, std, conc)

    vel = pose.pos - previous_pose.pos
    ang_vel = compute_angular_velocity(previous_pose, pose)
    return pose, vel, ang_vel, log_q
