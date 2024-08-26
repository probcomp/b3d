import jax
import jax.numpy as jnp
from genjax import Pytree

import b3d
from b3d.pose import Pose


@jax.jit
def enumerate_and_select_best(trace, address, values):
    potential_scores = b3d.enumerate_choices_get_scores(trace, address, values)
    trace = b3d.update_choices(trace, address, values[potential_scores.argmax()])
    return trace


def _enumerate_and_select_best_move(trace, addressses, key, all_deltas):
    addr = addressses[0]
    current_pose = trace.get_choices()[addr]
    for i in range(len(all_deltas)):
        test_poses = current_pose @ all_deltas[i]
        potential_scores = b3d.enumerate_choices_get_scores(
            trace, addressses, test_poses
        )
        current_pose = test_poses[potential_scores.argmax()]
    trace = b3d.update_choices(trace, key, addressses, current_pose)
    return trace, key


enumerate_and_select_best_move = jax.jit(
    _enumerate_and_select_best_move, static_argnames=["addressses"]
)


def _gvmf_and_select_best_move(trace, key, variance, concentration, address, number):
    test_poses = Pose.concatenate_poses(
        [
            jax.vmap(Pose.sample_gaussian_vmf_pose, in_axes=(0, None, None, None))(
                jax.random.split(key, number),
                trace.get_choices()[address],
                variance,
                concentration,
            ),
            trace.get_choices()[address][None, ...],
        ]
    )
    test_poses_batches = test_poses.split(10)
    scores = jnp.concatenate(
        [
            b3d.enumerate_choices_get_scores(trace, Pytree.const((address,)), poses)
            for poses in test_poses_batches
        ]
    )
    trace = b3d.update_choices(
        trace,
        Pytree.const((address,)),
        test_poses[scores.argmax()],
    )
    key = jax.random.split(key, 2)[-1]
    return trace, key


gvmf_and_select_best_move = jax.jit(
    _gvmf_and_select_best_move, static_argnames=["address", "number"]
)


def _gvmf_and_sample(trace, key, variance, concentration, address, number):
    addr = address
    test_poses = Pose.concatenate_poses(
        [
            jax.vmap(Pose.sample_gaussian_vmf_pose, in_axes=(0, None, None, None))(
                jax.random.split(key, number),
                trace.get_choices()[addr],
                variance,
                concentration,
            )
        ]
    )
    test_poses_batches = test_poses.split(10)
    scores = jnp.concatenate(
        [
            b3d.enumerate_choices_get_scores(trace, Pytree.const((addr,)), poses)
            for poses in test_poses_batches
        ]
    )
    trace = b3d.update_choices(
        trace,
        Pytree.const((addr,)),
        test_poses[jax.random.categorical(key, scores)],
    )
    key = jax.random.split(key, 2)[-1]
    return trace, key


gvmf_and_sample = jax.jit(_gvmf_and_sample, static_argnames=["address", "number"])
