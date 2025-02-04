import jax
import jax.numpy as jnp
from genjax import Pytree

import b3d
from b3d.pose import Pose


def maybe_swap_in_gt_pose(proposed_poses, gt_pose, include_previous_pose):
    proposed_poses = jax.tree.map(
        lambda x, y: x.at[0].set(
            jnp.where(include_previous_pose, y, x[0])
        ),
        proposed_poses,
        gt_pose,
    )
    return proposed_poses


@jax.jit
def enumerate_and_select_best(trace, address, values):
    potential_scores = b3d.enumerate_choices_get_scores(trace, address, values)
    trace = b3d.update_choices(trace, address, values[potential_scores.argmax()])
    return trace


def _enumerate_and_select_best_move_pose(trace, addressses, key, all_deltas, gt_pose=None, include_previous_pose=False, k=50):
    addr = addressses.const[0]
    current_pose = trace.get_choices()[addr]
    for i in range(len(all_deltas)):
        test_poses = current_pose @ all_deltas[i]
        test_poses = maybe_swap_in_gt_pose(test_poses, gt_pose, include_previous_pose)
        potential_scores = b3d.enumerate_choices_get_scores(
            trace, addressses, test_poses
        )
        current_pose = test_poses[potential_scores.argmax()]
    top_k_indices = jnp.argsort(potential_scores)[-k:][::-1]
    top_scores = [potential_scores[idx] for idx in top_k_indices]
    posterior_poses = [test_poses[idx] for idx in top_k_indices]
    trace = b3d.update_choices(trace, addressses, current_pose)
    return trace, key, posterior_poses, top_scores


enumerate_and_select_best_move_pose = jax.jit(
    _enumerate_and_select_best_move_pose, static_argnames=["addressses"]
)


def _enumerate_and_select_best_move_scale(trace, addressses, key, all_deltas, k=50):
    addr = addressses.const[0]
    current_scale = trace.get_choices()[addr]
    test_scales = current_scale * all_deltas
    potential_scores = b3d.enumerate_choices_get_scores(trace, addressses, test_scales)
    best_scale = test_scales[potential_scores.argmax()]
    top_k_indices = jnp.argsort(potential_scores)[-k:][::-1]
    top_scores = [potential_scores[idx] for idx in top_k_indices]
    posterior_scales = [test_scales[idx] for idx in top_k_indices]
    trace = b3d.update_choices(trace, addressses, best_scale)
    return trace, key, posterior_scales, top_scores


enumerate_and_select_best_move_scale = jax.jit(
    _enumerate_and_select_best_move_scale, static_argnames=["addressses"]
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
