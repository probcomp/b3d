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
    addr = addressses.const[0]
    current_pose = trace.get_choices()[addr]
    for i in range(len(all_deltas[0])):
        test_poses = current_pose @ all_deltas[0][i]
        # jax.debug.print("test_poses: {v}", v=test_poses)
        sweeps = [test_poses] + all_deltas[1:]
        # print("sweeps: ", sweeps)
        potential_scores = b3d.utils.grid_trace(trace, addressses, sweeps)
        # jax.debug.print("potential_scores: {v}", v=potential_scores)
        # print("potential_scores: ", potential_scores)
        indices = jnp.unravel_index(potential_scores.argmax(), potential_scores.shape)
        # jax.debug.print("indices: {v}", v=indices)
        # print("indices: ", indices)
        current_pose = test_poses[indices[0]]
        # jax.debug.print("current_pose: {v}", v=current_pose)
        # print("current_pose: ", current_pose)
    optimal_parameters = [sweep[index] for index, sweep in zip(indices, sweeps)]
    # jax.debug.print("optimal_parameters: {v}", v=optimal_parameters)
    # print("addressses.const: ", addressses.const)
    # print("optimal_parameters: ", optimal_parameters)
    # trace = b3d.update_choices(trace, addressses, optimal_parameters[0])
    for address, optimal_parameter in zip(addressses.const, optimal_parameters):
        # jax.debug.print("optimal_parameter: {v}", v=optimal_parameter)
        trace = b3d.update_choices(trace, Pytree.const((address,)), optimal_parameter)
    return trace, key


def _enumerate_and_select_best_move_old(trace, addressses, key, all_deltas):
    addr = addressses.const[0]
    current_pose = trace.get_choices()[addr]
    for i in range(len(all_deltas)):
        test_poses = current_pose @ all_deltas[i]
        # jax.debug.print("test_poses: {v}", v=test_poses)
        potential_scores = b3d.enumerate_choices_get_scores(
            trace, addressses, test_poses
        )
        # jax.debug.print("potential_scores: {v}", v=potential_scores)
        # print("potential_scores: ", potential_scores)
        # jax.debug.print("indices: {v}", v=potential_scores.argmax())
        current_pose = test_poses[potential_scores.argmax()]
        # jax.debug.print("current_pose: {v}", v=current_pose)
        # print("current_pose: ", current_pose)
    trace = b3d.update_choices(trace, addressses, current_pose)
    return trace, key


enumerate_and_select_best_move = jax.jit(
    _enumerate_and_select_best_move, static_argnames=["addressses"]
)


enumerate_and_select_best_move_old = jax.jit(
    _enumerate_and_select_best_move_old, static_argnames=["addressses"]
)


# def _enumerate_and_select_best_move_scale(trace, addressses, key, all_deltas):
#     potential_scores = b3d.enumerate_choices_get_scores(trace, addressses, all_deltas)
#     best_scale = all_deltas[potential_scores.argmax()]
#     trace = b3d.update_choices(trace, addressses, best_scale)
#     return trace, key


# enumerate_and_select_best_move_scale = jax.jit(
#     _enumerate_and_select_best_move_scale, static_argnames=["addressses"]
# )


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
