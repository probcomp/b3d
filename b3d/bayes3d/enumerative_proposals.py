import jax
import jax.numpy as jnp
import b3d
from b3d.pose import Pose
import genjax
from genjax import Pytree


def _enumerate_and_select_best_move(trace, addressses, key, all_deltas):
    addr = addressses[0]
    current_pose = trace.get_choices()[addr]
    for i in range(len(all_deltas)):
        test_poses = current_pose @ all_deltas[i]
        potential_scores = b3d.enumerate_choices_get_scores(
            trace, jax.random.PRNGKey(0), addressses, test_poses
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
                jax.random.split(key, number), trace.get_choices()[address.const], variance, concentration
            ),
            trace.get_choices()[address.const][None,...]
        ]
    )
    test_poses_batches = test_poses.split(10)
    scores = jnp.concatenate(
        [
            b3d.enumerate_choices_get_scores(
                trace, key, Pytree.const((address.const,)), poses
            )
            for poses in test_poses_batches
        ]
    )
    trace = b3d.update_choices(
        trace,
        jax.random.PRNGKey(0),
        (address,),
        test_poses[scores.argmax()],
    )
    key = jax.random.split(key, 2)[-1]
    return trace, key


gvmf_and_select_best_move = jax.jit(_gvmf_and_select_best_move, static_argnames=["address", "number"])



def _gvmf_and_sample(trace, key, variance, concentration, address, number):
    addr = address
    test_poses = Pose.concatenate_poses(
        [
            jax.vmap(Pose.sample_gaussian_vmf_pose, in_axes=(0, None, None, None))(
                jax.random.split(key, number), trace.get_choices()[addr], variance, concentration
            )
        ]
    )
    test_poses_batches = test_poses.split(10)
    scores = jnp.concatenate(
        [
            b3d.enumerate_choices_get_scores(
                trace, key, (addr,), poses
            )
            for poses in test_poses_batches
        ]
    )
    trace = b3d.update_choices(
        trace,
        jax.random.PRNGKey(0),
        (addr,),
        test_poses[jax.random.categorical(key, scores)],
    )
    key = jax.random.split(key, 2)[-1]
    return trace, key


gvmf_and_sample = jax.jit(_gvmf_and_sample, static_argnames=["address", "number"])

