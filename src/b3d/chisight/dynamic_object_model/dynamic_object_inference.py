from functools import partial

import jax
import jax.numpy as jnp
from genjax import Pytree

import b3d
from b3d import Pose

from .dynamic_object_model import info_from_trace

### Inference moves ###


@jax.jit
def score_with_give_pose_and_then_color_update(
    trace, address, pose, outlier_probability_sweep
):
    trace = b3d.update_choices(trace, address, pose)

    info = info_from_trace(trace)

    color_delta = info["observed_rgbd_masked"][..., :3] - trace.get_choices()["colors"]
    max_color_shift = trace.get_args()[0]["max_color_shift"].const
    color_delta_clipped = jnp.clip(color_delta, -max_color_shift, max_color_shift)

    trace = b3d.update_choices(
        trace,
        Pytree.const(("colors",)),
        trace.get_choices()["colors"] + color_delta_clipped,
    )

    trace = grid_move_on_outlier_probability(
        trace, Pytree.const(("color_outlier_probability",)), outlier_probability_sweep
    )
    trace = grid_move_on_outlier_probability(
        trace, Pytree.const(("depth_outlier_probability",)), outlier_probability_sweep
    )

    return trace.get_score()


@partial(jax.jit, static_argnames=("number",))
def gaussian_vmf_enumerative_move_with_other_updates(
    trace, key, address, variance, concentration, number, outlier_probability_sweep
):
    keys = jax.random.split(key, number)
    poses = Pose.concatenate_poses(
        [
            Pose.sample_gaussian_vmf_pose_vmap(
                keys, trace.get_choices()["pose"], 0.02, 2000.0
            ),
            trace.get_choices()["pose"][None, ...],
        ]
    )

    scores = jax.vmap(
        score_with_give_pose_and_then_color_update, in_axes=(None, None, 0, None)
    )(trace, Pytree.const(("pose",)), poses, outlier_probability_sweep)

    key = b3d.split_key(keys[-1])
    sampled_pose = poses[scores.argmax()]
    trace = b3d.update_choices(trace, Pytree.const(("pose",)), sampled_pose)
    return trace, key


@jax.jit
def grid_move_on_outlier_probability(trace, address, sweep):
    current_setting = trace.get_choices()[address.const]

    potential_values = sweep[..., None] * jnp.ones_like(current_setting)

    get_vertex_scores_vmap = jax.vmap(
        lambda x: info_from_trace(b3d.update_choices(trace, address, x))["scores"]
    )
    scores = get_vertex_scores_vmap(potential_values)
    best_setting = sweep[jnp.argmax(scores, axis=0)]
    return b3d.update_choices(trace, address, best_setting)


@jax.jit
def update_address_with_sweep(trace, address, sweep):
    scores = b3d.enumerate_choices_get_scores(trace, address, sweep)
    best_setting = sweep[jnp.argmax(scores)]
    return b3d.update_choices(trace, address, best_setting)


### Inference step loop ###


@jax.jit
def inference_step(trace, key, observed_rgbd):
    outlier_probability_sweep = jnp.array([0.01, 0.5, 1.0])
    trace = b3d.update_choices(
        trace,
        Pytree.const(("old_pose", "old_colors")),
        trace.get_choices()["pose"],
        trace.get_choices()["colors"],
    )

    trace = b3d.update_choices(trace, Pytree.const(("rgbd",)), observed_rgbd)

    for _ in range(2):
        trace, key = gaussian_vmf_enumerative_move_with_other_updates(
            trace,
            key,
            Pytree.const(("pose",)),
            0.04,
            200.0,
            10000,
            outlier_probability_sweep,
        )
        trace, key = gaussian_vmf_enumerative_move_with_other_updates(
            trace,
            key,
            Pytree.const(("pose",)),
            0.01,
            500.0,
            10000,
            outlier_probability_sweep,
        )
        trace, key = gaussian_vmf_enumerative_move_with_other_updates(
            trace,
            key,
            Pytree.const(("pose",)),
            0.005,
            1000.0,
            10000,
            outlier_probability_sweep,
        )
        trace, key = gaussian_vmf_enumerative_move_with_other_updates(
            trace,
            key,
            Pytree.const(("pose",)),
            0.001,
            2000.0,
            10000,
            outlier_probability_sweep,
        )

    trace = grid_move_on_outlier_probability(
        trace, Pytree.const(("color_outlier_probability",)), outlier_probability_sweep
    )
    trace = grid_move_on_outlier_probability(
        trace, Pytree.const(("depth_outlier_probability",)), outlier_probability_sweep
    )

    trace = update_address_with_sweep(
        trace,
        Pytree.const(("color_variance",)),
        jnp.array([0.005, 0.01, 0.05, 0.1, 0.2]),
    )
    trace = update_address_with_sweep(
        trace,
        Pytree.const(("depth_variance",)),
        jnp.array([0.0005, 0.001, 0.0025, 0.005, 0.01]),
    )

    info = info_from_trace(trace)
    color_delta = info["observed_rgbd_masked"][..., :3] - trace.get_choices()["colors"]
    max_color_shift = trace.get_args()[0]["max_color_shift"].const
    color_delta_clipped = jnp.clip(color_delta, -max_color_shift, max_color_shift)
    trace = b3d.update_choices(
        trace,
        Pytree.const(("colors",)),
        trace.get_choices()["colors"] + color_delta_clipped,
    )
    return trace
