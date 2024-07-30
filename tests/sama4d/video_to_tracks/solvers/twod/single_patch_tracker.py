import jax
import jax.numpy as jnp

from tests.common.solver import Solver


class SinglePatchTracker2D(Solver):
    def solve(self, task_specification):
        kp0 = task_specification["initial_keypoint_positions_2D"]
        assert kp0.shape == (
            1,
            2,
        ), "Currently only single-keypoint-tracking is supported."
        kp0 = kp0[0]
        rgb = task_specification["video"][:, :, :, :3]
        patch = get_patch_around_region_with_padding(rgb[0], kp0, size=15)
        bestfit_positions_x, bestfit_positions_y = jax.vmap(
            lambda img: get_best_fit_pos(img, patch)
        )(rgb)
        stacked_xy = jnp.stack([bestfit_positions_x, bestfit_positions_y], axis=-1)
        return stacked_xy[:, None, :]


def get_patch_around_region_with_padding(rgb, center, size=11, pad_value=-1):
    center = jnp.array(center, dtype=jnp.int32)
    x, y = center
    half_size = size // 2
    padded_rgb = jnp.pad(
        rgb,
        ((half_size, half_size), (half_size, half_size), (0, 0)),
        mode="constant",
        constant_values=-1,
    )
    return jax.lax.dynamic_slice(padded_rgb, (y, x, 0), (size, size, 3))


def patch_l1_error_at_position(rgb, center, patch):
    """Returns the L1 error between the patch and the patch centered at the given position."""
    return jnp.sum(
        jnp.abs(patch - get_patch_around_region_with_padding(rgb, center, size=15))
    )


def get_errors_across_image(rgb, patch):
    height, width, _ = rgb.shape
    return jax.vmap(
        jax.vmap(
            lambda x, y: patch_l1_error_at_position(rgb, (x, y), patch),
            in_axes=(0, None),
        ),
        in_axes=(None, 0),
    )(jnp.arange(0, height), jnp.arange(0, width))


def get_best_fit_pos(rgb, patch):
    errors = get_errors_across_image(rgb, patch)
    min_error = jnp.min(errors)
    y, x = jnp.where(errors == min_error, size=1)
    return x[0], y[0]
