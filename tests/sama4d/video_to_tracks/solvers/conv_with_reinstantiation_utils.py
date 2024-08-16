import jax
import jax.numpy as jnp


### Logic for finding the best patch in a frame ###
def get_patch_around_region_with_padding(rgb, center, size, pad_value=-1):
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
        jnp.abs(
            patch
            - get_patch_around_region_with_padding(rgb, center, size=patch.shape[0])
        )
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


def get_best_fit_pos(rgb, patch, mindist_for_second_error, maxdist_for_second_error):
    errors = get_errors_across_image(rgb, patch)
    min_error = jnp.min(errors)
    y, x = jnp.where(errors == min_error, size=1)

    # now, compute the minimum error at least 4 units
    # away from (x, y)
    (H, W) = errors.shape
    _, min_error2 = get_second_error_xy(
        H, W, y, x, errors, mindist_for_second_error, maxdist_for_second_error
    )

    return (x[0], y[0]), min_error, min_error2


def get_second_error_xy(H, W, y, x, errors, mindist, maxdist):
    yy, xx = jnp.mgrid[:H, :W]

    # Compute distances from (y, x)
    distances = jnp.sqrt((yy - y) ** 2 + (xx - x) ** 2)

    # Create a mask for points at least mindist units away
    mask = jnp.logical_and(distances >= mindist, distances <= maxdist)

    # Apply the mask to the errors array
    masked_errors = jnp.where(mask, errors, jnp.inf)

    # Find the minimum error and its location in the masked array
    min_error2 = jnp.min(masked_errors)
    y2, x2 = jnp.where(masked_errors == min_error2, size=1)

    return (y2, x2), min_error2


### Utils ###
def replace_using_elements_from_other_vector(boolmask, x, new_values):
    """
    This is used when we want to replace some values in x
    using elements from new_values.
    This function returns a version of x, where all elements
    at index i with boolmask[i] == False are replaced with elements
    from new_values.
    The elements of new_values are pulled out in order.
    This is JAX compatible.
    """

    # state = (current_array, idx_in_x, idx_in_new_indices_full)
    state = (x, 0, 0)

    def step(state, _):
        x, xidx, nidx = state
        x = x.at[xidx].set(jnp.where(boolmask, x[xidx], new_values[nidx]))
        new_nidx = jnp.where(boolmask, nidx, nidx + 1)
        new_xidx = xidx + 1
        return (x, new_xidx, new_nidx), None

    final_state, _ = jax.lax.scan(step, state, jnp.arange(x.size))
    return final_state[0]
