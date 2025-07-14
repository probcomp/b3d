from jax import numpy as jnp


def convert_rgb_float_to_uint8(rgb):
    min_val = jnp.min(rgb)
    max_val = jnp.max(rgb)
    assert jnp.issubdtype(rgb.dtype, jnp.floating), "data is not of type float"
    assert max_val <= 1 and min_val >= 0, (
        f"video input rgb is float between 0 and 1, got min: {min_val} and max: {max_val}"
    )
    return (rgb * 255).astype(jnp.uint8)
