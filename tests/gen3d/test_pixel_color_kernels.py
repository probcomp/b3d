from functools import partial

import jax
import jax.numpy as jnp
import pytest
from b3d.chisight.gen3d.pixel_kernels.pixel_color_kernels import (
    COLOR_MAX_VAL,
    COLOR_MIN_VAL,
    RenormalizedGaussianPixelColorDistribution,
    RenormalizedLaplacePixelColorDistribution,
    TruncatedLaplacePixelColorDistribution,
    UniformPixelColorDistribution,
)
from genjax.typing import FloatArray


@partial(jax.jit, static_argnums=(0,))
def generate_color_grid(n_grid_steps: int):
    """Generate a grid of colors in very small interval to test that our logpdfs
    sum to 1. Since enumerating all color combination in 3 channels is infeasible,
    we take advantage of the fact that the channels are independent and only
    grid over the first channel.

    Args:
        n_grid_steps (int): The number of grid steps to generate

    Returns:
        FloatArray: A grid of colors with shape (n_grid_steps, 3), where the first
        channel is swept from 0 to 1 and the other two channels are fixed at 1.
    """
    sweep_color_vals = jnp.linspace(COLOR_MIN_VAL, COLOR_MAX_VAL, n_grid_steps)
    fixed_color_vals = jnp.ones(n_grid_steps)
    return jnp.stack([sweep_color_vals, fixed_color_vals, fixed_color_vals], axis=-1)


sample_kernels_to_test = [
    (UniformPixelColorDistribution(), ()),
    (TruncatedLaplacePixelColorDistribution(), (0.1,)),
    (RenormalizedLaplacePixelColorDistribution(), (0.1,)),
    (RenormalizedGaussianPixelColorDistribution(), (0.1,)),
]


@pytest.mark.parametrize("latent_color", [jnp.array([0.2, 0.5, 0.3]), jnp.zeros(3)])
@pytest.mark.parametrize("kernel_spec", sample_kernels_to_test)
def test_logpdf_sum_to_1(kernel_spec, latent_color: FloatArray):
    kernel, additional_args = kernel_spec
    n_grid_steps = 10000000
    color_grid = generate_color_grid(n_grid_steps)
    logpdf_per_channels = jax.vmap(
        lambda color: kernel.logpdf_per_channel(color, latent_color, *additional_args)
    )(color_grid)
    log_pmass = jax.scipy.special.logsumexp(logpdf_per_channels[..., 0]) - jnp.log(
        n_grid_steps
    )
    assert jnp.isclose(log_pmass, 0.0, atol=1e-3)


@pytest.mark.parametrize(
    "latent_color", [jnp.array([0.25, 0.87, 0.31]), jnp.zeros(3), jnp.ones(3)]
)
@pytest.mark.parametrize("kernel_spec", sample_kernels_to_test)
def test_sample_in_valid_color_range(kernel_spec, latent_color):
    kernel, additional_args = kernel_spec
    num_samples = 1000
    keys = jax.random.split(jax.random.PRNGKey(0), num_samples)
    colors = jax.vmap(lambda key: kernel.sample(key, latent_color, *additional_args))(
        keys
    )
    assert colors.shape == (num_samples, 3)
    assert jnp.all(colors >= 0)
    assert jnp.all(colors <= 1)


# def test_relative_logpdf():
#     kernel = FullPixelColorDistribution()
#     scale = 0.01
#     obs_color = jnp.array([0.0, 0.0, 1.0])  # a blue pixel

#     # case 1: no color hit the pixel
#     latent_color = -jnp.ones(3)  # use -1 to denote invalid pixel
#     logpdf_1 = kernel.logpdf(obs_color, latent_color, scale, 0.8)
#     logpdf_2 = kernel.logpdf(obs_color, latent_color, scale, 0.2)
#     # the logpdf should be the same because the occluded probability is not used
#     # in the case when no color hit the pixel
#     assert jnp.allclose(logpdf_1, logpdf_2)

#     # case 2: a color hit the pixel, but the color is not close to the observed color
#     latent_color = jnp.array([1.0, 0.5, 0.0])
#     logpdf_3 = kernel.logpdf(obs_color, latent_color, scale, 0.8)
#     logpdf_4 = kernel.logpdf(obs_color, latent_color, scale, 0.2)
#     # the pixel should be more likely to be an occluded
#     assert logpdf_3 < logpdf_4

#     # case 3: a color hit the pixel, and the color is close to the observed color
#     latent_color = jnp.array([0.0, 0.0, 0.9])
#     logpdf_5 = kernel.logpdf(obs_color, latent_color, 0.01, 0.8)
#     logpdf_6 = kernel.logpdf(obs_color, latent_color, scale, 0.2)
#     # the pixel should be more likely to be an inlier
#     assert logpdf_5 > logpdf_6
#     # the score of the pixel should be higher when the color is closer
#     assert logpdf_5 > logpdf_3
