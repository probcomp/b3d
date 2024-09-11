import jax
import jax.numpy as jnp
import pytest
from b3d.chisight.gen3d.pixel_kernels import (
    DEPTH_NONRETURN_VAL,
    FullPixelRGBDDistribution,
)
from b3d.chisight.gen3d.pixel_kernels.pixel_color_kernels import (
    TruncatedLaplacePixelColorDistribution,
    UniformPixelColorDistribution,
)
from b3d.chisight.gen3d.pixel_kernels.pixel_depth_kernels import (
    TruncatedLaplacePixelDepthDistribution,
    UniformPixelDepthDistribution,
)

near = 0.01
far = 20.0

sample_kernels_to_test = [
    (
        FullPixelRGBDDistribution(
            TruncatedLaplacePixelColorDistribution(),
            UniformPixelColorDistribution(),
            TruncatedLaplacePixelDepthDistribution(near, far),
            UniformPixelDepthDistribution(near, far),
        ),
        (
            0.01,  # color_scale
            0.01,  # depth_scale
            1 - 0.3,  # visibility_prob
            0.1,  # depth_nonreturn_prob
        ),
    )
]


@pytest.mark.parametrize(
    "latent_rgbd",
    [
        jnp.array([0.25, 0.87, 0.31, 10.0]),
        jnp.array([0.0, 0.0, 0.0, 0.02]),
        jnp.array([1.0, 1.0, 1.0, 19.9]),
    ],
)
@pytest.mark.parametrize("kernel_spec", sample_kernels_to_test)
def test_sample_in_valid_rgbd_range(kernel_spec, latent_rgbd):
    kernel, additional_args = kernel_spec
    num_samples = 1000
    keys = jax.random.split(jax.random.PRNGKey(0), num_samples)
    rgbds = jax.vmap(lambda key: kernel.sample(key, latent_rgbd, *additional_args))(
        keys
    )
    assert rgbds.shape == (num_samples, 4)
    assert jnp.all(rgbds[..., :3] > 0)
    assert jnp.all(rgbds[..., :3] < 1)
    assert jnp.all((rgbds[..., 3] > near) | (rgbds[..., 3] == DEPTH_NONRETURN_VAL))
    assert jnp.all((rgbds[..., 3] < far) | (rgbds[..., 3] == DEPTH_NONRETURN_VAL))


@pytest.mark.parametrize("kernel_spec", sample_kernels_to_test)
def test_relative_logpdf(kernel_spec):
    kernel, _ = kernel_spec
    obs_rgbd = jnp.array([0.0, 0.0, 1.0, 0.02])  # a blue pixel

    # case 1: no vertex hit the pixel
    latent_rgbd = -jnp.ones(4)  # use -1 to denote invalid pixel
    logpdf_1 = kernel.logpdf(
        obs_rgbd, latent_rgbd, 0.01, 0.01, visibility_prob=0.8, depth_nonreturn_prob=0.1
    )
    logpdf_2 = kernel.logpdf(
        obs_rgbd, latent_rgbd, 0.01, 0.01, visibility_prob=0.2, depth_nonreturn_prob=0.1
    )
    # the logpdf should be the same because the occluded probability is not used
    # in the case when no vertex hit the pixel
    assert jnp.allclose(logpdf_1, logpdf_2)

    # case 2: a vertex hit the pixel, but the rgbd is not close to the observed rgbd
    latent_rgbd = jnp.array([1.0, 0.5, 0.0, 12.0])
    logpdf_3 = kernel.logpdf(
        obs_rgbd, latent_rgbd, 0.01, 0.01, visibility_prob=0.8, depth_nonreturn_prob=0.1
    )
    logpdf_4 = kernel.logpdf(
        obs_rgbd, latent_rgbd, 0.01, 0.01, visibility_prob=0.2, depth_nonreturn_prob=0.1
    )
    # the pixel should be more likely to be an occluded
    assert logpdf_3 < logpdf_4

    # case 3: a vertex hit the pixel, and the rgbd is close to the observed rgbd
    latent_rgbd = jnp.array([0.0, 0.0, 0.95, 0.022])
    logpdf_5 = kernel.logpdf(
        obs_rgbd, latent_rgbd, 0.01, 0.01, visibility_prob=0.8, depth_nonreturn_prob=0.1
    )
    logpdf_6 = kernel.logpdf(
        obs_rgbd, latent_rgbd, 0.01, 0.01, visibility_prob=0.2, depth_nonreturn_prob=0.1
    )
    # the pixel should be more likely to be an inlier
    assert logpdf_5 > logpdf_6
    # the score of the pixel should be higher when the rgbd is closer
    assert logpdf_5 > logpdf_3


@pytest.mark.parametrize("kernel_spec", sample_kernels_to_test)
def test_invalid_pixel(kernel_spec):
    kernel, additional_args = kernel_spec

    # Latent value of [-1, -1, -1, -1] indicates no point hits here.
    latent_rgbd = -jnp.ones(4)
    logpdf_1 = kernel.logpdf(
        jnp.array([1.0, 0.5, 0.2, 4.0]), latent_rgbd, *additional_args
    )
    logpdf_2 = kernel.logpdf(
        jnp.array([0.0, 0.0, 0.0, 0.02]), latent_rgbd, *additional_args
    )
    # the observation does not affect the logpdf
    assert logpdf_1 == logpdf_2

    logpdf_3 = kernel.logpdf(
        jnp.array([1.0, 0.5, 0.2, 4.0]), latent_rgbd, 0.1, 0.4, 0.2, 0.1
    )
    logpdf_4 = kernel.logpdf(
        jnp.array([0.0, 0.0, 0.0, 0.02]), latent_rgbd, 0.3, 0.5, 0.4, 0.2
    )
    # and the values of the parameters doesn't matter either
    assert logpdf_2 == logpdf_3
    assert logpdf_3 == logpdf_4

    # IMPORTANT: note that, by designed, every pixel should have a valid color,
    # and an observation of [-1, -1, -1, -1] is actually not within the support
    # of the pixel distribution.
    logpdf_5 = kernel.logpdf(
        jnp.array([-1.0, -1.0, -1.0, -1.0]), latent_rgbd, *additional_args
    )
    assert logpdf_5 == -jnp.inf
