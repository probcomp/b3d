import jax
import jax.numpy as jnp
import pytest
from b3d.chisight.gen3d.pixel_kernels.pixel_depth_kernels import (
    DEPTH_NONRETURN_VAL,
    UNEXPLAINED_DEPTH_NONRETURN_PROB,
    FullPixelDepthDistribution,
    MixturePixelDepthDistribution,
    TruncatedLaplacePixelDepthDistribution,
    UnexplainedPixelDepthDistribution,
    UniformPixelDepthDistribution,
)

near = 0.01
far = 20.0

# each kernel specs is a tuple of (kernel, additional_args)
sample_kernels_to_test = [
    (UniformPixelDepthDistribution(near, far), ()),
    (TruncatedLaplacePixelDepthDistribution(near, far, 0.25), ()),
    (UnexplainedPixelDepthDistribution(near, far), ()),
    (
        MixturePixelDepthDistribution(near, far, 0.15),
        (
            0.5,  # occluded_prob
            0.23,  # depth_nonreturn_prob
        ),
    ),
    (
        FullPixelDepthDistribution(near, far, 0.5),
        (
            0.3,  # occluded_prob
            0.1,  # depth_nonreturn_prob
        ),
    ),
]


@pytest.mark.parametrize("latent_depth", [0.2, 10.0])
@pytest.mark.parametrize("kernel_spec", sample_kernels_to_test)
def test_logpdf_sum_to_1(kernel_spec, latent_depth: float):
    kernel, additional_args = kernel_spec
    # compute the mass in [near, far]
    n_grid_steps = 10000000
    depth_grid = jnp.linspace(near, far, n_grid_steps)
    logpdfs = jax.vmap(
        lambda depth: kernel.logpdf(depth, latent_depth, *additional_args)
    )(depth_grid)
    log_pmass = (
        jax.scipy.special.logsumexp(logpdfs)
        + jnp.log(far - near)
        - jnp.log(n_grid_steps)
    )
    # compute the mass in { "nonreturn" }
    log_nonreturn_mass = kernel.logpdf(
        DEPTH_NONRETURN_VAL, latent_depth, *additional_args
    )
    log_pmass = jnp.logaddexp(log_pmass, log_nonreturn_mass)

    assert jnp.isclose(log_pmass, 0.0, atol=1e-3)


@pytest.mark.parametrize("latent_depth", [0.2, 10.0, 19.9])
@pytest.mark.parametrize("kernel_spec", sample_kernels_to_test)
def test_sample_in_valid_depth_range(kernel_spec, latent_depth):
    kernel, additional_args = kernel_spec
    num_samples = 1000
    keys = jax.random.split(jax.random.PRNGKey(0), num_samples)
    depths = jax.vmap(lambda key: kernel.sample(key, latent_depth, *additional_args))(
        keys
    )
    assert depths.shape == (num_samples,)
    assert jnp.all((depths > near) | (depths == DEPTH_NONRETURN_VAL))
    assert jnp.all((depths < far) | (depths == DEPTH_NONRETURN_VAL))


def test_relative_logpdf():
    kernel = FullPixelDepthDistribution(near, far, 0.1)

    # case 1: depth is missing in observation (nonreturn)
    obs_depth = DEPTH_NONRETURN_VAL
    latent_depth = DEPTH_NONRETURN_VAL
    depth_nonreturn_prob = 0.2
    logpdf_1 = kernel.logpdf(obs_depth, latent_depth, 0.8, depth_nonreturn_prob)
    assert logpdf_1 == jnp.log(depth_nonreturn_prob)

    latent_depth = -1.0  # no depth information from latent
    logpdf_2 = kernel.logpdf(obs_depth, latent_depth, 0.8, depth_nonreturn_prob)
    # nonreturn obs cannot be generates from latent that is not nonreturn
    assert logpdf_2 == jnp.log(UNEXPLAINED_DEPTH_NONRETURN_PROB)

    # case 2: valid depth is observed, but latent depth is far from the observed depth
    obs_depth = 10.0
    latent_depth = 0.01
    logpdf_3 = kernel.logpdf(obs_depth, latent_depth, 0.9, depth_nonreturn_prob)
    logpdf_4 = kernel.logpdf(obs_depth, latent_depth, 0.1, depth_nonreturn_prob)
    # the pixel should be more likely to be an occluded
    assert logpdf_3 > logpdf_4

    # case 3: valid depth is observed, but latent depth is close from the observed depth
    obs_depth = 6.0
    latent_depth = 6.01
    logpdf_5 = kernel.logpdf(obs_depth, latent_depth, 0.9, depth_nonreturn_prob)
    logpdf_6 = kernel.logpdf(obs_depth, latent_depth, 0.1, depth_nonreturn_prob)
    # the pixel should be more likely to be an inliner
    assert logpdf_5 < logpdf_6
