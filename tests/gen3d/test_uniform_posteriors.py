import b3d.chisight.gen3d.uniform_distributions as ud
import jax
import jax.numpy as jnp


def get_grid_posterior_approximation(prior, obsmodel, obs):
    tight_grid_latent = jnp.linspace(0.0, 1.0, 1000001)
    enum_posterior_pdf_tight = ud._get_enum_posterior_pdf(
        prior, obsmodel, obs, tight_grid_latent
    )
    subsampled_values = tight_grid_latent[::1000]
    subsampled_pdfs = enum_posterior_pdf_tight[::1000]
    return subsampled_values, subsampled_pdfs


def test_uniform_posteriors_are_computed_correctly():
    prior = ud.Uniform(0.3, 0.7)
    obsmodel = ud.NiceTruncatedCenteredUniform(0.05, 0.0, 1.0)
    for obs in [0.29, 0.3, 0.35, 0.67, 0.9]:
        obs = 0.29
        values, grid_pdfs = get_grid_posterior_approximation(prior, obsmodel, obs)
        exact_pdfs = jax.vmap(
            lambda lat: jnp.exp(
                ud.get_posterior_for_uniform_prior_ntcu_obs(
                    prior, obsmodel, obs
                ).logpdf(lat)
            )
        )(values)

        # Due to floating point error, it's hard to get the pdf exactly
        # at the boundaries of the uniform distribution right.
        # Hence, we may have a few points where the pdfs are off by a bit.
        # But at all points not near the boundary, the pdfs should be very close.
        # TODO: does this indicate that we should actually be using distributions
        # that do something more stochastic when very close to the boundaries?
        n_fail = jnp.sum(jnp.abs(grid_pdfs - exact_pdfs) > 1e-3)
        assert n_fail < 4

    prior = ud.MixtureOfUniforms(
        jnp.array([0.7, 0.25, 0.05]),
        ud.Uniform(jnp.array([0.1, 0.4, 0.49]), jnp.array([0.9, 0.6, 0.51])),
    )
    obsmodel = ud.MixtureOfNiceTruncatedCenteredUniforms(
        jnp.array([0.3, 0.7]),
        ud.NiceTruncatedCenteredUniform(
            jnp.array([0.1, 0.02]), jnp.zeros(2), jnp.ones(2)
        ),
    )
    for obs in [0.29, 0.3, 0.35, 0.67, 0.9]:
        obs = 0.3
        values, grid_pdfs = get_grid_posterior_approximation(prior, obsmodel, obs)
        exact_pdfs = jax.vmap(
            lambda lat: jnp.exp(
                ud.get_posterior_from_mix_of_uniform_prior_and_mix_of_nctus_obs(
                    prior, obsmodel, obs
                ).logpdf(lat)
            )
        )(values)
        # Due to floating point error, it's hard to get the pdf exactly
        # at the boundaries of the uniform distribution right.
        # Hence, we may have a few points where the pdfs are off by a bit.
        # But at all points not near the boundary, the pdfs should be very close.
        # TODO: does this indicate that we should actually be using distributions
        # that do something more stochastic when very close to the boundaries?
        n_fail = jnp.sum(jnp.abs(grid_pdfs - exact_pdfs) > 1e-3)
        assert n_fail < 4
