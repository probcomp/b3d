import b3d.chisight.dynamic_object_model.kfold_image_kernel as kik
import jax
import jax.numpy as jnp

# importlib.reload(kik)
# loc, scale, low, high, uniform_window_size = 0.0, 0.01, 0.0, 1.0, 0.1
# n_grid_steps = 1000
# x = jnp.linspace(low, high, n_grid_steps)
# stepsize = (high - low) / n_grid_steps
# logpdfs = jax.vmap(
#     lambda x: kik.truncated_laplace.logpdf(x, loc, scale, low, high, uniform_window_size)
# )(x)
# pdfs = jnp.exp(logpdfs)
# jnp.sum(pdfs * stepsize)


def confirm_logpdf_looks_valid(
    loc, scale, low, high, uniform_window_size, n_grid_steps=100000
):
    """
    Test that the pdf value seems to integrate to 1.
    """
    x = jnp.linspace(low, high, n_grid_steps)
    stepsize = (high - low) / n_grid_steps
    logpdfs = jax.vmap(
        lambda x: kik.truncated_laplace.logpdf(
            x, loc, scale, low, high, uniform_window_size
        )
    )(x)
    pdfs = jnp.exp(logpdfs)
    total_pmass = jnp.sum(pdfs * stepsize)
    assert jnp.isclose(total_pmass, 1.0, atol=1e-3)


def ensure_laplace_samples_have_sufficient_spread(
    key, loc, scale, low, high, uniform_window_size, scale_mult=0.1
):
    samples = jax.vmap(
        lambda k: kik.truncated_laplace.sample(
            k, loc, scale, low, high, uniform_window_size
        )
    )(jax.random.split(key, 3))
    assert (
        jnp.abs(samples[0] - samples[1]) > scale * scale_mult
        or jnp.abs(samples[0] - samples[2]) > scale * scale_mult
        or jnp.abs(samples[1] - samples[2]) > scale * scale_mult
    )


def test_truncated_laplace():
    confirm_logpdf_looks_valid(0.5, 1.0, 0.0, 1.0, 0.1)
    confirm_logpdf_looks_valid(1.0, 1.0, 0.0, 1.0, 0.1)
    confirm_logpdf_looks_valid(0.0, 1.0, 0.0, 1.0, 0.1)
    confirm_logpdf_looks_valid(0.5, 0.01, 0.0, 1.0, 0.1)
    confirm_logpdf_looks_valid(0.0, 0.01, 0.0, 1.0, 0.1)
    confirm_logpdf_looks_valid(1.0, 0.01, 0.0, 1.0, 0.1)
    confirm_logpdf_looks_valid(0.99, 0.1, 0.0, 1.0, 0.2)
    confirm_logpdf_looks_valid(0.01, 0.1, 0.0, 1.0, 0.2)
    confirm_logpdf_looks_valid(1.99, 0.1, -1.0, 2.0, 0.2)
    confirm_logpdf_looks_valid(-0.99, 0.1, -1.0, 2.0, 0.2)
    confirm_logpdf_looks_valid(0.0, 5.0, -1.0, 2.0, 0.2)

    ensure_laplace_samples_have_sufficient_spread(
        jax.random.PRNGKey(0), 0.5, 0.1, 0.0, 1.0, 0.1
    )

    # TODO: the code below ehre could be cleaner.
    # I think the functionality is right, though.

    key = jax.random.PRNGKey(1)
    for j in range(5):
        key, _ = jax.random.split(key)
        x = kik.truncated_laplace.sample(key, 0.01, 0.01, 0.0, 1.0, 0.001)
        assert 0.0 < x < 0.05

    # test that the logpdf function puts almost all mass to the left
    x = jnp.linspace(0.0, 1.0, int(1e6))
    stepsize = 1e-6
    logpdfs = jax.vmap(
        lambda x: kik.truncated_laplace.logpdf(x, 0.01, 0.01, 0.0, 1.0, 0.001)
    )(x)
    pdfs = jnp.exp(logpdfs)
    assert jnp.sum(pdfs[: int(1e6 * 0.05)] * stepsize) > 0.98

    for j in range(5):
        key, _ = jax.random.split(key)
        x = kik.truncated_laplace.sample(key, -0.04, 0.01, 0.0, 1.0, 0.001)
        assert 0.0 < x < 0.001

    # test that the logpdf function also puts almost all mass to the left of 0.001
    x = jnp.linspace(0.0, 1.0, int(1e6))
    stepsize = 1e-6
    logpdfs = jax.vmap(
        lambda x: kik.truncated_laplace.logpdf(x, -0.04, 0.01, 0.0, 1.0, 0.001)
    )(x)
    pdfs = jnp.exp(logpdfs)
    assert jnp.sum(pdfs[: int(1e6 * 0.001)] * stepsize) > 0.98


def test_color_truncated_logpdf():
    s1 = kik.truncated_color_laplace.sample(
        jax.random.PRNGKey(0), jnp.array([1.0, 0.0, 0.5]), 0.2
    )
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    r = kik.truncated_laplace.sample(keys[0], 1.0, 0.2, 0.0, 1.0, 1 / 255)
    g = kik.truncated_laplace.sample(keys[1], 0.0, 0.2, 0.0, 1.0, 1 / 255)
    b = kik.truncated_laplace.sample(keys[2], 0.5, 0.2, 0.0, 1.0, 1 / 255)
    assert jnp.allclose(s1, jnp.array([r, g, b]))

    logpdf = kik.truncated_color_laplace.logpdf(s1, jnp.array([1.0, 0.0, 0.5]), 0.2)
    logpdf_r = kik.truncated_laplace.logpdf(r, 1.0, 0.2, 0.0, 1.0, 1 / 255)
    logpdf_g = kik.truncated_laplace.logpdf(g, 0.0, 0.2, 0.0, 1.0, 1 / 255)
    logpdf_b = kik.truncated_laplace.logpdf(b, 0.5, 0.2, 0.0, 1.0, 1 / 255)
    assert jnp.allclose(logpdf, logpdf_r + logpdf_g + logpdf_b)
