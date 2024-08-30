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
