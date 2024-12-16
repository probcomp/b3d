### IMPORTS ###

import b3d.chisight.gen3d.transition_kernels as transition_kernels
import jax.numpy as jnp
import jax.random as r


def test_discrete_flip_kernel():
    num_values = 10
    possible_values = jnp.linspace(0, 1, num_values)
    flip_probability = 0.1
    kernel = transition_kernels.DiscreteFlipKernel(
        p_change_to_different_value=flip_probability, support=possible_values
    )

    assert jnp.isclose(
        kernel.logpdf(possible_values[0], possible_values[0]),
        jnp.log(1 - flip_probability),
    )
    assert jnp.isclose(
        kernel.logpdf(possible_values[0], possible_values[-1]),
        jnp.log(flip_probability / (num_values - 1)),
    )

    possible_values = jnp.array([0.01])
    flip_probability = 0.1
    kernel = transition_kernels.DiscreteFlipKernel(
        p_change_to_different_value=flip_probability, support=possible_values
    )
    assert jnp.isclose(
        kernel.logpdf(possible_values[0], possible_values[0]), jnp.log(1.0)
    )
    assert kernel.sample(r.PRNGKey(0), 0.01) == 0.01

    possible_values = jnp.array([0.01, 1.0])
    flip_probability = 0.1
    kernel = transition_kernels.DiscreteFlipKernel(
        p_change_to_different_value=flip_probability, support=possible_values
    )
    assert jnp.isclose(kernel.logpdf(0.01, 0.01), jnp.log(0.9))
    assert jnp.isclose(kernel.logpdf(1.0, 0.01), jnp.log(0.1))
