### IMPORTS ###

import b3d.chisight.gen3d.transition_kernels as transition_kernels
import jax.numpy as jnp


def test_discrete_flip_kernel():
    num_values = 10
    possible_values = jnp.linspace(0, 1, num_values)
    flip_probability = 0.1
    kernel = transition_kernels.DiscreteFlipKernel(
        resample_probability=flip_probability, support=possible_values
    )

    assert jnp.isclose(
        kernel.logpdf(possible_values[0], possible_values[0]),
        jnp.log(1 - flip_probability),
    )
    assert jnp.isclose(
        kernel.logpdf(possible_values[0], possible_values[-1]),
        jnp.log(flip_probability / (num_values - 1)),
    )
