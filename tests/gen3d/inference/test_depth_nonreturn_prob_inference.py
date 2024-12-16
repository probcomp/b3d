# import b3d.chisight.gen3d.inference_moves as im
# import b3d.chisight.gen3d.transition_kernels as transition_kernels
# import jax
# import jax.numpy as jnp
# import jax.random as r
# from b3d.chisight.gen3d.pixel_kernels.pixel_depth_kernels import (
#     FullPixelDepthDistribution,
# )

# near, far = 0.001, 1.0

# dnrp_transition_kernel = transition_kernels.DiscreteFlipKernel(
#     p_change_to_different_value=0.05, support=jnp.array([0.01, 0.99])
# )


# def propose_val(k):
#     return im._propose_vertex_depth_nonreturn_prob(
#         k,
#         observed_depth=0.8,
#         latent_depth=1.0,
#         visibility_prob=1.0,
#         depth_scale=0.00001,
#         previous_dnrp=0.01,
#         dnrp_transition_kernel=dnrp_transition_kernel,
#         obs_depth_kernel=FullPixelDepthDistribution(near, far),
#     )


# values, log_qs, _ = jax.vmap(propose_val)(r.split(r.PRNGKey(0), 1000))
# n_01 = jnp.sum((values == 0.01).astype(jnp.int32))
# assert n_01 >= 950
