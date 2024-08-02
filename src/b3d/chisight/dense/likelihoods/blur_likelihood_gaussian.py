# import functools

# import genjax
# import jax
# import jax.numpy as jnp


# def log_gaussian_kernel(size: int, sigma: float) -> jnp.ndarray:
#     """Creates a 2D Gaussian kernel."""
#     ax = jnp.arange(-size // 2 + 1.0, size // 2 + 1.0)
#     xx, yy = jnp.meshgrid(ax, ax)
#     kernel = -(xx**2 + yy**2) / (2.0 * sigma**2)
#     kernel = kernel - jax.nn.logsumexp(kernel)
#     return kernel


# lower_bound = jnp.array([0.0, 0.0, 0.0, 0.0])
# upper_bound = jnp.array([1.0, 1.0, 1.0, 100.0])

# filter_size = 3


# @jax.jit
# def blur_intermediate_sample_func(key, latent_rgbd, likelihood_args):
#     color_variance = likelihood_args["color_variance_0"]
#     depth_variance = likelihood_args["depth_variance_0"]
#     outlier_probability = likelihood_args["outlier_probability_0"]

#     ###########
#     @functools.partial(
#         jnp.vectorize,
#         signature="(m),(2)->(4)",
#         excluded=(
#             2,
#             3,
#             4,
#         ),
#     )
#     def per_pixel(
#         ij,
#         key,
#         latent_rgbd_padded,
#         log_kernel,
#         filter_size,
#     ):
#         latent_rgb_padded_window = jax.lax.dynamic_slice(
#             latent_rgbd_padded,
#             (ij[0], ij[1], 0),
#             (2 * filter_size + 1, 2 * filter_size + 1, 4),
#         )
#         # outliers_padded = jax.random.uniform(
#         #     key, (2 * filter_size + 1, 2 * filter_size + 1, 4), minval=0.0, maxval=1.0
#         # )

#         # no_mesh = latent_rgb_padded_window[..., 3] == 0.0
#         # latent_rgb_padded_window = (
#         #     (1 - no_mesh)[...,None] * latent_rgb_padded_window
#         #     + no_mesh[...,None] * outliers_padded
#         # )

#         index = jax.random.categorical(key, log_kernel.flatten())
#         sampled_rgbd_value = latent_rgb_padded_window.reshape(-1, 4)[index]
#         noisy_rgbd_value = genjax.truncated_normal.sample(
#             key,
#             sampled_rgbd_value,
#             jnp.array([color_variance, color_variance, color_variance, depth_variance]),
#             0.0,
#             1.0,
#         )

#         is_outlier = genjax.bernoulli.sample(
#             key, jax.scipy.special.logit(outlier_probability)
#         )
#         outlier_value = jax.random.uniform(
#             key, (4,), minval=jnp.zeros(4), maxval=jnp.ones(4)
#         )

#         return is_outlier * outlier_value + (1 - is_outlier) * noisy_rgbd_value

#     @jax.jit
#     def likelihood_per_pixel(latent_rgbd: jnp.ndarray, blur):
#         latent_rgbd = (latent_rgbd - lower_bound) / (upper_bound - lower_bound)

#         latent_rgbd_padded = jnp.pad(
#             latent_rgbd,
#             (
#                 (filter_size, filter_size),
#                 (filter_size, filter_size),
#                 (0, 0),
#             ),
#             mode="edge",
#         )
#         jj, ii = jnp.meshgrid(
#             jnp.arange(latent_rgbd.shape[1]), jnp.arange(latent_rgbd.shape[0])
#         )
#         indices = jnp.stack([ii, jj], axis=-1)

#         log_kernel = log_gaussian_kernel(2 * filter_size + 1, blur)

#         keys = jax.random.split(key, (latent_rgbd.shape[0], latent_rgbd.shape[1]))
#         values = per_pixel(
#             indices,
#             keys,
#             latent_rgbd_padded,
#             log_kernel,
#             filter_size,
#         )
#         return values * (upper_bound - lower_bound) + lower_bound

#     # noisy_image = genjax.truncated_normal.sample(key, latent_rgbd, color_variance, lower_bound, upper_bound)
#     # return noisy_image
#     return likelihood_per_pixel(latent_rgbd, likelihood_args["blur"])


# @jax.jit
# def blur_intermediate_likelihood_func(observed_rgbd, latent_rgbd, likelihood_args):
#     # k = likelihood_args["k"].const
#     color_variance = likelihood_args["color_variance_0"]
#     depth_variance = likelihood_args["depth_variance_0"]
#     outlier_probability = likelihood_args["outlier_probability_0"]
#     rows = likelihood_args["rows"]
#     cols = likelihood_args["cols"]

#     ###########
#     @functools.partial(
#         jnp.vectorize,
#         signature="(m)->()",
#         excluded=(
#             1,
#             2,
#             3,
#             4,
#         ),
#     )
#     def per_pixel(
#         ij,
#         observed_rgbd,
#         latent_rgbd_padded,
#         log_kernel,
#         filter_size,
#     ):
#         latent_rgb_padded_window = jax.lax.dynamic_slice(
#             latent_rgbd_padded,
#             (ij[0], ij[1], 0),
#             (2 * filter_size + 1, 2 * filter_size + 1, 4),
#         )

#         scores_inlier = genjax.truncated_normal.logpdf(
#             observed_rgbd[ij[0], ij[1], :],
#             latent_rgb_padded_window,
#             jnp.array([color_variance, color_variance, color_variance, depth_variance]),
#             0.0 - 0.00001,
#             1.0 + 0.00001,
#         ).sum(-1) + jnp.where(latent_rgb_padded_window[..., 3] == 0.0, -jnp.inf, 0.0)

#         # no_mesh = latent_rgb_padded_window[..., 3] == 0.
#         # outlier_probability_adjusted = (no_mesh) * 1.0 + (1 - no_mesh) * outlier_probability

#         score_mixed = jax.nn.logsumexp(scores_inlier + log_kernel)

#         final_score = jnp.logaddexp(
#             score_mixed + jnp.log(1.0 - outlier_probability),
#             jnp.log(outlier_probability),
#         )
#         return final_score

#     @jax.jit
#     def likelihood_per_pixel(
#         observed_rgbd: jnp.ndarray, latent_rgbd: jnp.ndarray, blur
#     ):
#         observed_rgbd = (observed_rgbd - lower_bound) / (upper_bound - lower_bound)
#         latent_rgbd = (latent_rgbd - lower_bound) / (upper_bound - lower_bound)

#         latent_rgbd_padded = jnp.pad(
#             latent_rgbd,
#             (
#                 (filter_size, filter_size),
#                 (filter_size, filter_size),
#                 (0, 0),
#             ),
#             mode="edge",
#         )

#         # jj, ii = jnp.meshgrid(
#         #     jnp.arange(observed_rgbd.shape[1]), jnp.arange(observed_rgbd.shape[0])
#         # )
#         indices = jnp.stack([rows, cols], axis=-1)
#         log_kernel = log_gaussian_kernel(2 * filter_size + 1, blur)

#         log_probabilities = per_pixel(
#             indices,
#             observed_rgbd,
#             latent_rgbd_padded,
#             log_kernel,
#             filter_size,
#         )
#         return log_probabilities

#     scores = likelihood_per_pixel(observed_rgbd, latent_rgbd, likelihood_args["blur"])

#     # score = genjax.truncated_normal.logpdf(observed_rgbd, latent_rgbd, color_variance, lower_bound, upper_bound)[...,:3].sum()
#     # score = (jax.nn.logsumexp(pixelwise_score) - jnp.log(pixelwise_score.size)) * k
#     score = scores.sum()

#     pixelwise_score = jnp.zeros((observed_rgbd.shape[0], observed_rgbd.shape[1]))
#     pixelwise_score = pixelwise_score.at[rows, cols].set(scores)

#     return {
#         "score": score,
#         "scores": scores,
#         "observed_color_space_d": observed_rgbd,
#         "latent_color_space_d": latent_rgbd,
#         "pixelwise_score": pixelwise_score,
#     }
