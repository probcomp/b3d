import functools

import genjax
import jax
import jax.numpy as jnp

from b3d.modeling_utils import get_interpenetration


def log_gaussian_kernel(size: int, sigma: float) -> jnp.ndarray:
    """Creates a 2D Gaussian kernel."""
    ax = jnp.arange(-size // 2 + 1.0, size // 2 + 1.0)
    xx, yy = jnp.meshgrid(ax, ax)
    kernel = -(xx**2 + yy**2) / (2.0 * sigma**2)
    kernel = kernel - jax.nn.logsumexp(kernel)
    return kernel


lower_bound = jnp.array([0.0, 0.0, 0.0, 0.0])
upper_bound = jnp.array([1.0, 1.0, 1.0, 100.0])

filter_size = 3


@jax.jit
def likelihood_func(observed_rgbd, likelihood_args):
    latent_rgbd = likelihood_args["latent_rgbd"]
    color_variance = likelihood_args["color_noise_variance"]
    depth_variance = likelihood_args["depth_noise_variance"]
    outlier_probability = likelihood_args["outlier_probability"]
    rows = likelihood_args["rows"]
    cols = likelihood_args["cols"]

    @functools.partial(
        jnp.vectorize,
        signature="(m)->()",
        excluded=(
            1,
            2,
            3,
            4,
        ),
    )
    def per_pixel(
        ij,
        observed_rgbd,
        latent_rgbd_padded,
        log_kernel,
        filter_size,
    ):
        latent_rgb_padded_window = jax.lax.dynamic_slice(
            latent_rgbd_padded,
            (ij[0], ij[1], 0),
            (2 * filter_size + 1, 2 * filter_size + 1, 4),
        )

        scores_inlier = genjax.truncated_normal.logpdf(
            observed_rgbd[ij[0], ij[1], :],
            latent_rgb_padded_window,
            jnp.array([color_variance, color_variance, color_variance, depth_variance]),
            0.0 - 0.00001,
            1.0 + 0.00001,
        ).sum(-1) + jnp.where(latent_rgb_padded_window[..., 3] == 0.0, -jnp.inf, 0.0)

        score_mixed = jax.nn.logsumexp(scores_inlier + log_kernel)

        final_score = jnp.logaddexp(
            score_mixed + jnp.log(1.0 - outlier_probability),
            jnp.log(outlier_probability),
        )
        return final_score

    @jax.jit
    def likelihood_per_pixel(
        observed_rgbd: jnp.ndarray, latent_rgbd: jnp.ndarray, blur
    ):
        observed_rgbd = (observed_rgbd - lower_bound) / (upper_bound - lower_bound)
        latent_rgbd = (latent_rgbd - lower_bound) / (upper_bound - lower_bound)

        latent_rgbd_padded = jnp.pad(
            latent_rgbd,
            (
                (filter_size, filter_size),
                (filter_size, filter_size),
                (0, 0),
            ),
            mode="edge",
        )

        indices = jnp.stack([rows, cols], axis=-1)
        log_kernel = log_gaussian_kernel(2 * filter_size + 1, blur)

        log_probabilities = per_pixel(
            indices,
            observed_rgbd,
            latent_rgbd_padded,
            log_kernel,
            filter_size,
        )
        return log_probabilities

    scores = likelihood_per_pixel(observed_rgbd, latent_rgbd, likelihood_args["blur"])

    valid_window = latent_rgbd[..., 0:3].sum(axis=2) > 0.0  # latent_rgbd[..., 3] > 0.0
    if likelihood_args["masked"].unwrap():
        observed_window = observed_rgbd[..., 0:3].sum(axis=2) > 0.0
        invalid_window = jnp.multiply(observed_window, ~valid_window)
        near = 0.001
        far = jnp.inf
    else:
        invalid_window = ~valid_window
        far = 1
        near = 0

    pixelwise_score = (
        jax.scipy.stats.laplace.logpdf(
            observed_rgbd,
            latent_rgbd,
            jnp.array([color_variance, color_variance, color_variance, depth_variance]),
        ).sum(-1)
        * valid_window
        - (jnp.log(1.0 / 1.0**3) + jnp.log(far - near)) * invalid_window
    )

    pixelwise_score = jnp.logaddexp(
        pixelwise_score + jnp.log(1.0 - outlier_probability),
        jnp.log(outlier_probability),
    )

    score = pixelwise_score.sum()

    if likelihood_args["check_interp"].unwrap():
        interpeneration = get_interpenetration(
            likelihood_args["scene_mesh"], likelihood_args["num_mc_sample"].unwrap()
        )
        interpeneration_score = (
            likelihood_args["interp_penalty"].unwrap() * interpeneration
        )
        # jax.debug.print("interpeneration_score: {v}", v=interpeneration_score)
        score -= interpeneration_score

    return {
        "score": score,
        "pixelwise_score": pixelwise_score,
        "latent_rgbd": latent_rgbd,
    }
