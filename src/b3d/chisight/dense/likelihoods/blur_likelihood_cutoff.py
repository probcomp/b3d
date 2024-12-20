import functools

import jax
import jax.numpy as jnp


def log_gaussian_kernel(size: int, sigma: float) -> jnp.ndarray:
    """Creates a 2D Gaussian kernel."""
    ax = jnp.arange(-size // 2 + 1.0, size // 2 + 1.0)
    xx, yy = jnp.meshgrid(ax, ax)
    kernel = -(xx**2 + yy**2) / (2.0 * sigma**2)
    kernel = kernel - jax.nn.logsumexp(kernel)
    return kernel


filter_half_width = 4
filter_size = filter_half_width


@jax.jit
def blur_intermediate_likelihood_func(observed_rgbd, likelihood_args):
    # fx = likelihood_args["fx"]
    # fy = likelihood_args["fy"]
    # cx = likelihood_args["cx"]
    # cy = likelihood_args["cy"]

    color_variance = likelihood_args["color_variance_0"]
    depth_variance = likelihood_args["depth_variance_0"]
    outlier_probability = likelihood_args["outlier_probability_0"]
    rows = likelihood_args["rows"]
    cols = likelihood_args["cols"]
    latent_rgbd = likelihood_args["latent_rgbd"]

    ###########
    @functools.partial(
        jnp.vectorize,
        signature="(m)->(),()",
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
        blur,
        filter_size,
    ):
        log_kernel = log_gaussian_kernel(2 * filter_size + 1, blur)

        latent_rgb_padded_window = jax.lax.dynamic_slice(
            latent_rgbd_padded,
            (ij[0], ij[1], 0),
            (2 * filter_size + 1, 2 * filter_size + 1, 4),
        )

        error = jnp.abs(latent_rgb_padded_window - observed_rgbd[ij[0], ij[1], :])
        inlier = jnp.all(
            error
            < jnp.array(
                [color_variance, color_variance, color_variance, depth_variance]
            ),
            axis=-1,
        )

        valid = latent_rgb_padded_window[..., 3] != 0.0

        scores_inlier = (valid * inlier) * 5.00 + (valid * ~inlier) * -jnp.inf

        # scores_inlier = genjax.truncated_normal.logpdf(
        #     observed_rgbd[ij[0], ij[1], :],
        #     latent_rgb_padded_window,
        #     jnp.array([color_variance, color_variance, color_variance, depth_variance]),
        #     0.0 - 0.00001,
        #     1.0 + 0.00001,
        # ).sum(-1)

        # no_mesh = latent_rgb_padded_window[..., 3] == 0.
        # outlier_probability_adjusted = (no_mesh) * 1.0 + (1 - no_mesh) * outlier_probability

        score_mixed = jax.nn.logsumexp(scores_inlier + log_kernel)

        final_score = jnp.logaddexp(
            score_mixed + jnp.log(1.0 - outlier_probability),
            jnp.log(outlier_probability),
        )
        return score_mixed, final_score

    latent_rgbd_padded = jnp.pad(
        latent_rgbd,
        (
            (filter_size, filter_size),
            (filter_size, filter_size),
            (0, 0),
        ),
        mode="edge",
    )
    # jj, ii = jnp.meshgrid(
    #     jnp.arange(observed_rgbd.shape[1]), jnp.arange(observed_rgbd.shape[0])
    # )
    indices = jnp.stack([rows, cols], axis=-1)

    blur = likelihood_args["blur"]
    _, scores = per_pixel(
        indices,
        observed_rgbd,
        latent_rgbd_padded,
        blur,
        filter_size,
    )

    # score = genjax.truncated_normal.logpdf(observed_rgbd, latent_rgbd, color_variance, lower_bound, upper_bound)[...,:3].sum()
    # score = (jax.nn.logsumexp(pixelwise_score) - jnp.log(pixelwise_score.size)) * k
    score = scores.sum()

    pixelwise_score = jnp.zeros((observed_rgbd.shape[0], observed_rgbd.shape[1]))
    pixelwise_score = pixelwise_score.at[rows, cols].set(scores)

    return {
        "score": score,
        "scores": scores,
        "observed_color_space_d": observed_rgbd,
        "latent_color_space_d": latent_rgbd,
        "pixelwise_score": pixelwise_score,
    }
