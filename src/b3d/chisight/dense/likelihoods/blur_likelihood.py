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

    rasterize_results = likelihood_args["rasterize_results"]
    latent_rgbd = likelihood_args["latent_rgbd"]

    mesh_transformed = likelihood_args["scene_mesh"]
    vertices_mean_image = (
        mesh_transformed.vertices[
            mesh_transformed.faces[rasterize_results[..., 3].astype(jnp.int32) - 1]
        ].mean(axis=-2)
    ) * rasterize_results[..., 2:3]
    colors_image = latent_rgbd[..., :3]

    color_variance = likelihood_args["color_variance_0"]
    depth_variance = likelihood_args["depth_variance_0"]
    outlier_probability = likelihood_args["outlier_probability_0"]
    blur = likelihood_args["blur"]

    rows = likelihood_args["rows"]
    cols = likelihood_args["cols"]

    ###########
    @functools.partial(
        jnp.vectorize,
        signature="(m)->()",
    )
    def score_pixel(ij):
        log_kernel = log_gaussian_kernel(2 * filter_size + 1, blur)

        lower_indices = (ij[0] - filter_half_width, ij[1] - filter_half_width, 0)
        vertices_mean_window = jax.lax.dynamic_slice(
            vertices_mean_image,
            lower_indices,
            (2 * filter_half_width + 1, 2 * filter_half_width + 1, 3),
        )
        colors_window = jax.lax.dynamic_slice(
            colors_image,
            lower_indices,
            (2 * filter_half_width + 1, 2 * filter_half_width + 1, 3),
        )

        latent_rgbd_padded_window = jnp.concatenate(
            [colors_window, vertices_mean_window[..., 2:3]], axis=-1
        )

        # Case 1
        scores_inlier = (
            jax.scipy.stats.norm.logpdf(
                observed_rgbd[ij[0], ij[1], :],
                latent_rgbd_padded_window,
                jnp.array(
                    [color_variance, color_variance, color_variance, depth_variance]
                ),
            )
        ).sum(-1)
        # # Case 2
        # error = jnp.abs(latent_rgbd_padded_window - observed_rgbd[ij[0], ij[1], :])
        # inlier = jnp.all(
        #     error
        #     < jnp.array(
        #         [color_variance, color_variance, color_variance, depth_variance]
        #     ),
        #     axis=-1,
        # )

        # valid = latent_rgbd_padded_window[..., 3] != 0.0

        # scores_inlier = (valid * inlier) * 5.00 + (valid * ~inlier) * -jnp.inf

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
        return final_score

    indices = jnp.stack([rows, cols], axis=-1)
    scores = score_pixel(indices)

    pixelwise_score = jnp.zeros((observed_rgbd.shape[0], observed_rgbd.shape[1]))
    pixelwise_score = pixelwise_score.at[rows, cols].set(scores)

    return {
        "score": scores.sum(),
        "pixelwise_score": pixelwise_score,
    }
