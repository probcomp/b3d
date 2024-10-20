import functools

import jax
import jax.numpy as jnp

import b3d


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
    fx = likelihood_args["fx"]
    fy = likelihood_args["fy"]
    cx = likelihood_args["cx"]
    cy = likelihood_args["cy"]

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

        probability = jax.scipy.stats.norm.logpdf(
            observed_rgbd[ij[0], ij[1], :],
            jnp.concatenate([colors_window, vertices_mean_window[..., 2:3]], axis=-1),
            jnp.array([color_variance, color_variance, color_variance, depth_variance]),
        ).sum(-1)
        valid_window = vertices_mean_window[..., 2] != 0.0
        scores_inlier = valid_window * probability

        pixel_coordinates = b3d.xyz_to_pixel_coordinates(
            vertices_mean_window, fx, fy, cx, cy
        )
        pixel_coordinates_baseline = (
            jnp.stack(
                jnp.meshgrid(
                    jnp.arange(-filter_half_width, filter_half_width + 1),
                    jnp.arange(-filter_half_width, filter_half_width + 1),
                ),
                axis=-1,
            )
            + ij
        )

        pixel_coordinates = (
            pixel_coordinates * valid_window[..., None]
            + pixel_coordinates_baseline * ~valid_window[..., None]
        )

        _log_kernel = ((pixel_coordinates - ij) ** 2).sum(-1) * -1.0 / (2.0 * blur**2)
        log_kernel = _log_kernel - jax.nn.logsumexp(_log_kernel)
        log_kernel = log_kernel - jax.nn.logsumexp(log_kernel)

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
        "latent_rgbd": latent_rgbd,
    }
