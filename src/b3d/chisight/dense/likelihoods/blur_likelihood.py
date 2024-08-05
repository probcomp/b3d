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


@jax.jit
def blur_intermediate_likelihood_func(observed_rgbd, likelihood_args):
    fx = likelihood_args["fx"]
    fy = likelihood_args["fy"]
    cx = likelihood_args["cx"]
    cy = likelihood_args["cy"]

    rasterize_results = likelihood_args["rasterize_results"]
    triangle_collision_indices_image = rasterize_results[..., 3].astype(jnp.int32) - 1
    valid = triangle_collision_indices_image >= 0

    latent_rgbd = likelihood_args["latent_rgbd"]
    mesh_transformed = likelihood_args["scene_mesh"]

    far_plane_xyz = b3d.xyz_from_depth(
        jnp.ones((latent_rgbd.shape[0], latent_rgbd.shape[1])), fx, fy, cx, cy
    )
    vertices_mean_image = (
        valid[..., None]
        * mesh_transformed.vertices[
            mesh_transformed.faces[triangle_collision_indices_image]
        ].mean(axis=-2)
        + ~valid[..., None] * far_plane_xyz
    )
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
        length = (2 * filter_half_width + 1, 2 * filter_half_width + 1, 3)
        vertices_mean_window = jax.lax.dynamic_slice(
            vertices_mean_image, lower_indices, length
        )
        colors_window = jax.lax.dynamic_slice(colors_image, lower_indices, length)
        valid_window = jax.lax.dynamic_slice(valid, lower_indices[:2], length[:2])

        observed_color = observed_rgbd[ij[0], ij[1], :3]
        observed_depth = observed_rgbd[ij[0], ij[1], 3]
        depth_probability = jax.scipy.stats.norm.logpdf(
            observed_depth,
            vertices_mean_window[..., 2],
            depth_variance,
        )
        color_probability = jax.scipy.stats.norm.logpdf(
            observed_color,
            colors_window,
            color_variance,
        )
        probability = color_probability.sum(-1) + depth_probability
        scores_inlier = valid_window * probability

        pixel_coordinates = b3d.xyz_to_pixel_coordinates(
            vertices_mean_window, fx, fy, cx, cy
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
    }
