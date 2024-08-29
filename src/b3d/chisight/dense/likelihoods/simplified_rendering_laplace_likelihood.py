import jax
import jax.numpy as jnp

import b3d.chisight.dense.dense_model

b3d.reload(b3d.chisight.dense.dense_model)


@jax.jit
def simplified_rendering_laplace_likelihood(observed_rgbd, likelihood_args):
    scene_mesh = likelihood_args["scene_mesh"]
    transformed_points = scene_mesh.vertices
    template_colors = scene_mesh.vertex_attributes

    fx, fy, cx, cy = (
        likelihood_args["fx"],
        likelihood_args["fy"],
        likelihood_args["cx"],
        likelihood_args["cy"],
    )
    projected_pixels = jnp.rint(
        b3d.xyz_to_pixel_coordinates(transformed_points, fx, fy, cx, cy)
    ).astype(jnp.int32)

    outlier_probability_per_vertex = likelihood_args["outlier_probability"]

    corresponding_observed_rgbd = observed_rgbd[
        projected_pixels[..., 0], projected_pixels[..., 1]
    ]

    color_probability = jax.scipy.stats.laplace.logpdf(
        corresponding_observed_rgbd[..., :3],
        template_colors,
        likelihood_args["color_noise_variance"],
    ).sum(-1)

    depth_probability = jax.scipy.stats.laplace.logpdf(
        corresponding_observed_rgbd[..., 3],
        transformed_points[..., 2],
        likelihood_args["depth_noise_variance"],
    )

    color_probability_outlier_adjusted = jnp.logaddexp(
        color_probability + jnp.log(1 - outlier_probability_per_vertex),
        jnp.log(outlier_probability_per_vertex) + jnp.log(1.0 / 1.0),
    )

    depth_probability_outlier_adjusted = jnp.logaddexp(
        depth_probability + jnp.log(1 - outlier_probability_per_vertex),
        jnp.log(outlier_probability_per_vertex) + jnp.log(1.0 / 1.0),
    )

    lmbda = 0.5
    scores = (
        lmbda * color_probability_outlier_adjusted
        + (1.0 - lmbda) * depth_probability_outlier_adjusted
    )

    # Visualization
    latent_rgbd = jnp.zeros_like(observed_rgbd)
    latent_rgbd = latent_rgbd.at[
        projected_pixels[..., 0], projected_pixels[..., 1], :3
    ].set(template_colors)
    latent_rgbd = latent_rgbd.at[
        projected_pixels[..., 0], projected_pixels[..., 1], 3
    ].set(transformed_points[..., 2])

    return {
        "score": scores.sum(),
        "scores": scores,
        "latent_rgbd": latent_rgbd,
        "color_probability_outlier_adjusted": color_probability_outlier_adjusted,
        "depth_probability_outlier_adjusted": depth_probability_outlier_adjusted,
        "scene_mesh": scene_mesh,
        "model_rgbd": jnp.concatenate(
            [template_colors, transformed_points[..., 2:]], -1
        ),
        "corresponding_observed_rgbd": corresponding_observed_rgbd,
        "pixelwise_score": jnp.zeros_like(observed_rgbd[..., 0]),
    }
