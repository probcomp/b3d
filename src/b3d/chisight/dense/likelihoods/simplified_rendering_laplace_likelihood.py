import jax
import jax.numpy as jnp

import b3d.chisight.dense.dense_model

b3d.reload(b3d.chisight.dense.dense_model)


@jax.jit
def simplified_rendering_laplace_likelihood(rgbd, likelihood_args):
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

    latent_rgbd = jnp.zeros_like(rgbd)
    latent_rgbd = latent_rgbd.at[
        projected_pixels[..., 0], projected_pixels[..., 1], :3
    ].set(template_colors)
    latent_rgbd = latent_rgbd.at[
        projected_pixels[..., 0], projected_pixels[..., 1], 3
    ].set(transformed_points[..., 2])

    c_outlier_prob = likelihood_args["color_outlier_probability"]
    d_outlier_prob = likelihood_args["depth_outlier_probability"]

    corresponding_observed_rgbd = rgbd[
        projected_pixels[..., 0], projected_pixels[..., 1]
    ]

    color_probability = jax.scipy.stats.laplace.logpdf(
        corresponding_observed_rgbd[..., :3], template_colors, 0.1
    ).sum(-1)
    depth_probability = jax.scipy.stats.laplace.logpdf(
        corresponding_observed_rgbd[..., 3], transformed_points[..., 2], 0.01
    )

    color_probability_outlier_adjusted = jnp.logaddexp(
        color_probability + jnp.log(1 - c_outlier_prob),
        jnp.log(c_outlier_prob) + jnp.log(1.0 / 1.0),
    )
    depth_probability_outlier_adjusted = jnp.logaddexp(
        depth_probability + jnp.log(1 - d_outlier_prob),
        jnp.log(d_outlier_prob) + jnp.log(1.0 / 1.0),
    )

    lmbda = 0.5
    scores = (
        lmbda * color_probability_outlier_adjusted
        + (1.0 - lmbda) * depth_probability_outlier_adjusted
    )

    return {
        "score": scores.sum(),
        "scores": scores,
        "latent_rgbd": latent_rgbd,
        "color_probability_outlier_adjusted": color_probability_outlier_adjusted,
        "depth_probability_outlier_adjusted": depth_probability_outlier_adjusted,
        "scene_mesh": scene_mesh,
        "corresponding_observed_rgbd": corresponding_observed_rgbd,
        "pixelwise_score": jnp.zeros_like(rgbd[..., 0]),
    }
