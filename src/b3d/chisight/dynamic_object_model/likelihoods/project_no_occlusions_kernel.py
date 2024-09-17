import jax
import jax.numpy as jnp

import b3d


@jax.jit
def likelihood_func(observed_rgbd, args):
    transformed_points = args["pose"].apply(args["vertices"])

    projected_pixel_coordinates = jnp.rint(
        b3d.xyz_to_pixel_coordinates(
            transformed_points, args["fx"], args["fy"], args["cx"], args["cy"]
        )
    ).astype(jnp.int32)

    observed_rgbd_masked = observed_rgbd[
        projected_pixel_coordinates[..., 0], projected_pixel_coordinates[..., 1]
    ]

    color_outlier_probabilities = args["color_outlier_probabilities"]
    depth_outlier_probabilities = args["depth_outlier_probabilities"]

    color_probability = jnp.logaddexp(
        jax.scipy.stats.laplace.logpdf(
            observed_rgbd_masked[..., :3], args["colors"], args["color_variance"]
        ).sum(axis=-1)
        + jnp.log(1 - color_outlier_probabilities),
        jnp.log(color_outlier_probabilities)
        + jnp.log(1 / 1.0**3),  # <- log(1) == 0 tho
    )
    depth_probability = jnp.logaddexp(
        jax.scipy.stats.laplace.logpdf(
            observed_rgbd_masked[..., 3],
            transformed_points[..., 2],
            args["depth_variance"],
        )
        + jnp.log(1 - depth_outlier_probabilities),
        jnp.log(depth_outlier_probabilities) + jnp.log(1 / 1.0),
    )

    scores = color_probability + depth_probability

    # Visualization
    latent_rgbd = jnp.zeros_like(observed_rgbd)
    latent_rgbd = latent_rgbd.at[
        projected_pixel_coordinates[..., 0], projected_pixel_coordinates[..., 1], :3
    ].set(args["colors"])
    latent_rgbd = latent_rgbd.at[
        projected_pixel_coordinates[..., 0], projected_pixel_coordinates[..., 1], 3
    ].set(transformed_points[..., 2])

    return {
        "score": scores.sum(),
        "scores": scores,
        "transformed_points": transformed_points,
        "observed_rgbd_masked": observed_rgbd_masked,
        "color_probability": color_probability,
        "depth_probability": depth_probability,
        "latent_rgbd": latent_rgbd,
    }


@jax.jit
def sample_func(key, likelihood_args):
    return jnp.zeros(
        (
            likelihood_args["image_height"].unwrap(),
            likelihood_args["image_width"].unwrap(),
            4,
        )
    )


project_no_occlusions_kernel_likelihood_func_and_sample_func = (
    likelihood_func,
    sample_func,
)
