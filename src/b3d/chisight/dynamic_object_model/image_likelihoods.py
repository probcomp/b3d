import jax
import jax.numpy as jnp

import b3d


@jax.jit
def simple_projection_image_likelihood_function(observed_rgbd, args):
    transformed_points = args["pose"].apply(args["vertices"])

    projected_pixel_coordinates = jnp.rint(
        b3d.xyz_to_pixel_coordinates(
            transformed_points, args["fx"], args["fy"], args["cx"], args["cy"]
        )
    ).astype(jnp.int32)

    observed_rgbd_masked = observed_rgbd[
        projected_pixel_coordinates[..., 0], projected_pixel_coordinates[..., 1]
    ]

    color_visible_branch_score = jax.scipy.stats.laplace.logpdf(
        observed_rgbd_masked[..., :3], args["colors"], args["color_scale"]
    ).sum(axis=-1)
    color_not_visible_score = jnp.log(1 / 1.0**3)
    color_score = jnp.logaddexp(
        color_visible_branch_score + jnp.log(args["is_visible_probabilities"]),
        color_not_visible_score + jnp.log(1 - args["is_visible_probabilities"]),
    )

    depth_visible_branch_score = jax.scipy.stats.laplace.logpdf(
        observed_rgbd_masked[..., 3], transformed_points[..., 2], args["depth_scale"]
    )
    depth_not_visible_score = jnp.log(1 / 1.0)
    _depth_score = jnp.logaddexp(
        depth_visible_branch_score + jnp.log(args["is_visible_probabilities"]),
        depth_not_visible_score + jnp.log(1 - args["is_visible_probabilities"]),
    )
    is_depth_non_return = observed_rgbd_masked[..., 3] < 0.0001

    non_return_probability = 0.05
    depth_score = jnp.where(
        is_depth_non_return, jnp.log(non_return_probability), _depth_score
    )

    lmbda = 0.5
    scores = lmbda * color_score + (1.0 - lmbda) * depth_score

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
        "pixel_coordinates": projected_pixel_coordinates,
        "transformed_points": transformed_points,
        "observed_rgbd_masked": observed_rgbd_masked,
        "latent_rgbd": latent_rgbd,
    }
