import genjax
import jax
import jax.numpy as jnp

import b3d

# # Version 2 [approximate this with a mean]:
# # `pts` is the set of all points projecting to one pixel (i, j)
# # `nonregistration_prob[pt]` = probability a point is not registered, if it is the only one observed (nonregistration_prob = "outlier prob")
# # `color[pt]` = RGB or D value for the point
# # 1. Compute overall_p_nonregistered = prod_{pt} nonregistration_prob[pt].  [Set to 1.0 if pts is empty.]
# # 2. Compute the mean of all the colors for each point, where the color for `pt` is weighted proportionally to (1 - nonregistration_prob[pt])
# # 3. Sample from a mixture of [1] a uniform, with probability `nonregistration_prob[pt]`, and [2] a laplace around the mean color


@jax.jit
def sample_func(key, args):
    transformed_points = args["pose"].apply(args["vertices"])
    pixels = jnp.rint(
        b3d.xyz_to_pixel_coordinates(
            transformed_points, args["fx"], args["fy"], args["cx"], args["cy"]
        )
        - 0.5
    ).astype(jnp.int32)

    latent_image_sum = jnp.zeros(
        (args["image_height"].unwrap(), args["image_width"].unwrap(), 4)
    )
    latent_image_sum = latent_image_sum.at[pixels[..., 0], pixels[..., 1], :3].add(
        args["colors"] * (1 - args["color_outlier_probability"])[:, None]
    )
    latent_image_sum = latent_image_sum.at[pixels[..., 0], pixels[..., 1], 3].add(
        transformed_points[..., 2] * (1 - args["depth_outlier_probability"])
    )

    projected_points_count = jnp.zeros(
        (args["image_height"].unwrap(), args["image_width"].unwrap())
    )
    projected_points_count = projected_points_count.at[
        pixels[..., 0], pixels[..., 1]
    ].add(1)

    non_registration_probability = jnp.ones(
        (args["image_height"].unwrap(), args["image_width"].unwrap())
    )
    non_registration_probability = non_registration_probability.at[
        pixels[..., 0], pixels[..., 1]
    ].multiply(args["color_outlier_probability"])

    latent_image_mean = latent_image_sum / (projected_points_count[..., None] + 1e-10)

    is_outlier_pixel = (
        jax.random.uniform(key, non_registration_probability.shape)
        < non_registration_probability
    )
    variances = jnp.array(
        [
            args["color_variance"],
            args["color_variance"],
            args["color_variance"],
            args["depth_variance"],
        ]
    )
    latent_image_noised = genjax.laplace.sample(key, latent_image_mean, variances)
    latent_image_uniform = jax.random.uniform(key, latent_image_mean.shape) * jnp.array(
        [1.0, 1.0, 1.0, 5.0]
    )
    return (
        latent_image_noised * ~is_outlier_pixel[..., None]
        + latent_image_uniform * is_outlier_pixel[..., None]
    )


@jax.jit
def likelihood_func(observed_rgbd, args):
    transformed_points = args["pose"].apply(args["vertices"])
    pixels = jnp.rint(
        b3d.xyz_to_pixel_coordinates(
            transformed_points, args["fx"], args["fy"], args["cx"], args["cy"]
        )
    ).astype(jnp.int32)

    latent_image_sum = jnp.zeros(
        (args["image_height"].unwrap(), args["image_width"].unwrap(), 4)
    )
    latent_image_sum = latent_image_sum.at[pixels[..., 0], pixels[..., 1], :3].add(
        args["colors"] * (1 - args["color_outlier_probability"])[:, None]
    )
    latent_image_sum = latent_image_sum.at[pixels[..., 0], pixels[..., 1], 3].add(
        transformed_points[..., 2] * (1 - args["depth_outlier_probability"])
    )

    projected_points_count = jnp.zeros(
        (args["image_height"].unwrap(), args["image_width"].unwrap())
    )
    projected_points_count = projected_points_count.at[
        pixels[..., 0], pixels[..., 1]
    ].add(1)

    non_registration_probability = jnp.ones(
        (args["image_height"].unwrap(), args["image_width"].unwrap())
    )
    non_registration_probability = non_registration_probability.at[
        pixels[..., 0], pixels[..., 1]
    ].multiply(args["color_outlier_probability"])

    latent_image_mean = latent_image_sum / (projected_points_count[..., None] + 1e-10)

    variances = jnp.array(
        [
            args["color_variance"],
            args["color_variance"],
            args["color_variance"],
            args["depth_variance"],
        ]
    )

    # pixel_probability = jax.scipy.stats.laplace.logpdf(
    #     observed_rgbd, latent_image_mean, variances
    # ) + jnp.log(1 - non_registration_probability)[...,None]

    pixel_probability = jnp.logaddexp(
        jax.scipy.stats.laplace.logpdf(observed_rgbd, latent_image_mean, variances)
        + jnp.log(1 - non_registration_probability)[..., None],
        (jnp.log(non_registration_probability) + jnp.log(1 / 1.0**3))[..., None]
        * jnp.ones_like(observed_rgbd),
    )

    return {
        "score": pixel_probability.sum(),
        "scores": pixel_probability,
        "latent_rgbd": latent_image_mean,
        "transformed_points": transformed_points,
        "observed_rgbd_masked": observed_rgbd[pixels[..., 0], pixels[..., 1]],
    }


aggregate_mean_image_kerenel_likelihood_func_and_sample_func = (
    likelihood_func,
    sample_func,
)
