import jax
import jax.numpy as jnp


@jax.jit
def likelihood_func(observed_rgbd, likelihood_args):
    latent_rgbd = likelihood_args["latent_rgbd"]
    color_variance = likelihood_args["color_noise_variance"]
    depth_variance = likelihood_args["depth_noise_variance"]
    outlier_probability = likelihood_args["outlier_probability"]

    valid_window = latent_rgbd[..., 3] > 0.0
    if "masked" in likelihood_args:
        valid_window_rgb = observed_rgbd[..., 3] > 0.0
        invalid_window = jnp.multiply(valid_window_rgb, ~valid_window)
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
    if "object_interpenetration" in likelihood_args.keys():
        interpenetration_penalty = likelihood_args["interpenetration_penalty"]
        interpeneration = likelihood_args["object_interpenetration"]
        # jax.debug.print("interpeneration: {v}", v=interpeneration)
        interpeneration_score = interpenetration_penalty.const * interpeneration
        # jax.debug.print("interpeneration_score: {v}", v=interpeneration_score)
        score -= interpeneration_score

    return {
        "score": score,
        "pixelwise_score": pixelwise_score,
        "latent_rgbd": latent_rgbd,
    }
