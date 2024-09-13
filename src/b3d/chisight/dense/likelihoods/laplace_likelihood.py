import jax
import jax.numpy as jnp


@jax.jit
def likelihood_func(observed_rgbd, likelihood_args):
    latent_rgbd = likelihood_args["latent_rgbd"]
    color_variance = likelihood_args["color_noise_variance"]
    depth_variance = likelihood_args["depth_noise_variance"]
    outlier_probability = likelihood_args["outlier_probability"]

    valid_window = latent_rgbd[..., 3] > 0.0
    pixelwise_score = (
        jax.scipy.stats.laplace.logpdf(
            observed_rgbd,
            latent_rgbd,
            jnp.array([color_variance, color_variance, color_variance, depth_variance]),
        ).sum(-1)
        * valid_window
        + jnp.log(1.0 / 1.0) * ~valid_window
    )

    pixelwise_score = jnp.logaddexp(
        pixelwise_score + jnp.log(1.0 - outlier_probability),
        jnp.log(outlier_probability),
    )

    score = pixelwise_score.sum()
    if "interpenetration_penalty" in likelihood_args.keys():
        interpenetration_penalty = likelihood_args["interpenetration_penalty"]
        interpeneration = likelihood_args["object_interpenetration"]
        interpeneration_score = interpenetration_penalty * interpeneration.sum()
        score -= interpeneration_score

    return {
        "score": score,
        "pixelwise_score": pixelwise_score,
        "latent_rgbd": latent_rgbd,
    }
