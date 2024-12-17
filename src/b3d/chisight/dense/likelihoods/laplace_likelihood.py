import jax
import jax.numpy as jnp

from b3d.modeling_utils import get_interpenetration


@jax.jit
def likelihood_func(observed_rgbd, likelihood_args):
    latent_rgbd = likelihood_args["latent_rgbd"]
    color_variance = likelihood_args["color_noise_variance"]
    depth_variance = likelihood_args["depth_noise_variance"]
    outlier_probability = likelihood_args["outlier_probability"]

    valid_window = latent_rgbd[..., 0:3].sum(axis=2) > 0.0  # latent_rgbd[..., 3] > 0.0
    if likelihood_args["masked"].unwrap():
        observed_window = observed_rgbd[..., 0:3].sum(axis=2) > 0.0
        invalid_window = jnp.multiply(observed_window, ~valid_window)
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

    if likelihood_args["check_interp"].unwrap():
        interpeneration = get_interpenetration(
            likelihood_args["scene_mesh"], likelihood_args["num_mc_sample"].unwrap()
        )
        interpeneration_score = (
            likelihood_args["interp_penalty"].unwrap() * interpeneration
        )
        # jax.debug.print("interpeneration_score: {v}", v=interpeneration_score)
        score -= interpeneration_score

    return {
        "score": score,
        "pixelwise_score": pixelwise_score,
        "latent_rgbd": latent_rgbd,
    }
