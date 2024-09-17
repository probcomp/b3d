from functools import partial

import b3d
import jax
import jax.numpy as jnp
from b3d.chisight.dense.likelihoods.image_likelihoods import (
    gaussian_iid_pix_likelihood,
    kray_likelihood_intermediate,
    threedp3_gmm_likelihood,
)
from image_likelihood_tests import (
    mug_posterior_peakiness_samples,
    test_distance_invariance,
    test_mode_volume,
    test_noise_invariance,
    test_resolution_invariance,
)


# set up latent image as likelihood arg
def rgbd_latent_likelihood(likelihood, observed_rgbd, rendered_rgbd, likelihood_args):
    likelihood_args["latent_rgbd"] = rendered_rgbd
    return likelihood(observed_rgbd, likelihood_args)


rgbd_latent_model_vec = jax.vmap(rgbd_latent_likelihood, in_axes=(None, None, 0, None))


@partial(jax.jit, static_argnums=(0,))
def rgbd_latent_model_jit(likelihood, observed_rgbd, rendered_rgbd, likelihood_args):
    return rgbd_latent_model_vec(
        likelihood, observed_rgbd, rendered_rgbd, likelihood_args
    )


gaussian_iid_pix_likelihood_vec = partial(
    rgbd_latent_model_jit, gaussian_iid_pix_likelihood
)
threedp3_gmm_likelihood_vec = partial(rgbd_latent_model_jit, threedp3_gmm_likelihood)
kray_likelihood_intermediate_vec = partial(
    rgbd_latent_model_jit, kray_likelihood_intermediate
)

# gaussian iid args
gaussian_iid_pix_likelihood_args = {
    "rgb_tolerance": 50.0,
    "depth_tolerance": 0.025,
    "outlier_prob": 0.01,
}

# GMM args
threedp3_gmm_likelihood_args = {
    "variance": 0.1,
    "outlier_prob": 0.1,
    "outlier_volume": 10**3,
    "filter_size": 3,
    "intrinsics": (
        100,
        100,
        200.0,
        200.0,
        50.0,
        50.0,
        0.01,
        10.0,
    ),
}

# ray-tracing likelihood args
kray_likelihood_args = {
    "color_tolerance": 50.0,
    "depth_tolerance": 0.01,
    "inlier_score": 25,  # 2.5,
    "outlier_prob": 0.005,
    "multiplier": 10.0,
    "intrinsics": (
        100,
        100,
        200.0,
        200.0,
        50.0,
        50.0,
        0.01,
        10.0,
    ),
}


def get_renderer_original():
    h, w, fx, fy, cx, cy = 50, 50, 100.0, 100.0, 25.0, 25.0
    scale = 9
    renderer = b3d.RendererOriginal(
        scale * h, scale * w, scale * fx, scale * fy, scale * cx, scale * cy, 0.01, 10.0
    )
    return renderer


models = [
    gaussian_iid_pix_likelihood_vec,
    threedp3_gmm_likelihood_vec,
    kray_likelihood_intermediate_vec,
]
model_names = ["i.i.d Normal", "3DP3 GMM", "Surface-ray"]
model_args = [
    gaussian_iid_pix_likelihood_args,
    threedp3_gmm_likelihood_args,
    kray_likelihood_args,
]

# set up renderer specs
h, w, fx, fy, cx, cy = 50, 50, 100.0, 100.0, 25.0, 25.0
scale = 9
renderer = b3d.RendererOriginal(
    scale * h, scale * w, scale * fx, scale * fy, scale * cx, scale * cy, 0.01, 10.0
)
scales = jnp.array([1, 3, 5, 7, 9])
renderers_scaled_resolution = [
    b3d.RendererOriginal(
        scale * h, scale * w, scale * fx, scale * fy, scale * cx, scale * cy, 0.01, 10.0
    )
    for scale in scales
]

# run tests
test_distance_invariance(renderer, models, model_names, model_args)
test_resolution_invariance(renderers_scaled_resolution, models, model_names, model_args)
test_noise_invariance(renderer, models, model_names, model_args)
test_mode_volume(renderer, models, model_names, model_args)
mug_posterior_peakiness_samples(renderer, models, model_names, model_args)
