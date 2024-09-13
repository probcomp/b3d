import b3d.chisight.gen3d.image_kernel as image_kernel
import b3d.chisight.gen3d.inference as inference
import b3d.chisight.gen3d.inference_moves as inference_moves
import b3d.chisight.gen3d.transition_kernels as transition_kernels
import jax
import jax.numpy as jnp
import pytest


@pytest.fixture
def hyperparams_and_inference_hyperparams():
    near, far, image_height, image_width = 0.001, 5.0, 480, 640
    img_model = image_kernel.NoOcclusionPerVertexImageKernel(
        near, far, image_height, image_width
    )
    color_transiton_scale = 0.05
    p_resample_color = 0.005

    # This parameter is needed for the inference hyperparameters.
    # See the `InferenceHyperparams` docstring in `inference.py` for details.
    inference_hyperparams = inference.InferenceHyperparams(
        n_poses=6000,
        pose_proposal_std=0.04,
        pose_proposal_conc=1000.0,
        do_stochastic_color_proposals=True,
        prev_color_proposal_laplace_scale=0.001,
        obs_color_proposal_laplace_scale=0.001,
    )

    hyperparams = {
        "pose_kernel": transition_kernels.UniformPoseDriftKernel(max_shift=0.1),
        "color_kernel": transition_kernels.MixtureDriftKernel(
            [
                transition_kernels.LaplaceNotTruncatedColorDriftKernel(
                    scale=color_transiton_scale
                ),
                transition_kernels.UniformDriftKernel(
                    max_shift=0.15, min_val=jnp.zeros(3), max_val=jnp.ones(3)
                ),
            ],
            jnp.array([1 - p_resample_color, p_resample_color]),
        ),
        "visibility_prob_kernel": transition_kernels.DiscreteFlipKernel(
            resample_probability=0.1, support=jnp.array([0.001, 0.999])
        ),
        "depth_nonreturn_prob_kernel": transition_kernels.DiscreteFlipKernel(
            resample_probability=0.1, support=jnp.array([0.001, 0.999])
        ),
        "depth_scale_kernel": transition_kernels.DiscreteFlipKernel(
            resample_probability=0.1,
            support=jnp.array([0.0025, 0.01, 0.02, 0.1, 0.4, 1.0]),
        ),
        "color_scale_kernel": transition_kernels.DiscreteFlipKernel(
            resample_probability=0.1, support=jnp.array([0.05, 0.1, 0.15, 0.3, 0.8])
        ),
        "image_kernel": img_model,
    }
    return hyperparams, inference_hyperparams


def test_visibility_prob_inference(hyperparams_and_inference_hyperparams):
    hyperparams, inference_hyperparams = hyperparams_and_inference_hyperparams

    color_scale = 0.01
    depth_scale = 0.001

    depth_nonreturn_prob_kernel = hyperparams["depth_nonreturn_prob_kernel"]
    visibility_prob_kernel = hyperparams["visibility_prob_kernel"]
    color_kernel = hyperparams["color_kernel"]
    obs_rgbd_kernel = hyperparams["image_kernel"].get_rgbd_vertex_kernel()

    previous_color = jnp.array([0.1, 0.2, 0.3])
    previous_dnrp = depth_nonreturn_prob_kernel.support[0]
    latent_depth = 1.0

    def get_visibility_prob_sample(
        key, observed_rgbd_for_point, previous_visibility_prob
    ):
        _, visibility_prob, _, _, _ = inference_moves._propose_a_points_attributes(
            key,
            observed_rgbd_for_point,
            latent_depth,
            previous_color,
            previous_visibility_prob,
            previous_dnrp,
            depth_nonreturn_prob_kernel,
            visibility_prob_kernel,
            color_kernel,
            obs_rgbd_kernel,
            color_scale,
            depth_scale,
            inference_hyperparams,
        )
        return visibility_prob

    get_visibility_prob_samples = jax.vmap(
        get_visibility_prob_sample, in_axes=(0, None, None)
    )

    keys = jax.random.split(jax.random.PRNGKey(0), 1000)

    # Verify that when the color matches exactly but the depth change drasticaly, the visibility prob switches to low.
    previous_visibility_prob = visibility_prob_kernel.support[-1]
    observed_rgbd_for_this_vertex = jnp.array([0.1, 0.2, 0.3, 4.0])
    visibility_prob_samples = get_visibility_prob_samples(
        keys, observed_rgbd_for_this_vertex, previous_visibility_prob
    )
    assert visibility_prob_samples.mean() < 0.15

    # Verify that when the color matches exactly but the depth change drasticaly, the visibility prob stays low.
    previous_visibility_prob = visibility_prob_kernel.support[0]
    observed_rgbd_for_this_vertex = jnp.array([0.1, 0.2, 0.3, 4.0])
    visibility_prob_samples = get_visibility_prob_samples(
        keys, observed_rgbd_for_this_vertex, previous_visibility_prob
    )
    assert visibility_prob_samples.mean() < 0.03

    # Verify that when the color matches exactly and the depth is close, the visibility prob switches to being high.
    previous_visibility_prob = visibility_prob_kernel.support[0]
    observed_rgbd_for_this_vertex = jnp.array([0.1, 0.2, 0.3, 1.0])
    visibility_prob_samples = get_visibility_prob_samples(
        keys, observed_rgbd_for_this_vertex, previous_visibility_prob
    )
    assert visibility_prob_samples.mean() > 0.85

    # Verify that when the color matches exactly and the depth is close, the visibility prob stays high.
    previous_visibility_prob = visibility_prob_kernel.support[-1]
    observed_rgbd_for_this_vertex = jnp.array([0.1, 0.2, 0.3, 1.0])
    visibility_prob_samples = get_visibility_prob_samples(
        keys, observed_rgbd_for_this_vertex, previous_visibility_prob
    )
    assert visibility_prob_samples.mean() > 0.97


def test_depth_nonreturn_prob_inference(hyperparams_and_inference_hyperparams):
    hyperparams, inference_hyperparams = hyperparams_and_inference_hyperparams

    color_scale = 0.01
    depth_scale = 0.001

    depth_nonreturn_prob_kernel = hyperparams["depth_nonreturn_prob_kernel"]
    visibility_prob_kernel = hyperparams["visibility_prob_kernel"]
    color_kernel = hyperparams["color_kernel"]
    obs_rgbd_kernel = hyperparams["image_kernel"].get_rgbd_vertex_kernel()

    previous_color = jnp.array([0.1, 0.2, 0.3])
    previous_visibility_prob = visibility_prob_kernel.support[-1]
    latent_depth = 1.0

    def get_dnr_prob_sample(key, observed_rgbd_for_point, previous_dnrp):
        _, _, dnr_prob, _, _ = inference_moves._propose_a_points_attributes(
            key,
            observed_rgbd_for_point,
            latent_depth,
            previous_color,
            previous_visibility_prob,
            previous_dnrp,
            depth_nonreturn_prob_kernel,
            visibility_prob_kernel,
            color_kernel,
            obs_rgbd_kernel,
            color_scale,
            depth_scale,
            inference_hyperparams,
        )
        return dnr_prob

    get_dnr_prob_samples = jax.vmap(get_dnr_prob_sample, in_axes=(0, None, None))

    keys = jax.random.split(jax.random.PRNGKey(0), 1000)

    # If depth is nonreturn, the depth nonreturn prob should stay high.
    previous_dnrp = depth_nonreturn_prob_kernel.support[-1]
    observed_rgbd_for_this_vertex = jnp.array([0.1, 0.2, 0.3, 0.0])
    dnr_prob_samples = get_dnr_prob_samples(
        keys, observed_rgbd_for_this_vertex, previous_dnrp
    )
    assert dnr_prob_samples.mean() > 0.95

    # If depth is nonreturn, the depth nonreturn prob should become high.
    previous_dnrp = depth_nonreturn_prob_kernel.support[0]
    observed_rgbd_for_this_vertex = jnp.array([0.1, 0.2, 0.3, 0.0])
    dnr_prob_samples = get_dnr_prob_samples(
        keys, observed_rgbd_for_this_vertex, previous_dnrp
    )
    assert dnr_prob_samples.mean() > 0.90

    # If depth is valid, the depth nonreturn prob should become low.
    previous_dnrp = depth_nonreturn_prob_kernel.support[-1]
    observed_rgbd_for_this_vertex = jnp.array([0.1, 0.2, 0.3, 1.0])
    dnr_prob_samples = get_dnr_prob_samples(
        keys, observed_rgbd_for_this_vertex, previous_dnrp
    )
    assert dnr_prob_samples.mean() < 0.10

    # If depth is valid, the depth nonreturn prob should stay low.
    previous_dnrp = depth_nonreturn_prob_kernel.support[0]
    observed_rgbd_for_this_vertex = jnp.array([0.1, 0.2, 0.3, 1.0])
    dnr_prob_samples = get_dnr_prob_samples(
        keys, observed_rgbd_for_this_vertex, previous_dnrp
    )
    assert dnr_prob_samples.mean() < 0.1


def test_color_prob_inference(hyperparams_and_inference_hyperparams):
    hyperparams, inference_hyperparams = hyperparams_and_inference_hyperparams

    color_scale = 0.01
    depth_scale = 0.001

    depth_nonreturn_prob_kernel = hyperparams["depth_nonreturn_prob_kernel"]
    visibility_prob_kernel = hyperparams["visibility_prob_kernel"]
    color_kernel = hyperparams["color_kernel"]
    obs_rgbd_kernel = hyperparams["image_kernel"].get_rgbd_vertex_kernel()

    previous_visibility_prob = visibility_prob_kernel.support[-1]
    previous_dnrp = depth_nonreturn_prob_kernel.support[0]
    latent_depth = 1.0

    def get_color_sample(key, observed_rgbd_for_point, previous_color):
        rgb, _, _, _, _ = inference_moves._propose_a_points_attributes(
            key,
            observed_rgbd_for_point,
            latent_depth,
            previous_color,
            previous_visibility_prob,
            previous_dnrp,
            depth_nonreturn_prob_kernel,
            visibility_prob_kernel,
            color_kernel,
            obs_rgbd_kernel,
            color_scale,
            depth_scale,
            inference_hyperparams,
        )
        return rgb

    get_color_samples = jax.vmap(get_color_sample, in_axes=(0, None, None))

    keys = jax.random.split(jax.random.PRNGKey(0), 1000)

    # If depth match and colors match, the color should stay the same.
    previous_color = jnp.array([0.1, 0.2, 0.3])
    observed_rgbd_for_this_vertex = jnp.array([0.1, 0.2, 0.3, latent_depth])
    color_samples = get_color_samples(
        keys, observed_rgbd_for_this_vertex, previous_color
    )
    assert jnp.max(jnp.abs(color_samples - previous_color)) < 0.02

    # # If depths match and colors slightly change, then the color should move.
    previous_color = jnp.array([0.15, 0.25, 0.35])
    observed_rgbd_for_this_vertex = jnp.array([0.1, 0.2, 0.3, latent_depth])
    color_samples = get_color_samples(
        keys, observed_rgbd_for_this_vertex, previous_color
    )
    assert (
        jnp.max(
            jnp.abs(
                jnp.median(color_samples, axis=0) - observed_rgbd_for_this_vertex[:3]
            )
        )
        < 0.03
    )
