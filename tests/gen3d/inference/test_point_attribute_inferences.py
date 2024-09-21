import b3d.chisight.gen3d.inference.point_attribute_proposals as point_attribute_proposals
import b3d.chisight.gen3d.settings
import jax
import jax.numpy as jnp
import pytest
from genjax import Pytree


@pytest.fixture
def hyperparams_and_inference_hyperparams():
    near, far, image_height, image_width = 0.001, 5.0, 480, 640
    intrinsics = {
        "image_height": Pytree.const(image_height),
        "image_width": Pytree.const(image_width),
        "near": near,
        "far": far,
    }

    hyperparams = b3d.chisight.gen3d.settings.hyperparams
    inference_hyperparams = b3d.chisight.gen3d.settings.inference_hyperparams
    hyperparams["intrinsics"] = intrinsics
    return hyperparams, inference_hyperparams


def test_visibility_prob_inference(hyperparams_and_inference_hyperparams):
    hyperparams, inference_hyperparams = hyperparams_and_inference_hyperparams

    color_scale = 0.01
    depth_scale = 0.001

    depth_nonreturn_prob_kernel = hyperparams["depth_nonreturn_prob_kernel"]
    visibility_prob_kernel = hyperparams["visibility_prob_kernel"]

    previous_color = jnp.array([0.1, 0.2, 0.3])
    previous_dnrp = depth_nonreturn_prob_kernel.support[0]

    def get_visibility_prob_sample(
        key, observed_rgbd_for_point, previous_visibility_prob
    ):
        sample, _ = point_attribute_proposals._propose_a_points_attributes(
            key,
            observed_rgbd_for_point,
            jnp.array(1.0),  # point depth
            previous_color,
            previous_visibility_prob,
            previous_dnrp,
            color_scale,
            depth_scale,
            hyperparams,
            inference_hyperparams,
        )
        return sample["visibility_prob"]

    get_visibility_prob_samples = jax.vmap(
        get_visibility_prob_sample, in_axes=(0, None, None)
    )

    keys = jax.random.split(jax.random.PRNGKey(0), 1000)

    # Verify that when the color matches exactly but the depth change drasticaly, the visibility prob switches to low.
    previous_visibility_prob = visibility_prob_kernel.support[-1]
    observed_rgbd_for_this_vertex = jnp.array([0.1, 0.2, 0.35, 4.0])
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

    previous_color = jnp.array([0.1, 0.2, 0.3])
    previous_visibility_prob = visibility_prob_kernel.support[-1]
    latent_rgbd_for_point = jnp.concatenate([previous_color, jnp.array([1.0])])

    def get_dnr_prob_sample(key, observed_rgbd_for_point, previous_dnrp):
        sample, _ = point_attribute_proposals._propose_a_points_attributes(
            key,
            observed_rgbd_for_point,
            latent_rgbd_for_point,
            previous_color,
            previous_visibility_prob,
            previous_dnrp,
            color_scale,
            depth_scale,
            hyperparams,
            inference_hyperparams,
        )
        return sample["depth_nonreturn_prob"]

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

    previous_visibility_prob = visibility_prob_kernel.support[-1]
    previous_dnrp = depth_nonreturn_prob_kernel.support[0]
    latent_depth = 1.0
    latent_rgbd_for_point = jnp.concatenate(
        [jnp.array([0.1, 0.2, 0.3]), jnp.array([latent_depth])]
    )

    def get_color_sample(key, observed_rgbd_for_point, previous_color):
        sample, _ = point_attribute_proposals._propose_a_points_attributes(
            key,
            observed_rgbd_for_point,
            latent_rgbd_for_point,
            previous_color,
            previous_visibility_prob,
            previous_dnrp,
            color_scale,
            depth_scale,
            hyperparams,
            inference_hyperparams,
        )
        return sample["colors"]

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
