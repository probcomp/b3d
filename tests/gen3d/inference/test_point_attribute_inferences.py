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


def get_sample(
    key,
    observed_rgbd_for_point,
    latent_depth,
    previous_color,
    previous_visibility_prob,
    previous_dnrp,
    hyperparams,
    inference_hyperparams,
):
    color_scale = hyperparams["color_scale_kernel"].support[0]
    depth_scale = hyperparams["depth_scale_kernel"].support[0]

    sample, _ = point_attribute_proposals._propose_a_points_attributes(
        key,
        observed_rgbd_for_point,
        latent_depth,
        previous_color,
        previous_visibility_prob,
        previous_dnrp,
        color_scale,
        depth_scale,
        hyperparams,
        inference_hyperparams,
    )
    return sample


get_samples = jax.vmap(
    get_sample, in_axes=(0, None, None, None, None, None, None, None)
)

keys = jax.random.split(jax.random.PRNGKey(0), 1000)


def test_color_visibility_inference(hyperparams_and_inference_hyperparams):
    hyperparams, inference_hyperparams = hyperparams_and_inference_hyperparams
    depth_nonreturn_prob_kernel = hyperparams["depth_nonreturn_prob_kernel"]
    visibility_prob_kernel = hyperparams["visibility_prob_kernel"]


    previous_dnrp = depth_nonreturn_prob_kernel.support[0]

    depth_change = 0.05
    # Color matches but depth changes by 2.5cm --> Visibility should be 0.
    observed_rgbd_for_this_vertex = jnp.array([0.1, 0.2, 0.3, 1.0 + depth_change])
    samples = get_samples(
        keys,
        observed_rgbd_for_this_vertex,
        1.0,
        jnp.array([0.1, 0.2, 0.3]),
        visibility_prob_kernel.support[-1],
        previous_dnrp,
        hyperparams,
        inference_hyperparams,
    )
    assert samples["visibility_prob"].mean() < 0.001
    assert jnp.allclose(samples["colors"], jnp.array([0.1, 0.2, 0.3]), atol=0.001)

    samples = get_samples(
        keys,
        observed_rgbd_for_this_vertex,
        1.0,
        jnp.array([0.1, 0.2, 0.3]),
        visibility_prob_kernel.support[0],
        previous_dnrp,
        hyperparams,
        inference_hyperparams,
    )
    assert samples["visibility_prob"].mean() < 0.001
    assert jnp.allclose(samples["colors"], jnp.array([0.1, 0.2, 0.3]), atol=0.001)

    # Color matches and depth matches --> Visibility should be 1.
    observed_rgbd_for_this_vertex = jnp.array([0.12, 0.22, 0.32, 1.0])
    samples = get_samples(
        keys,
        observed_rgbd_for_this_vertex,
        1.0,
        jnp.array([0.1, 0.2, 0.3]),
        visibility_prob_kernel.support[0],
        previous_dnrp,
        hyperparams,
        inference_hyperparams,
    )
    assert samples["visibility_prob"].mean() > 0.999
    # assert jnp.allclose(samples["colors"], jnp.array([0.1, 0.2, 0.3]), atol=0.001)

    observed_rgbd_for_this_vertex = jnp.array([0.12, 0.22, 0.32, 1.0])
    samples = get_samples(
        keys,
        observed_rgbd_for_this_vertex,
        1.0,
        jnp.array([0.1, 0.2, 0.3]),
        visibility_prob_kernel.support[-1],
        previous_dnrp,
        hyperparams,
        inference_hyperparams,
    )
    assert samples["visibility_prob"].mean() > 0.999
    # assert jnp.allclose(samples["colors"], observed_rgbd_for_this_vertex[:3], atol=0.001)

    # Color matches but depth changes by 2.5cm --> Visibility should be 0.
    observed_rgbd_for_this_vertex = jnp.array([0.5, 0.2, 0.3, 1.0])
    samples = get_samples(
        keys,
        observed_rgbd_for_this_vertex,
        1.0,
        jnp.array([0.1, 0.2, 0.3]),
        visibility_prob_kernel.support[-1],
        previous_dnrp,
        hyperparams,
        inference_hyperparams,
    )
    assert samples["visibility_prob"].mean() < 0.001
    assert jnp.allclose(samples["colors"], jnp.array([0.1, 0.2, 0.3]), atol=0.001)
