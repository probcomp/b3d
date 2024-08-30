import importlib

import b3d.chisight.dynamic_object_model.kfold_image_kernel as kik
import jax
import jax.numpy as jnp

importlib.reload(kik)


def test_pixel_distribution_sampling_with_two_of_three_slots():
    registered_point_indices = jnp.array([0, -1, 1])
    all_rgbds = jnp.array(
        [[1.0, 0.0, 0.0, 2.0], [0.0, 1.0, 0.0, 3.0], [0.0, 0.0, 1.0, 4.0]]
    )
    near, far = 0.001, 100.0
    color_outlier_probs = jnp.array([0.01, 0.5, 0.95])
    depth_outlier_probs = jnp.array([0.5, 0.01, 0.1])
    color_scale = 0.04
    depth_scale = 0.01

    args = (
        registered_point_indices,
        all_rgbds,
        color_outlier_probs,
        depth_outlier_probs,
        color_scale,
        depth_scale,
        near,
        far,
    )

    key = jax.random.PRNGKey(0)
    samples_100 = jax.vmap(lambda k: kik.pixel_distribution.sample(k, *args))(
        jax.random.split(key, 100)
    )
    samples_100

    # shouldn't be able to pick index 2, so should have very few
    # that are close to [0.0, 0.0, 1.0, 4.0]
    def is_close_to_0014(rgbd):
        return jnp.logical_and(
            jnp.allclose(rgbd[:3], jnp.array([0.0, 0.0, 1.0]), atol=0.1),
            jnp.abs(rgbd[3] - 4.0) < 0.1,
        )

    n_close = jnp.sum(jax.vmap(is_close_to_0014)(samples_100))
    assert n_close < 4

    # Should have close to half the samples pick index 0.
    # Most of these should have color values close to [1.0, 0.0, 0.0].
    def color_near_100(rgbd):
        return jnp.allclose(rgbd[:3], jnp.array([1.0, 0.0, 0.0]), atol=0.15)

    n_color_near_100 = jnp.sum(jax.vmap(color_near_100)(samples_100))
    assert n_color_near_100 > 40
    assert n_color_near_100 < 60

    # Of these, about half should have a depth outlier, and half should have
    # a depth value close to 2.0.
    def color_near_100_and_depth_near_2(rgbd):
        return jnp.logical_and(color_near_100(rgbd), jnp.abs(rgbd[3] - 2.0) < 0.15)

    n_color_near_100_and_depth_near_2 = jnp.sum(
        jax.vmap(color_near_100_and_depth_near_2)(samples_100)
    )
    assert n_color_near_100_and_depth_near_2 > n_color_near_100 / 2 - 5
    assert n_color_near_100_and_depth_near_2 < n_color_near_100 / 2 + 5

    # Test that of the outliers, some are far from 2.0
    def color_near_100_and_far_depth_outlier(rgbd):
        return jnp.logical_and(color_near_100(rgbd), jnp.abs(rgbd[3] - 2.0) > 5.0)

    n_color_near_100_and_far_depth_outlier = jnp.sum(
        jax.vmap(color_near_100_and_far_depth_outlier)(samples_100)
    )
    assert n_color_near_100_and_far_depth_outlier > 1

    # Should have close to half the samples pick index 1.
    # Most of these should have depth values close to 3.0.
    def depth_near_3(rgbd):
        return jnp.abs(rgbd[3] - 3.0) < 0.15

    n_depth_near_3 = jnp.sum(jax.vmap(depth_near_3)(samples_100))
    assert n_depth_near_3 > 40
    assert n_depth_near_3 < 60

    # Of these, about half should have a color outlier, and half should have
    # a color value close to [0.0, 1.0, 0.0].
    def depth_near_3_and_color_near_010(rgbd):
        return jnp.logical_and(
            depth_near_3(rgbd),
            jnp.allclose(rgbd[:3], jnp.array([0.0, 1.0, 0.0]), atol=0.15),
        )

    n_depth_near_3_and_color_near_010 = jnp.sum(
        jax.vmap(depth_near_3_and_color_near_010)(samples_100)
    )
    assert n_depth_near_3_and_color_near_010 > n_depth_near_3 / 2 - 5
    assert n_depth_near_3_and_color_near_010 < n_depth_near_3 / 2 + 5

    # Test that of the outliers, some are far from [0.0, 1.0, 0.0]
    def depth_near_3_and_far_color_outlier(rgbd):
        return jnp.logical_and(
            depth_near_3(rgbd),
            ~jnp.allclose(rgbd[:3], jnp.array([0.0, 1.0, 0.0]), atol=0.3),
        )

    n_depth_near_3_and_far_color_outlier = jnp.sum(
        jax.vmap(depth_near_3_and_far_color_outlier)(samples_100)
    )
    assert n_depth_near_3_and_far_color_outlier > 1


def test_pixel_distribution_sampling_with_no_slots():
    registered_point_indices = jnp.array([-1, -1, -1])
    all_rgbds = jnp.array(
        [[1.0, 0.0, 0.0, 2.0], [0.0, 1.0, 0.0, 3.0], [0.0, 0.0, 1.0, 4.0]]
    )
    near, far = 0.001, 100.0
    color_outlier_probs = jnp.array([0.01, 0.5, 0.95])
    depth_outlier_probs = jnp.array([0.5, 0.01, 0.1])
    color_scale = 0.04
    depth_scale = 0.01

    args = (
        registered_point_indices,
        all_rgbds,
        color_outlier_probs,
        depth_outlier_probs,
        color_scale,
        depth_scale,
        near,
        far,
    )

    key = jax.random.PRNGKey(0)
    samples_100 = jax.vmap(lambda k: kik.pixel_distribution.sample(k, *args))(
        jax.random.split(key, 100)
    )
    samples_100

    # Check that almost all of these seem to be outliers in both color
    # and depth.
    def is_color_and_depth_outlier(rgbd, base_rgbd):
        return jnp.logical_and(
            ~jnp.allclose(rgbd[:3], base_rgbd[:3], atol=0.15),
            jnp.abs(rgbd[3] - base_rgbd[3]) > 2.0,
        )

    def is_color_and_depth_outlier_for_all(rgbd):
        return jnp.all(
            jax.vmap(lambda base_rgbd: is_color_and_depth_outlier(rgbd, base_rgbd))(
                all_rgbds
            )
        )

    n_color_and_depth_outliers = jnp.sum(
        jax.vmap(is_color_and_depth_outlier_for_all)(samples_100)
    )
    assert n_color_and_depth_outliers > 90


def test_singleslot_pixel_distribution_sampling():
    registered_point_indices = jnp.array([0, 0, 0])
    all_rgbds = jnp.array(
        [[1.0, 0.0, 0.0, 2.0], [0.0, 1.0, 0.0, 3.0], [0.0, 0.0, 1.0, 4.0]]
    )
    near, far = 0.001, 100.0
    color_outlier_probs = jnp.array([0.01, 0.5, 0.95])
    depth_outlier_probs = jnp.array([0.5, 0.01, 0.1])
    color_scale = 0.04
    depth_scale = 0.01

    args = (
        registered_point_indices,
        all_rgbds,
        color_outlier_probs,
        depth_outlier_probs,
        color_scale,
        depth_scale,
        near,
        far,
    )
    key = jax.random.PRNGKey(0)
    samples_100 = jax.vmap(lambda k: kik.pixel_distribution.sample(k, *args))(
        jax.random.split(key, 100)
    )

    # Should have close to all the samples pick index 0.
    # Most of these should have color values close to [1.0, 0.0, 0.0].
    def color_near_100(rgbd):
        return jnp.allclose(rgbd[:3], jnp.array([1.0, 0.0, 0.0]), atol=0.15)

    n_color_near_100 = jnp.sum(jax.vmap(color_near_100)(samples_100))
    assert n_color_near_100 > 90

    # Of these, about half should have a depth outlier, and half should have
    # a depth value close to 2.0.
    def color_near_100_and_depth_near_2(rgbd):
        return jnp.logical_and(color_near_100(rgbd), jnp.abs(rgbd[3] - 2.0) < 0.15)

    n_color_near_100_and_depth_near_2 = jnp.sum(
        jax.vmap(color_near_100_and_depth_near_2)(samples_100)
    )
    assert n_color_near_100_and_depth_near_2 > 40
    assert n_color_near_100_and_depth_near_2 < 60
