import importlib

import jax.numpy as jnp
from jax.random import PRNGKey, split

import b3d
import b3d.chisight.dynamic_object_model.kfold_image_kernel as kfk

importlib.reload(kfk)


def run_test():
    intrinsics = (
        5,
        5,
        200.0,
        200.0,
        50.0,
        50.0,
        0.01,
        10.0,
    )
    image_width, image_height, fx, fy, cx, cy, _, _ = intrinsics

    depth_image = jnp.ones((image_height, image_width), dtype=jnp.float32)
    points = b3d.camera.camera_from_depth(depth_image, intrinsics).reshape(-1, 3)

    result = kfk.raycast_to_image_nondeterministic(
        PRNGKey(0),
        {
            "height": image_height,
            "width": image_width,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
        },
        points,
        2,
    )
    assert jnp.all(
        jnp.max(result, axis=2)
        == jnp.arange(points.shape[0]).reshape(image_height, image_width)
    )

    # With 3x points per pixel, most of the time we'll register two of them.
    # The probability of not registering 1 is 0.25.
    # (This is the probability of the second and third point writing to the same
    # slot as the first, which is 1/4.)
    # Test that this seems to work out.
    points_replicated_3x = jnp.tile(points, (3, 1))
    n_tests = 10
    keys = split(PRNGKey(0), n_tests)
    total_num_misses = 0
    for key in keys:
        result_x3 = kfk.raycast_to_image_nondeterministic(
            key,
            {
                "height": image_height,
                "width": image_width,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
            },
            points_replicated_3x,
            2,
        )
        total_num_misses += jnp.sum(result_x3 == -1)
    mean_num_misses = total_num_misses / n_tests
    estimated_p_miss = mean_num_misses / (image_width * image_height)
    assert jnp.allclose(estimated_p_miss, 0.25, atol=0.05)
