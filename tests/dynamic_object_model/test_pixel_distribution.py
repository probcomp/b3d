# import jax.numpy as jnp


# def test_pixel_distribution():
#     registered_point_indices = [0, -1, 1]
#     all_rgbds = jnp.array(
#         [[1.0, 0.0, 0.0, 2.0], [0.0, 1.0, 0.0, 3.0], [0.0, 0.0, 1.0, 4.0]]
#     )
#     color_outlier_probs = jnp.array([0.01, 0.5, 0.99])
#     depth_outlier_probs = jnp.array([0.5, 0.9, 0.1])
#     color_scale = 0.1
#     depth_scale = 0.01
