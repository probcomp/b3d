# import jax.numpy as jnp
# import jax
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import trimesh
# import b3d
# from jax.scipy.spatial.transform import Rotation as Rot
# from b3d import Pose
# import rerun as rr
# import functools
# import genjax
# from tqdm import tqdm
# import jax
# import jax.numpy as jnp
# import optax
# import b3d.differentiable_renderer as rendering
# import b3d.likelihoods as likelihoods
# import demos.differentiable_renderer.utils as utils
# from functools import partial

# rr.init("gradients")
# rr.connect("127.0.0.1:8812")

# def map_nested_fn(fn):
#   '''Recursively apply `fn` to the key-value pairs of a nested dict.'''
#   def map_fn(nested_dict):
#     return {k: (map_fn(v) if isinstance(v, dict) else fn(k, v))
#             for k, v in nested_dict.items()}
#   return map_fn

# # Set up OpenGL renderer
# image_width = 100
# image_height = 100
# fx = 50.0
# fy = 50.0
# cx = 50.0
# cy = 50.0
# near = 0.001
# far = 16.0
# renderer = b3d.Renderer(image_width, image_height, fx, fy, cx, cy, near, far)
# # renderer.set_intrinsics(image_width, image_height, fx, fy, cx, cy, near, far)

# WINDOW = 5

# particle_centers = jnp.zeros((1,3))
# particle_widths = jnp.array([0.1])
# particle_colors = jnp.array([[0.0, 0.0, 1.0]])

# # Get triangle "mesh" for the scene:
# vertices, faces, vertex_colors, triangle_index_to_particle_index = jax.vmap(
#     b3d.square_center_width_color_to_vertices_faces_colors
# )(jnp.arange(len(particle_centers)), particle_centers, particle_widths / 2, particle_colors)
# vertices = vertices.reshape(-1, 3)
# faces = faces.reshape(-1, 3)
# vertex_rgbs = vertex_colors.reshape(-1, 3)


# hyperparams = rendering.DEFAULT_HYPERPARAMS
# def render(pose):
#     image = rendering.render_to_average_rgbd(
#         renderer,
#         pose.apply(vertices),
#         faces,
#         vertex_rgbs,
#         background_attribute=jnp.array([0.0, 0.0, 0.0, 0])
#     )
#     return image

# render_jit = jax.jit(render)

# gt_pose = Pose(jnp.array([0.0, 0.0, 0.2]), Rot.from_euler('y', jnp.pi/4).as_quat())
# gt_image = render_jit(gt_pose)
# rr.set_time_sequence("frame", 0)
# rr.log('image', rr.Image(gt_image[...,:3]),timeless=True)
# rr.log('depth', rr.Image(gt_image[...,3]),timeless=True)
# rr.log('cloud', rr.Points3D(b3d.xyz_from_depth(gt_image[...,3],fx,fy,cx,cy).reshape(-1,3)),timeless=True)


# def loss_func_rgbd(params, gt):
#     image = render(b3d.Pose(params["position"], params["quaternion"]))
#     error = jnp.clip(jnp.abs(image[...,3] - gt[...,3]), 0.0, 0.05)
#     return jnp.mean(error**2)
# loss_func_rgbd_grad = jax.jit(jax.value_and_grad(loss_func_rgbd, argnums=(0,)))

# @partial(jax.jit, static_argnums=(0,))
# def update_params(tx, params, gt_image, state):
#     loss, (gradients,) = loss_func_rgbd_grad(params, gt_image)
#     updates, state = tx.update(gradients, state, params)
#     params = optax.apply_updates(params, updates)
#     return params, state, loss

# label_fn = map_nested_fn(lambda k, _: k)



# test_cases = [
#     (
#         "just position",
#         optax.multi_transform(
#             {
#                 'position': optax.adam(5e-3),
#                 'quaternion': optax.adam(5e-3),
#             },
#             label_fn
#         ),
#         {
#             "position": jnp.array([0.0, 0.05, 0.2]),
#             "quaternion": Rot.from_euler('y', -0.2).as_quat()
#         }
#     ),
#     (
#         "just position",
#         optax.multi_transform(
#             {
#                 'position': optax.adam(5e-3),
#                 'quaternion': optax.adam(5e-3),
#             },
#             label_fn
#         ),
#         {
#             "position": jnp.array([0.04, 0.05, 0.2]),
#             "quaternion": Rot.from_euler('y', -0.4).as_quat()
#         }
#     ),
#     (
#         "just position",
#         optax.multi_transform(
#             {
#                 'position': optax.adam(5e-3),
#                 'quaternion': optax.adam(5e-3),
#             },
#             label_fn
#         ),
#         {
#             "position": jnp.array([0.04, 0.05, 0.22]),
#             "quaternion": Rot.from_euler('y', -0.4).as_quat()
#         }
#     ),
# ]



# images = []
# for (title, tx, params) in test_cases:

#     image = render_jit(b3d.Pose(params["position"], params["quaternion"]))
#     rr.set_time_sequence("frame", 0)
#     rr.log('image/reconstruction', rr.Image(image[...,:3]),timeless=True)
#     rr.log('depth/reconstruction', rr.Image(image[...,3]),timeless=True)
#     rr.log('cloud/reconstruction', rr.Points3D(b3d.xyz_from_depth(image[...,3],fx,fy,cx,cy).reshape(-1,3)),timeless=True)

#     print(title)
#     pbar = tqdm(range(400))
#     state = tx.init(params)
#     for t in pbar:
#         params, state, loss = update_params(tx, params, gt_image, state)
#         pbar.set_description(f"Loss: {loss}")


#     image = render_jit(b3d.Pose(params["position"], params["quaternion"]))
#     rr.set_time_sequence("frame", 0)
#     rr.log('image/reconstruction', rr.Image(image[...,:3]),timeless=True)
#     rr.log('depth/reconstruction', rr.Image(image[...,3]),timeless=True)
#     rr.log('cloud/reconstruction', rr.Points3D(b3d.xyz_from_depth(image[...,3],fx,fy,cx,cy).reshape(-1,3)),timeless=True)

#     assert jnp.allclose(gt_pose.pos, params["position"], atol=1e-3)
#     assert jnp.allclose(gt_pose.quat / jnp.linalg.norm(gt_pose.quat), params["quaternion"] / jnp.linalg.norm(params["quaternion"]), atol=1e-2)

