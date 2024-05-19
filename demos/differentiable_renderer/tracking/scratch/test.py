import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np
import os
import trimesh
import b3d
from jax.scipy.spatial.transform import Rotation as Rot
from b3d import Pose
import rerun as rr
import functools
import genjax

import b3d.differentiable_renderer as rendering
import b3d.likelihoods as likelihoods
import demos.differentiable_renderer.utils as utils

rr.init("gradients")
rr.connect("127.0.0.1:8812")

# Load date
path = os.path.join(
    b3d.get_root_path(),
    "assets/shared_data_bucket/input_data/shout_on_desk.r3d.video_input.npz",
)
video_input = b3d.VideoInput.load(path)

# Set up OpenGL renderer
image_width = 100
image_height = 100
fx = 50.0
fy = 50.0
cx = 50.0
cy = 50.0
near = 0.001
far = 16.0
renderer = b3d.Renderer(image_width, image_height, fx, fy, cx, cy, near, far)

WINDOW = 5

def render(particle_centers, particle_widths, particle_colors):
    # particle_colors = jax.nn.sigmoid(particle_colors)
    # particle_widths = jnp.abs(particle_widths)
    # Get triangle "mesh" for the scene:
    vertices, faces, vertex_colors, triangle_index_to_particle_index = jax.vmap(
        b3d.square_center_width_color_to_vertices_faces_colors
    )(jnp.arange(len(particle_centers)), particle_centers, particle_widths / 2, particle_colors)
    vertices = vertices.reshape(-1, 3)
    faces = faces.reshape(-1, 3)
    vertex_colors = vertex_colors.reshape(-1, 3)
    triangle_index_to_particle_index = triangle_index_to_particle_index.reshape(-1)
    image = rendering.render_to_average_rgbd(
        renderer,
        vertices,
        faces,
        vertex_colors,
    )
    return image

render_jit = jax.jit(render)

# triangle_id_image, depth_image = render_jit(particle_centers, particle_widths, particle_colors)
# Set up 3 squares oriented toward the camera, with different colors

particle_centers = jnp.array([[0.0, 0.01, 0.4]])
particle_colors = jnp.array([[1.0, 0.0, 0.0]])
particle_widths = jnp.array([[0.05]])
print(jax.jit(jax.value_and_grad(lambda a,b,c: jnp.mean(render(a,b,c)), argnums=(0,1,2)))(particle_centers, particle_widths, particle_colors))


# vertices, faces, vertex_colors, triangle_index_to_particle_index = jax.vmap(
#     b3d.square_center_width_color_to_vertices_faces_colors
# )(jnp.arange(len(particle_centers)), particle_centers, particle_widths / 2, particle_colors)
# vertices = vertices.reshape(-1, 3)
# faces = faces.reshape(-1, 3)
# vertex_colors = vertex_colors.reshape(-1, 3)
# triangle_index_to_particle_index = triangle_index_to_particle_index.reshape(-1)

# v_and_grads = jax.value_and_grad(lambda *args: rendering.render_to_rgbd_dist_params(*args)[0][50, 50, 0], argnums=(1, 3))(
#     renderer,
#     vertices,
#     faces,
#     vertex_colors,
# )

# uvs, _, triangle_id_image, depth_image = renderer.rasterize(
#     Pose.identity()[None, ...], vertices, faces,  jnp.array([[0, len(faces)]])
# )
# hyperparams = rendering.DEFAULT_HYPERPARAMS
# triangle_intersected_padded = jnp.pad(
#     triangle_id_image, pad_width=[(hyperparams.WINDOW, hyperparams.WINDOW)], constant_values=-1
# )
# hyperparams_and_intrinsics = rendering.HyperparamsAndIntrinsics(hyperparams, renderer.fx, renderer.fy, renderer.cx, renderer.cy)
# ij = jnp.array([0, 0])
# (indices, weights, bc) = rendering.get_weights_and_barycentric_coords(ij, vertices, faces, triangle_intersected_padded, hyperparams_and_intrinsics)
# v = jax.value_and_grad(lambda *args: jnp.mean(rendering.get_weights_and_barycentric_coords(*args)[1]), argnums=(1,))(
#     ij, vertices, faces, triangle_intersected_padded, hyperparams_and_intrinsics
# )
# v

