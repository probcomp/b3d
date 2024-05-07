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

# Set up 3 squares oriented toward the camera, with different colors
particle_centers = jnp.array(
    [
        [0.0, 0.0, 1.0],
        [0.2, 0.2, 2.0],
        [0., 0., 5.]
    ]
)
particle_widths = jnp.array([0.1, 0.3, 20.])
particle_colors = jnp.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ]
)

#############################
# Visualize scene + renders #
#############################

# Get triangle "mesh" for the scene:
vertices_og, faces, vertex_colors, triangle_to_particle_index = jax.vmap(
     b3d.particle_center_width_color_to_vertices_faces_colors
)(jnp.arange(len(particle_centers)), particle_centers, particle_widths, particle_colors)

vertices = vertices_og.reshape(-1, 3)
faces = faces.reshape(-1, 3)
vertex_colors = vertex_colors.reshape(-1, 3)

# Manually change these so there is a gradient on some triangles
vertex_colors = vertex_colors.at[0, :].set(jnp.array([1., 1., 0.]))
vertex_colors = vertex_colors.at[6, :].set(jnp.array([1., 1., 1.]))


triangle_to_particle_index = triangle_to_particle_index.reshape(-1)
_, _, triangle_id_image, depth_image = renderer.rasterize(
    Pose.identity()[None, ...], vertices, faces, jnp.array([[0, len(faces)]])
)
particle_intersected = triangle_to_particle_index[triangle_id_image - 1] * (triangle_id_image > 0) + -1 * (triangle_id_image ==0 )
blank_color = jnp.array([0.1, 0.1, 0.1]) # gray for unintersected particles
extended_colors = jnp.concatenate([jnp.array([blank_color]), particle_colors], axis=0)
# color_image = extended_colors[particle_intersected + 1]
triangle_colors = particle_colors[triangle_to_particle_index]

color_image, _ = renderer.render_attribute(
    Pose.identity()[None, ...], vertices, faces, jnp.array([[0, len(faces)]]), vertex_colors
)


# visualize the scene
rr.init("differentiable_rendering--test_barycentric_interpolation3")
rr.connect("127.0.0.1:8812")
rr.log("scene/triangles", rr.Mesh3D(vertex_positions=vertices, indices=faces, vertex_colors=vertex_colors), timeless=True)
rr.log("scene/camera", rr.Pinhole(focal_length=rendering.fx, width=rendering.image_width, height=rendering.image_height), timeless=True)
rr.log("/img/opengl_rendering", rr.Image(color_image), timeless=True)

# Soft rendering of scene
SIGMA = 5e-5
GAMMA = 0.25
# GAMMA = 1e-4
EPSILON = -1
hyperparams = (SIGMA, GAMMA, EPSILON)

### Averaged rendering ###

hyperparams = rendering.DifferentiableRendererHyperparams(3, SIGMA, GAMMA, EPSILON)
soft_img_rgbd = rendering.render_to_average_rgbd(
    renderer, vertices, faces, vertex_colors,
    background_attribute = jnp.array([0.1, 0.1, 0.1, 0]),
    hyperparams=hyperparams
)

# Check that the old and new renderer do the same thing
rr.log("/img/averaged_rgb", rr.Image(soft_img_rgbd[:, :, :3]), timeless=True)
rr.log("/img/averaged_depth", rr.DepthImage(soft_img_rgbd[:, :, 3]), timeless=True)

### Stochastic rendering ###

def get_render(key, weights, colors):
    lab_color_space_noise_scale = 3.0
    depth_noise_scale = 0.07
    return likelihoods.mixture_rgbd_sensor_model.simulate(
        key,
        (weights, colors, lab_color_space_noise_scale, depth_noise_scale, 0., 10.)
    ).get_retval().reshape(100, 100, 4)
(weights, colors) = rendering.render_to_rgbd_dist_params(
    renderer, vertices, faces, vertex_colors, hyperparams
)
get_render(jax.random.PRNGKey(0), weights, colors)

# Generate + visualize 100 stochastic renders
keys = jax.random.split(jax.random.PRNGKey(0), 100)
renders = jax.vmap(get_render, in_axes=(0, None, None))(
    keys, weights, colors
)
for t in range(100):
    rr.set_time_sequence("stochastic_render", t)
    rr.log("img/stochastic_rgb_render", rr.Image(renders[t, :, :, :3]))
    rr.log("img/stochastic_depth_render", rr.DepthImage(renders[t, :, :, 3]))
    rr.log("scene/camera", rr.Image(renders[t, :, :, :3]))

