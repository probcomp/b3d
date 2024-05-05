"""
Stochastic rendering test.
"""


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


from demos.differentiable_renderer.utils import (
    center_and_width_to_vertices_faces_colors, rr_log_gt, ray_from_ij,
    fx, fy, cx, cy
)
from demos.differentiable_renderer.rendering import all_pairs, render_to_dist_parameters, renderer, project_pixel_to_plane

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

ij = jnp.array([51, 52])

vertices_og, faces, colors, triangle_to_particle_index = jax.vmap(
    center_and_width_to_vertices_faces_colors
)(jnp.arange(len(particle_centers)), particle_centers, particle_widths, particle_colors)
vertices = vertices_og.reshape(-1, 3)
faces = faces.reshape(-1, 3)
colors = colors.reshape(-1, 3)
triangle_to_particle_index = triangle_to_particle_index.reshape(-1)
_, _, triangle_id_image, depth_image = renderer.rasterize(
    Pose.identity()[None, ...], vertices, faces, jnp.array([[0, len(faces)]])
)
particle_intersected = triangle_to_particle_index[triangle_id_image - 1] * (triangle_id_image > 0) + -1 * (triangle_id_image ==0 )
blank_color = jnp.array([0.1, 0.1, 0.1]) # gray for unintersected particles
extended_colors = jnp.concatenate([jnp.array([blank_color]), particle_colors], axis=0)
color_image = extended_colors[particle_intersected + 1]
triangle_colors = particle_colors[triangle_to_particle_index]

rr.init("softras_gd_1")
rr.connect("127.0.0.1:8812")

rr_log_gt("gt", particle_centers, particle_widths, particle_colors)

SIGMA = 1e-4
GAMMA = 0.4
EPSILON = 1e-5
hyperparams = (SIGMA, GAMMA, EPSILON)
(weights, colors) = render_to_dist_parameters(vertices, faces, triangle_colors, hyperparams)

from demos.differentiable_renderer.likelihoods import mixture_rgb_sensor_model

# key = jax.random.PRNGKey(0)
# trace = mixture_rgb_sensor_model.simulate(key, (weights, colors, 3.))

rr.log("c/gt", rr.Image(color_image), timeless=True)
# rr.log("c/rendered", rr.Image(trace.get_retval().reshape(100, 100, 3)), timeless=True)

def get_render(key, weights, colors):
    return mixture_rgb_sensor_model.simulate(key, (weights, colors, 3.)).get_retval().reshape(100, 100, 3)

keys = jax.random.split(jax.random.PRNGKey(0), 100)
renders = jax.vmap(get_render, in_axes=(0, None, None))(
    keys, weights, colors
)
for t in range(100):
    rr.set_time_sequence("stochastic_render", t)
    rr.log("c/rendered", rr.Image(renders[t, ...]))

mean_render = jnp.mean(renders, axis=0)
rr.log("c/mean_render", rr.Image(mean_render), timeless=True)