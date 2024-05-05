"""
First test of the triangle-based soft renderer.
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
from demos.differentiable_renderer.rendering import all_pairs, render, renderer, project_pixel_to_plane

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

vertices, faces, colors, triangle_to_particle_index = jax.vmap(
    center_and_width_to_vertices_faces_colors
)(jnp.arange(len(particle_centers)), particle_centers, particle_widths, particle_colors)
vertices = vertices.reshape(-1, 3)
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

####
# rr.init("check_point_plane_intersection2")
# rr.connect("127.0.0.1:8812")
# rr.log("triangle", rr.Mesh3D(
#     vertex_positions=vertices, indices=faces, vertex_colors=colors
# ))

# # points = jax.vmap(project_pixel_to_plane, in_axes=(0, None))(
# #     10 * all_pairs(10, 10),
# #     vertices[faces[2]]
# # )
# ij = jnp.array([20, 80])
# triangle = vertices[faces[2]]
# rr.log("mytriangle", rr.Mesh3D(vertex_positions = triangle, indices=jnp.arange(3)))
# point = project_pixel_to_plane(ij, triangle)
# rr.log("point_on_plane", rr.Points3D(point))
# ray = ray_from_ij(ij[0], ij[1], fx, fy, cx, cy)
# rr.log("point_on_camera_plane", rr.Points3D(ray))
# rr.log("origin", rr.Points3D(jnp.zeros(3)))

# Useful for debugging the geometry in project_pixel_to_plane:
# rr.log("v1_v2", rr.LineStrips3D([jnp.array([vertex1, vertex2])]))
# rr.log("v1_v3", rr.LineStrips3D([jnp.array([vertex1, vertex3])]))
# rr.log("normal", rr.LineStrips3D([jnp.array([vertex2, vertex1 + normal])]))
# rr.log("ray_dir", rr.LineStrips3D([jnp.array([jnp.zeros(3), ray_dir])]))
# rr.log("ray_dir_ext", rr.LineStrips3D([jnp.array([jnp.zeros(3), ray_dir * t])]))

####

rr.init("softras_4")
rr.connect("127.0.0.1:8812")

rr_log_gt("gt", particle_centers, particle_widths, particle_colors)

SIGMA = 1e-4
GAMMA = 1e-4
EPSILON = 1e-5
hyperparams = (SIGMA, GAMMA, EPSILON)
rendered_soft = render(vertices, faces, triangle_colors, hyperparams)
rr.log("c/gt", rr.Image(color_image), timeless=True)
rr.log("c/rendered", rr.Image(rendered_soft), timeless=True)

ij = jnp.array([55, 55]) 