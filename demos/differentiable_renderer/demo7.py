"""
GD fitting test.
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
GAMMA = 0.2
EPSILON = 1e-5
hyperparams = (SIGMA, GAMMA, EPSILON)
rendered_soft = render(vertices, faces, triangle_colors, hyperparams)
rr.log("c/gt", rr.Image(color_image), timeless=True)
rr.log("c/rendered", rr.Image(rendered_soft), timeless=True)

def render_from_centers(new_particle_centers):
    particle_center_delta  = new_particle_centers - particle_centers
    new_vertices = vertices_og + jnp.expand_dims(particle_center_delta, 1)
    return render(new_vertices.reshape(-1, 3), faces.reshape(-1, 3), particle_colors[triangle_to_particle_index], hyperparams)

def compute_error(centers):
    rendered = render_from_centers(centers)
    return jnp.sum(jnp.abs((rendered - color_image)))

particle_centers_shifted = jnp.array(
    [
        [0.05, 0.0, 1.0],
        [0.15, 0.2, 2.0],
        [0., 0., 5.]
    ]
)
rendered_shifted = render_from_centers(particle_centers_shifted)
rr.log("shifted", rr.Image(rendered_shifted), timeless=True)

rr.log("gt_mesh", rr.Mesh3D(
    vertex_positions=vertices,
    indices=faces,
    vertex_colors=colors),
    timeless=True
)

grad_jit = jax.jit(jax.grad(compute_error))
current_centers = particle_centers_shifted
eps = 1e-5
for i in range(40):
    g = grad_jit(current_centers)
    current_centers = current_centers - eps * g
    print(f"ERROR: {compute_error(current_centers)}")
    rr.set_time_sequence("gd", i)
    rendered = render_from_centers(current_centers)
    rr.log("gd", rr.Image(rendered))
    rr.log("gd-error", rr.Image(jnp.abs((rendered - rendered_soft))))

    v, f, c, t2pidx = jax.vmap(
        center_and_width_to_vertices_faces_colors
    )(jnp.arange(len(current_centers)), current_centers, particle_widths, particle_colors)
    v = v.reshape(-1, 3)
    f = f.reshape(-1, 3)
    c = c.reshape(-1, 3)
    rr.log("gd_mesh", rr.Mesh3D(
        vertex_positions=v,
        indices=f)
    )