"""
Test rendering based on signed distance weighting.
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
    image_width, image_height, fx, fy, cx, cy, near, far, renderer,
    center_and_width_to_vertices_faces_colors, ray_from_ij, WINDOW,
    get_pixel_color_from_ij, render, rr_log_gt,
    get_signed_dist,
    all_pairs
)

rr.init("signed_dist_weighting_5")
rr.connect("127.0.0.1:8812")

particle_centers = jnp.array(
    [
        [0.0, 0.0, 1.0],
        [0.2, 0.2, 2.0],
    ]
)
particle_widths = jnp.array([0.1, 0.3])
particle_colors = jnp.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)
rr_log_gt("gt", particle_centers, particle_widths, particle_colors)

ij = jnp.array([50, 51])

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
color_image = colors[triangle_id_image - 1]

point_of_intersection = b3d.xyz_from_depth(depth_image, fx, fy, cx, cy)
particle_intersected = triangle_to_particle_index[triangle_id_image - 1] * (triangle_id_image > 0) + -1 * (triangle_id_image ==0 )

WINDOW = 3
particle_intersected_padded = jnp.pad(
    particle_intersected, pad_width=[(WINDOW, WINDOW)], constant_values=-1
)

i,j = ij
particle_intersected_padded_in_window = jax.lax.dynamic_slice(
    particle_intersected_padded,
    (ij[0], ij[1]),
    (2 * WINDOW + 1, 2 * WINDOW + 1),
)

offset_window = all_pairs(2 * WINDOW + 1, 2 * WINDOW + 1)
ij_window = offset_window + jnp.array([i - 2*WINDOW - 1, j - 2*WINDOW - 1])

signed_dist_from_square_boundary = jax.vmap(get_signed_dist, in_axes=(0, 0, None, None))(
    ij_window, particle_intersected_padded_in_window.reshape(-1), particle_centers, particle_widths
).reshape(2 * WINDOW + 1, 2 * WINDOW + 1)

# (2W + 1, 2W + 1)
# z_dists = ij_plane_intersections[:, :, 2]

# Signed_dist_score 
signed_dist_score = 0.5 * (1 - jnp.tanh(3 * (signed_dist_from_square_boundary - 0.3)))

# signed_dist_score = jnp.where(
#     particle_intersected_padded_in_window == -1,
#     jnp.zeros_like(signed_dist_score),
#     signed_dist_score
# )

rr.log("sd_score", rr.DepthImage(signed_dist_score), timeless=True)
total_scores = signed_dist_score + 1e-10 # + z_score

normalized_scores = total_scores / total_scores.sum()
rr.log("normalized_sd_score", rr.DepthImage(normalized_scores), timeless=True)

rendered_soft = render(particle_centers, particle_widths, particle_colors)
rr.log("rendered", rr.Image(rendered_soft), timeless=True)

###

def compute_error(centers):
    rendered = render(centers, particle_widths, particle_colors)
    return jnp.sum(jnp.abs((rendered - rendered_soft)))


particle_centers_shifted = jnp.array(
    [
        [0.05, 0.0, 1.0],
        [0.15, 0.2, 2.0],
    ]
)
rendered_shifted = render(particle_centers_shifted, particle_widths, particle_colors)
rr.log("shifted", rr.Image(rendered_shifted), timeless=True)

print("ERROR:")
print(compute_error(particle_centers_shifted))
print("GRAD:")
g = jax.grad(compute_error)(particle_centers_shifted)
print(g)

grad_jit = jax.jit(jax.grad(compute_error))
current_centers = particle_centers_shifted
eps = 1e-5
for i in range(300):
    g = grad_jit(current_centers)
    current_centers = current_centers - eps * g
    if i % 10 == 0:
        # rr.log("error", rr.Scalar(compute_error(current_centers)))
        print(f"ERROR: {compute_error(current_centers)}")
        rr.set_time_sequence("gd", i)
        rendered = render(current_centers, particle_widths, particle_colors)
        rr.log("gd", rr.Image(rendered))
        rr.log("gd-error", rr.Image(jnp.abs((rendered - rendered_soft))))