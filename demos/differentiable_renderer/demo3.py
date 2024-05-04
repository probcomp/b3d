"""
Signed dist testing.
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
    get_pixel_color_from_ij, render, rr_log_gt
)

rr.init("signed_dist_1")
rr.connect("127.0.0.1:8812")

particle_centers = jnp.array(
    [
        [0.0, 0.0, 1.0],
        [0.2, 0.2, 2.0],
    ]
)
particle_widths = jnp.array([0.3, 0.3])
particle_colors = jnp.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)
rr_log_gt("gt", particle_centers, particle_widths, particle_colors)

def all_pairs(X, Y):
    return jnp.stack(jnp.meshgrid(jnp.arange(X), jnp.arange(Y)), axis=-1).reshape(-1, 2)

# ij = jnp.array([50, 51])
# center = particle_centers[0]
# intersection_point = jnp.array([x, y, z])
# particle_center = particle_centers[0]
# particle_width = particle_widths[0]

def get_ray_plane_intersection(ij, center):
    u = ij[0]
    v = ij[1]
    z = center[2]
    x = z * (u - cx) / fx
    y = z * (v - cy) / fy
    return jnp.array([x, y, z])

def get_signed_dist_from_int_point(intersection_point, particle_center, particle_width):
    center = particle_center
    cenx, ceny = center[0], center[1]
    x, y = intersection_point[0], intersection_point[1]
    minx, maxx = cenx - particle_width / 2, cenx + particle_width / 2
    miny, maxy = ceny - particle_width / 2, ceny + particle_width / 2

    # Positive = point is inside region
    signed_dist_x = jnp.minimum(x - minx, maxx - x)
    signed_dist_y = jnp.minimum(y - miny, maxy - y)
    
    dist_same_sign = jnp.sqrt(signed_dist_x**2 + signed_dist_y**2)

    both_pos = jnp.logical_and(signed_dist_x > 0, signed_dist_y > 0)
    both_neg = jnp.logical_and(signed_dist_x < 0, signed_dist_y < 0)
    same_sign = jnp.logical_or(both_pos, both_neg)

    # Positive = point is inside region
    signed_dist = jnp.where(
        same_sign,
        jnp.where(both_pos, jnp.minimum(signed_dist_x, signed_dist_y), -dist_same_sign),
        jnp.where(signed_dist_x < 0, signed_dist_x, signed_dist_y),
    )

    return -signed_dist

def get_signed_dist(ij, particle_idx, particle_centers, particle_widths):
    center = particle_centers[particle_idx]
    intersection_point = get_ray_plane_intersection(ij, center)
    signed_dist = get_signed_dist_from_int_point(intersection_point, center, particle_widths[particle_idx])
    return signed_dist

def ray_from_ij(i,j, fx, fy, cx, cy):
    x = (j - cx) / fx
    y = (i - cy) / fy
    return jnp.array([x, y, 1])

img = jax.vmap(get_signed_dist, in_axes=(0, None, None, None))(
    all_pairs(100, 100), 0, particle_centers, particle_widths
).reshape(100, 100)
rr.log("signed_dist", rr.Image(img))