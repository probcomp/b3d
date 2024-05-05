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

image_width = 100
image_height = 100
fx = 50.0
fy = 50.0
cx = 50.0
cy = 50.0
near = 0.001
far = 16.0
renderer = b3d.Renderer(image_width, image_height, fx, fy, cx, cy, near, far)
WINDOW = 3
SIGMA = 0.03
GAMMA = 1e-4
EPSILON = 1e-5

def render(
        vertices,
        faces,
        triangle_colors
):
    uvs, _, triangle_id_image, depth_image = renderer.rasterize(
        Pose.identity()[None, ...], vertices, faces, jnp.array([[0, len(faces)]])
    )

    triangle_intersected_padded = jnp.pad(
        triangle_id_image, pad_width=[(WINDOW, WINDOW)], constant_values=-1
    )

    colors = jax.vmap(_get_pixel_color, in_axes=(0, None))(
        all_pairs(image_height, image_width),
        (vertices, faces, triangle_colors, triangle_intersected_padded)
    ).reshape(image_height, image_width, 3)
    
    return colors

def all_pairs(X, Y):
    return jnp.stack(jnp.meshgrid(jnp.arange(X), jnp.arange(Y)), axis=-1).reshape(-1, 2)

def get_weights(ij, vertices, faces, triangle_intersected_padded):
    """
    Returns pair (unique_triangle_indices, weights).
    `unique_triangle_indices` will have:
    - one token -10 for the background, at index 0
    - one token -2 of padding (ignore these)
    - the rest are the indices of the unique triangles in the window

    `weights` will have the same length as `unique_triangle_indices` and will have the weights for each triangle.
    Will be 0 in every slot where `unique_triangle_indices` is -2.
    """
    triangle_intersected_padded_in_window = jax.lax.dynamic_slice(
        triangle_intersected_padded,
        (ij[0], ij[1]),
        (2 * WINDOW + 1, 2 * WINDOW + 1),
    )
    # This will have the value -2 in slots we should ignore
    # and -1 in slots which hit the background.
    unique_triangle_values = jnp.unique(
        triangle_intersected_padded_in_window + 1, size=triangle_intersected_padded_in_window.size
    ) - 1
    unique_triangle_values_safe = jnp.where(unique_triangle_values < 0, 0, unique_triangle_values)
    
    signed_dist_values = get_signed_dists(ij, unique_triangle_values_safe, vertices, faces)
    z_values = get_z_values(ij, unique_triangle_values_safe, vertices, faces)
    
    # Math from the softras paper
    signed_dist_scores = jax.nn.sigmoid(jnp.sign(signed_dist_values) * signed_dist_values ** 2 / SIGMA)

    # following https://github.com/kach/softraxterizer/blob/main/softraxterizer.py
    z = (z_values.max() - z_values) / (z_values.max() - z_values.min())
    zexp = jnp.exp(jnp.clip(z / GAMMA, -20., 20.))

    unnorm_weights = signed_dist_scores * zexp

    # filter out the padding
    unnorm_weights = jnp.where(unique_triangle_values >= 0, unnorm_weights, 0.0)
    unnorm_weights = jnp.concatenate([jnp.array([EPSILON/GAMMA]), unnorm_weights])
    weights = unnorm_weights / jnp.sum(unnorm_weights)
    
    return (jnp.concatenate([jnp.array([-10]), unique_triangle_values]), weights)

def get_pixel_color(
        ij, vertices, faces, triangle_colors, triangle_intersected_padded,
        background_color = jnp.array([0., 0., 0.])    
    ):
    unique_triangle_indices, weights = get_weights(ij, vertices, faces, triangle_intersected_padded)
    extended_triangle_colors = jnp.concatenate([jnp.array([background_color]), triangle_colors], axis=0)
    colors = extended_triangle_colors[jnp.where(unique_triangle_indices < 0, 0, unique_triangle_indices)]
    colors = colors.at[0, :].set(background_color)
    color = (weights[..., None] * colors).sum(axis=0)
    return color

def get_z_values(ij, unique_triangle_values, vertices, faces):
    return jax.vmap(get_z_value, in_axes=(NotImplementedError, 0, None, None))(
        ij, unique_triangle_values, vertices, faces
    )
def get_z_value(ij, triangle_idx, vertices, faces):
    triangle = vertices[faces[triangle_idx]] # 3 x 3 (face_idx, vertex_idx)
    point_on_plane = project_pixel_to_plane(ij, triangle)
    return point_on_plane[2]

def get_signed_dists(ij, unique_triangle_values, vertices, faces):
    return jax.vmap(get_signed_dist, in_axes=(NotImplementedError, 0, None, None))(
        ij, unique_triangle_values, vertices, faces
    )

def get_signed_dist(ij, triangle_idx, vertices, faces):
    triangle = vertices[faces[triangle_idx]] # 3 x 3 (face_idx, vertex_idx)
    point_on_plane = project_pixel_to_plane(ij, triangle)
    
    # distances to 3 lines making up the triangle
    d1 = dist_to_line_seg(triangle[0], triangle[1], point_on_plane)
    d2 = dist_to_line_seg(triangle[1], triangle[2], point_on_plane)
    d3 = dist_to_line_seg(triangle[2], triangle[0], point_on_plane)
    d = jnp.minimum(d1, jnp.minimum(d2, d3))

    in_triangle = get_in_triangle(triangle, point_on_plane)

    return jnp.where(in_triangle, d, -d)

# From ChatGPT
# TODO: test this
def project_pixel_to_plane(ij, triangle):
    x, y = ij - jnp.array([cx, cy])
    vertex1, vertex2, vertex3 = triangle

    # Calculate the normal vector of the plane
    normal = jnp.cross(vertex2 - vertex1, vertex3 - vertex1)
    normal /= jnp.linalg.norm(normal)

    # Calculate the distance from the origin to the plane
    d = -jnp.dot(normal, vertex1)

    # Calculate the direction vector from the camera center to the pixel
    direction = jnp.array([x, y, -fx])

    # Calculate the distance from the camera center to the intersection point
    t = -(jnp.dot(normal, jnp.array([0, 0, d])) / jnp.dot(normal, direction))

    # Calculate the intersection point
    intersection_point = jnp.array([0, 0, d]) + t * direction

    return intersection_point

def _get_pixel_color(ij, args):
    return get_pixel_color(ij, *args)

# The functions below here are following
# https://github.com/kach/softraxterizer/blob/main/softraxterizer.py
def dist_to_line_seg(a, b, p):
    Va = b - a
    Vp = p - a
    projln = Vp.dot(Va) / Va.dot(Va)
    projln = jnp.clip(projln, 0., 1.)
    return jnp.linalg.norm(Vp - projln * Va)

def get_in_triangle(triangle, point):
    a = _signed_area_to_point(triangle[0], triangle[1], point)
    b = _signed_area_to_point(triangle[1], triangle[2], point)
    c = _signed_area_to_point(triangle[2], triangle[0], point)
    return jnp.logical_and(
        jnp.equal(jnp.sign(a), jnp.sign(b)),
        jnp.equal(jnp.sign(b), jnp.sign(c))
    )

def _signed_area_to_point(a, b, p):
    Va = b - a
    area = jnp.cross(Va, p - a) / 2
    return area