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
renderer = b3d.Renderer(image_width, image_height, fx, fy, cx, cy, near, far)\

def center_and_width_to_vertices_faces_colors(i, center, width, color):
    vertices = (
        jnp.array(
            [
                [-0.5, -0.5, 0.0],
                [0.5, -0.5, 0.0],
                [0.5, 0.5, 0.0],
                [-0.5, 0.5, 0.0],
            ]
        )
        * width
        + center
    )
    faces = (
        jnp.array(
            [
                [0, 1, 2],
                [0, 2, 3],
            ]
        )
        + 4 * i
    )
    colors = jnp.ones((4, 3)) * color
    return vertices, faces, colors, jnp.ones(len(faces), dtype=jnp.int32) * i


def ray_from_ij(i,j, fx, fy, cx, cy):
    x = (j - cx) / fx
    y = (i - cy) / fy
    return jnp.array([x, y, 1])

WINDOW = 3

#############

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

#############

def get_pixel_color_from_ij_v3(
    ij, particle_centers, particle_widths, particle_colors, point_of_intersection_padded, particle_intersected_padded
):
    i,j = ij
    particle_intersected_padded_in_window = jax.lax.dynamic_slice(
        particle_intersected_padded,
        (ij[0], ij[1]),
        (2 * WINDOW + 1, 2 * WINDOW + 1),
    )

    unique_particle_values = jnp.unique(particle_intersected_padded_in_window)

    offset_window = all_pairs(2 * WINDOW + 1, 2 * WINDOW + 1)
    # ij_window = offset_window + jnp.array([i - 2*WINDOW - 1, j - 2*WINDOW - 1])

    signed_dist_from_square_boundary = jax.vmap(get_signed_dist, in_axes=(None, 0, None, None))(
        ij, unique_particle_values, particle_centers, particle_widths
    )

    # (2W + 1, 2W + 1)
    # z_dists = ij_plane_intersections[:, :, 2]

    # Signed_dist_score 
    signed_dist_score = 0.5 * (1 - jnp.tanh(50 * (signed_dist_from_square_boundary)))

    signed_dist_score = jnp.where(
        unique_particle_values == -1,
        jnp.ones_like(signed_dist_score) * 0.02,
        signed_dist_score
    )

    # z score
    # z_score = 0.5 * 1/(z_dists**2 + 1)

    total_scores = signed_dist_score + 1e-10 # + z_score

    normalized_scores = total_scores / total_scores.sum()

    blank_color = jnp.array([0.1, 0.1, 0.1]) # gray for unintersected particles
    extended_colors = jnp.concatenate([jnp.array([blank_color]), particle_colors], axis=0)
    colors_in_window = extended_colors[particle_intersected_padded_in_window + 1]
    color = (normalized_scores[..., None] * colors_in_window).sum(axis=(0, 1))
    color = jnp.minimum(color, jnp.ones(3))
    # color = jnp.where(
    #     jnp.isnan(color),
    #     blank_color,
    #     color
    # )
    return color 


#############

def get_pixel_color_from_ij_v2(
    ij, particle_centers, particle_widths, particle_colors, point_of_intersection_padded, particle_intersected_padded
):
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

    # z score
    # z_score = 0.5 * 1/(z_dists**2 + 1)

    total_scores = signed_dist_score + 1e-10 # + z_score

    normalized_scores = total_scores / total_scores.sum()

    blank_color = jnp.array([0.1, 0.1, 0.1]) # gray for unintersected particles
    extended_colors = jnp.concatenate([jnp.array([blank_color]), particle_colors], axis=0)
    colors_in_window = extended_colors[particle_intersected_padded_in_window + 1]
    color = (normalized_scores[..., None] * colors_in_window).sum(axis=(0, 1))
    color = jnp.minimum(color, jnp.ones(3))
    # color = jnp.where(
    #     jnp.isnan(color),
    #     blank_color,
    #     color
    # )
    return color 

def _get_pixel_color_from_ij_v2(ij, args):
    return get_pixel_color_from_ij_v2(ij, *args)


def get_pixel_color_from_ij(
    ij, particle_centers, particle_width, particle_colors, point_of_intersection_padded, particle_intersected_padded
):
    i,j = ij
    particle_intersected_padded_in_window = jax.lax.dynamic_slice(
        particle_intersected_padded,
        (ij[0], ij[1]),
        (2 * WINDOW + 1, 2 * WINDOW + 1),
    )

    ray_through_center_pixel = ray_from_ij(i, j, fx, fy, cx, cy)

    # point_of_intersection_padded_in_window = jax.vmap()(
    #     jnp.arange()
    #     particle_intersected_padded_in_window
    # )

    point_of_intersection_padded_in_window = jax.lax.dynamic_slice(
        point_of_intersection_padded,
        (ij[0], ij[1], 0),
        (2 * WINDOW + 1, 2 * WINDOW + 1, 3),
    )

    # This is useless, I think, since it should just project to the pixel center, which shouldn't depend on the triangle's pose at all...
    rays_on_image_plane = point_of_intersection_padded_in_window / (point_of_intersection_padded_in_window[...,2][...,None] + 1e-10)

    distances = (
        jnp.linalg.norm(rays_on_image_plane - ray_through_center_pixel[None, None, :], axis=-1) +
        1000.0 * (particle_intersected_padded_in_window == -1)
    )

    weight = 1 / (distances**2 + 1e-4)
    weight_normalized = weight / (weight.sum() + 1e-10)
    print(weight_normalized)

    blank_color = jnp.array([0.1, 0.1, 0.1]) # gray for unintersected particles
    extended_colors = jnp.concatenate([jnp.array([blank_color]), particle_colors], axis=0)
    colors_in_window = extended_colors[particle_intersected_padded_in_window + 1]
    color = (weight_normalized[..., None] * colors_in_window).sum(axis=(0, 1))
    color = jnp.minimum(color, jnp.ones(3))
    return color #(point_of_intersection_padded[i, j], particle_intersected_padded[i, j])

def _get_pixel_color_from_ij(ij, args):
    return get_pixel_color_from_ij(ij, *args)

def all_pairs(X, Y):
    return jnp.stack(jnp.meshgrid(jnp.arange(X), jnp.arange(Y)), axis=-1).reshape(-1, 2)

def render(
    particle_centers,
    particle_widths,
    particle_colors
):
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

    point_of_intersection = b3d.xyz_from_depth(depth_image, fx, fy, cx, cy)
    particle_intersected = triangle_to_particle_index[triangle_id_image - 1] * (triangle_id_image > 0) + -1 * (triangle_id_image ==0 )

    # rr.log("particle_intersected", rr.DepthImage(particle_intersected))

    WINDOW = 3
    particle_intersected_padded = jnp.pad(
        particle_intersected, pad_width=[(WINDOW, WINDOW)], constant_values=-1
    )
    point_of_intersection_padded = jnp.pad(
        point_of_intersection, pad_width=[(WINDOW, WINDOW), (WINDOW, WINDOW), (0, 0)]
    )

    colors = jax.vmap(_get_pixel_color_from_ij_v2, in_axes=(0, None))(
        all_pairs(image_height, image_width),
        (particle_centers, particle_widths, particle_colors, point_of_intersection_padded, particle_intersected_padded)
    ).reshape(image_height, image_width, 3)
    return colors

###
def rr_log_gt(name, particle_centers, particle_widths, particle_colors):
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
    rr.log(f"{name}/triangle_id_image", rr.DepthImage(triangle_id_image), timeless=True)
    rr.log(f"{name}/depth_image", rr.DepthImage(depth_image), timeless=True)

    point_of_intersection = b3d.xyz_from_depth(depth_image, fx, fy, cx, cy)
    particle_intersected = triangle_to_particle_index[triangle_id_image - 1] * (triangle_id_image > 0) + -1 * (triangle_id_image ==0 )

    rr.log(f"{name}/particle_intersected", rr.DepthImage(particle_intersected), timeless=True)

    blank_color = jnp.array([0.1, 0.1, 0.1]) # gray for unintersected particles
    extended_colors = jnp.concatenate([jnp.array([blank_color]), particle_colors], axis=0)
    color_image = extended_colors[particle_intersected + 1]
    rr.log(f"{name}/color_image", rr.Image(color_image), timeless=True)