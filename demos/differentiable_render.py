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

rr.init("diff_rendering")
rr.connect("127.0.0.1:8812")

image_width = 100
image_height = 100
fx = 50.0
fy = 50.0
cx = 50.0
cy = 50.0
near = 0.001
far = 16.0
renderer = b3d.Renderer(image_width, image_height, fx, fy, cx, cy, near, far)


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
rr.log("triangle_id_image", rr.DepthImage(triangle_id_image))
rr.log("depth_image", rr.DepthImage(depth_image))

point_of_intersection = b3d.xyz_from_depth(depth_image, fx, fy, cx, cy)
particle_intersected = triangle_to_particle_index[triangle_id_image - 1] * (triangle_id_image > 0) + -1 * (triangle_id_image ==0 )

rr.log("particle_intersected", rr.DepthImage(particle_intersected))

blank_color = jnp.array([0.1, 0.1, 0.1]) # gray for unintersected particles
extended_colors = jnp.concatenate([jnp.array([blank_color]), particle_colors], axis=0)
color_image = extended_colors[particle_intersected + 1]
rr.log("color_image", rr.Image(color_image))


WINDOW = 3
particle_intersected_padded = jnp.pad(
    particle_intersected, pad_width=[(WINDOW, WINDOW)], constant_values=-1
)
print(particle_intersected.shape)
print(particle_intersected_padded.shape)
point_of_intersection_padded = jnp.pad(
    point_of_intersection, pad_width=[(WINDOW, WINDOW), (WINDOW, WINDOW), (0, 0)]
)
print(point_of_intersection.shape)
print(point_of_intersection_padded.shape)

def ray_from_ij(i,j, fx, fy, cx, cy):
    x = (j - cx) / fx
    y = (i - cy) / fy
    return jnp.array([x, y, 1])

ij = jnp.array([52,53])

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

    point_of_intersection_padded_in_window = jax.lax.dynamic_slice(
        point_of_intersection_padded,
        (ij[0], ij[1], 0),
        (2 * WINDOW + 1, 2 * WINDOW + 1, 3),
    )

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
    particle_colors,
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

    rr.log("particle_intersected", rr.DepthImage(particle_intersected))

    WINDOW = 3
    particle_intersected_padded = jnp.pad(
        particle_intersected, pad_width=[(WINDOW, WINDOW)], constant_values=-1
    )
    point_of_intersection_padded = jnp.pad(
        point_of_intersection, pad_width=[(WINDOW, WINDOW), (WINDOW, WINDOW), (0, 0)]
    )
    colors = jax.vmap(_get_pixel_color_from_ij, in_axes=(0, None))(
        all_pairs(image_height, image_width),
        (particle_centers, particle_widths, particle_colors, point_of_intersection_padded, particle_intersected_padded)
    ).reshape(image_height, image_width, 3)
    return colors

colors = render(particle_centers, particle_widths, particle_colors)
rr.log("softened_image", rr.Image(colors))