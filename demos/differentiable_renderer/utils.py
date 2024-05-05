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

def get_mesh_and_gt_render(particle_centers, particle_widths, particle_colors):
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

    return (vertices, faces, colors, triangle_colors, triangle_to_particle_index, color_image)

def ray_from_ij(i,j, fx, fy, cx, cy):
    x = (j - cx) / fx
    y = (i - cy) / fy
    return jnp.array([x, y, 1])

WINDOW = 3

def all_pairs(X, Y):
    return jnp.stack(jnp.meshgrid(jnp.arange(X), jnp.arange(Y)), axis=-1).reshape(-1, 2)