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

def get_mesh_and_gt_render(renderer, particle_centers, particle_widths, particle_colors):
    vertices_og, faces, colors, triangle_to_particle_index = jax.vmap(
        b3d.square_center_width_color_to_vertices_faces_colors
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
