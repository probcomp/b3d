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

image_width=100
image_height=100
fx=50.0
fy=50.0
cx=50.0
cy=50.0
near=0.001
far=16.0
renderer = b3d.Renderer(
    image_width, image_height, fx, fy, cx, cy, near, far
)

def center_and_width_to_vertices_faces_colors(
    i, center, width, color
):
    vertices = jnp.array([
        [-0.5, -0.5, 0.0],
        [0.5, -0.5, 0.0],
        [0.5, 0.5, 0.0],
        [-0.5, 0.5, 0.0],
    ]) * width + center
    faces = jnp.array([
        [0, 1, 2],
        [0, 2, 3],
    ]) + 4*i
    colors = jnp.ones((4,3)) * color
    return vertices, faces, colors, jnp.ones(len(faces), dtype=jnp.int32) * i

particle_centers = jnp.array([
    [0.0, 0.0, 1.0],
    [0.2, 0.2, 1.0],
])
particle_widths = jnp.array([0.1, 0.3])
particle_colors = jnp.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])

vertices, faces, colors, triangle_to_particle_index = jax.vmap(center_and_width_to_vertices_faces_colors)(jnp.arange(len(particle_centers)), particle_centers, particle_widths, particle_colors)
vertices = vertices.reshape(-1, 3)
faces = faces.reshape(-1, 3)
colors = colors.reshape(-1, 3)
triangle_to_particle_index = triangle_to_particle_index.reshape(-1)
_, _, triangle_id_image, depth_image = renderer.rasterize(Pose.identity()[None,...], vertices, faces, jnp.array([[0, len(faces)]]))

point_of_intersection = b3d.xyz_from_depth(depth_image, fx,fy,cx,cy)
particle_intersected = triangle_to_particle_index[triangle_id_image - 1]

WINDOW = 3
particle_intersected_padded = jnp.pad(particle_intersected, pad_width=[(WINDOW, WINDOW)])

def get_value_from_pixel(i,j, point_of_intersection_padded, particle_intersected_padded):
    return (point_of_intersection_padded[i,j], particle_intersected_padded[i,j])




