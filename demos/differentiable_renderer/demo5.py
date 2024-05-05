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

jax.config.update("jax_debug_nans", False)
jax.config.update("jax_enable_x64", False)


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

rr.init("softras_5")
rr.connect("127.0.0.1:8812")

rr_log_gt("gt", particle_centers, particle_widths, particle_colors)

SIGMA = 1e-4
GAMMA = 1e-4
EPSILON = 1e-5
hyperparams = (SIGMA, GAMMA, EPSILON)
rendered_soft = render(vertices, faces, triangle_colors, hyperparams)
rr.log("c/gt", rr.Image(color_image), timeless=True)
rr.log("c/rendered", rr.Image(rendered_soft), timeless=True)

###

def compute_error(centers):
    rendered = render_from_centers(centers)
    return jnp.sum(jnp.abs((rendered - rendered_soft)))

def render_from_centers(new_particle_centers):
    particle_center_delta  = new_particle_centers - particle_centers
    # vertices, faces, colors, triangle_to_particle_index = jax.vmap(
    #     center_and_width_to_vertices_faces_colors
    # )(jnp.arange(len(new_particle_centers)), new_particle_centers, particle_widths, particle_colors)
    # vertices = vertices.reshape(-1, 3)
    # faces = faces.reshape(-1, 3)
    # triangle_to_particle_index = triangle_to_particle_index.reshape(-1)
    new_vertices = vertices_og + jnp.expand_dims(particle_center_delta, 1)
    return render(new_vertices.reshape(-1, 3), faces.reshape(-1, 3), particle_colors[triangle_to_particle_index], hyperparams)

particle_centers_shifted = jnp.array(
    [
        [0.05, 0.0, 1.0],
        [0.15, 0.2, 2.0],
        [0., 0., 5.]
    ]
)
rendered_shifted = render_from_centers(particle_centers_shifted)
rr.log("shifted", rr.Image(rendered_shifted), timeless=True)

print("ERROR:")
print(compute_error(particle_centers_shifted))
print("GRAD:")
g = jax.grad(compute_error)(particle_centers_shifted)
print(g)

#########
def compute_error_from_vertices(vertices):
    rendered = render(vertices.reshape(-1, 3), faces, particle_colors[triangle_to_particle_index], hyperparams)
    return jnp.sum(jnp.abs((rendered - rendered_soft)))

particle_center_delta  = particle_centers_shifted - particle_centers
vertices_shifted = vertices_og + jnp.expand_dims(particle_center_delta, 1)

print("ERROR:")
print(compute_error_from_vertices(vertices_shifted))
print("GRAD:")
g = jax.grad(compute_error_from_vertices)(vertices_shifted)
print(g)

uvs, _, triangle_id_image, depth_image = renderer.rasterize(
    Pose.identity()[None, ...], vertices, faces, jnp.array([[0, len(faces)]])
)

triangle_intersected_padded = jnp.pad(
    triangle_id_image, pad_width=[(WINDOW, WINDOW)], constant_values=-1
)

ij = jnp.array([54, 47])
def get_pixel_color_from_vertices(ij, vertices):
    return get_pixel_color(
        ij, vertices, faces, triangle_colors, triangle_intersected_padded,
        hyperparams
    ).sum()
color = get_pixel_color_from_vertices(vertices_shifted.reshape(-1, 3))
grads = jax.vmap(
    jax.grad(get_pixel_color_from_vertices, argnums=1),
    in_axes=(0, None)
)(all_pairs(100, 100), vertices_shifted.reshape(-1, 3))
isnan_img = jnp.any(jnp.isnan(grads), axis=(1, 2)).reshape(100, 100).astype(float)
rr.log("isnan_img", rr.DepthImage(isnan_img))

jax.grad(get_pixel_color_from_vertices, argnums=1)(ij, vertices_shifted.reshape(-1, 3))

#######

grad_jit = jax.jit(jax.grad(compute_error))
current_centers = particle_centers_shifted
eps = 1e-5
for i in range(40):
    g = grad_jit(current_centers)
    current_centers = current_centers - eps * g
        # rr.log("error", rr.Scalar(compute_error(current_centers)))
    print(f"ERROR: {compute_error(current_centers)}")
    rr.set_time_sequence("gd", i)
    rendered = render_from_centers(current_centers)
    rr.log("gd", rr.Image(rendered))
    rr.log("gd-error", rr.Image(jnp.abs((rendered - rendered_soft))))