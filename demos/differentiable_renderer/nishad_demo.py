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
import genjax

import b3d.differentiable_renderer as rendering
import b3d.likelihoods as likelihoods
import demos.differentiable_renderer.utils as utils

rr.init("gradients")
rr.connect("127.0.0.1:8812")

# Set up OpenGL renderer
image_width = 100
image_height = 100
fx = 50.0
fy = 50.0
cx = 50.0
cy = 50.0
near = 0.001
far = 16.0
renderer = b3d.Renderer(image_width, image_height, fx, fy, cx, cy, near, far)


WINDOW = 5

###########
@functools.partial(
    jnp.vectorize,
    signature="(m)->(k)",
    excluded=(
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
    ),
)
def get_pixel_color(ij, triangle_intersected_padded, particle_centers, particle_widths, particle_colors, triangle_index_to_particle_index, vertices, faces, vertex_colors):
    triangle_intersected_padded_in_window = jax.lax.dynamic_slice(
        triangle_intersected_padded,
        (ij[0], ij[1]),
        (2 * WINDOW + 1, 2 * WINDOW + 1),
    )
    particle_indices_in_window = triangle_index_to_particle_index[triangle_intersected_padded_in_window - 1]
    particle_centers_in_window = particle_centers[particle_indices_in_window]
    particle_widths_in_window = particle_widths[particle_indices_in_window]
    # return jnp.sum(particle_centers_in_window, axis=[0,1]) # checksout

    particle_centers_on_image_plane = particle_centers_in_window[..., :2] / particle_centers_in_window[..., 2:]
    particle_pixel_coordinate = particle_centers_on_image_plane * jnp.array([fx, fy]) + jnp.array([cx, cy])
    # return jnp.sum(particle_pixel_coordinate, axis=[0,1]) # checks out

    distance_on_image_plane = jnp.linalg.norm(particle_pixel_coordinate - ij + 1e-5, axis=-1) 

    particle_width_on_image_plane = particle_widths_in_window / (particle_centers_in_window[..., 2] + 1e-10) + 1e-10
    # return jnp.array([jnp.sum(particle_width_on_image_plane, axis=[0,1])]) # checks out


    scaled_distance_on_image_plane = distance_on_image_plane / particle_width_on_image_plane
    scaled_distance_on_image_plane_masked = scaled_distance_on_image_plane * (triangle_intersected_padded_in_window > 0)
    # return jnp.array([jnp.sum(scaled_distance_on_image_plane_masked, axis=[0,1])])

    influence = jnp.exp(- (scaled_distance_on_image_plane_masked ** 2))

    particle_colors_in_window = particle_colors[particle_indices_in_window]
    particle_colors_in_window_masked = particle_colors_in_window * (triangle_intersected_padded_in_window > 0)[...,None]
    
    return jnp.sum(particle_colors_in_window_masked * influence[...,None], axis=[0,1]) / influence.size


def render(particle_centers, particle_widths, particle_colors):
    # Get triangle "mesh" for the scene:
    vertices, faces, vertex_colors, triangle_index_to_particle_index = jax.vmap(
        b3d.particle_center_width_color_to_vertices_faces_colors
    )(jnp.arange(len(particle_centers)), particle_centers, particle_widths, particle_colors)
    vertices = vertices.reshape(-1, 3)
    faces = faces.reshape(-1, 3)
    vertex_colors = vertex_colors.reshape(-1, 3)
    triangle_index_to_particle_index = triangle_index_to_particle_index.reshape(-1)


    uvs, _, triangle_id_image, depth_image = renderer.rasterize(
        Pose.identity()[None, ...], vertices, faces, jnp.array([[0, len(faces)]])
    )

    triangle_intersected_padded = jnp.pad(
        triangle_id_image, pad_width=[(WINDOW, WINDOW)], constant_values=0
    )

    image_size = (renderer.height, renderer.width)
    jj, ii = jnp.meshgrid(
        jnp.arange(image_size[1]), jnp.arange(image_size[0])
    )
    ijs = jnp.stack([ii, jj], axis=-1)

    image = get_pixel_color(ijs, triangle_intersected_padded, particle_centers, particle_widths, particle_colors, triangle_index_to_particle_index, vertices, faces, vertex_colors)

    return jnp.clip(image, 0.0, 1.0)


render_jit = jax.jit(render)

# Set up 3 squares oriented toward the camera, with different colors
gt_particle_centers = jnp.array(
    [
        [0.0, 0.0, 0.1],
    ]
)
gt_particle_widths = jnp.array([0.9])
gt_particle_colors = jnp.array(
    [
        [1.0, 0.0, 0.0],
    ]
)
particle_centers, particle_widths, particle_colors = gt_particle_centers, gt_particle_widths, gt_particle_colors
# 3 unit tests on 3 arguments

gt_image = render_jit(gt_particle_centers, gt_particle_widths, gt_particle_colors)
loss_func_grad = jax.jit(jax.value_and_grad(lambda a,b,c: render(a,b,c).sum(), argnums=(0,)))
loss_func_grad(particle_centers, gt_particle_widths, gt_particle_colors)
rr.log('image', rr.Image(gt_image[...,:3]), timeless=True)


# Set up 3 squares oriented toward the camera, with different colors
particle_centers = jnp.array(
    [
        [0.02, 0.02, 0.2],
    ]
)
particle_widths = jnp.array([0.4])
particle_colors = jnp.array([[0.3, 0.4, 0.9]])

# 3 unit tests on 3 arguments
image = render_jit(particle_centers, particle_widths, particle_colors)
rr.log('image/reconstruction', rr.Image(image[...,:3]))

loss_func = lambda a,b,c,gt: jnp.mean(jnp.abs(render_jit(a,b,c) - gt))
loss_func_grad = jax.jit(jax.value_and_grad(loss_func, argnums=(0,1,2)))
loss_func_grad(particle_centers, particle_widths, particle_colors, gt_image)

for t in range(100):
    rr.set_time_sequence("frame", t)
    loss, (grad_centers,grad_widths, grad_colors) = loss_func_grad(particle_centers, particle_widths, particle_colors, gt_image)
    particle_centers -= 1e-1 * grad_centers
    particle_widths -= 1e1 * grad_widths
    particle_colors -= 1e2 * grad_colors
    image = render_jit(particle_centers, particle_widths, particle_colors)
    rr.log('image/reconstruction', rr.Image(image[...,:3]))



gt_triangle_image, _, _ = rasterize(gt_particle_centers, gt_particle_widths, gt_particle_colors)
rr.log('triangles', rr.DepthImage(gt_triangle_image))


def loss(particle_centers, particle_widths, particle_colors):
    # Get triangle "mesh" for the scene:
    vertices, faces, vertex_colors, triangle_index_to_particle_index = jax.vmap(
        b3d.particle_center_width_color_to_vertices_faces_colors
    )(jnp.arange(len(particle_centers)), particle_centers, particle_widths, particle_colors)
    vertices = vertices.reshape(-1, 3)
    faces = faces.reshape(-1, 3)
    vertex_colors = vertex_colors.reshape(-1, 3)
    triangle_index_to_particle_index = triangle_index_to_particle_index.reshape(-1)
    return jnp.mean(vertices)


def loss(x):
    return jnp.linalg.norm(x)

loss_grad = jax.jit(jax.grad(loss))
loss_grad(jnp.zeros(3))

import torch

def f(x):
    return torch.linalg.norm(x)

x = torch.zeros(3,requires_grad=True)
y = f(x)
y.backward()
x.grad

# def render(particle_centers, particle_widths, particle_colors):
#     # Get triangle "mesh" for the scene:
#     vertices, faces, vertex_colors, _ = jax.vmap(
#         b3d.particle_center_width_color_to_vertices_faces_colors
#     )(jnp.arange(len(particle_centers)), particle_centers, particle_widths, particle_colors)
#     vertices = vertices.reshape(-1, 3)
#     faces = faces.reshape(-1, 3)
#     vertex_colors = vertex_colors.reshape(-1, 3)

#     WINDOW = 6
#     SIGMA = 5e-5
#     GAMMA = 0.25
#     EPSILON = -1
#     hyperparams = rendering.DifferentiableRendererHyperparams(WINDOW, SIGMA, GAMMA, EPSILON)
#     image = rendering.render_to_average_rgbd(
#         renderer, vertices, faces, vertex_colors, hyperparams=hyperparams
#     )
#     return image

# render_jit = jax.jit(render)

# # Set up 3 squares oriented toward the camera, with different colors
# gt_particle_centers = jnp.array(
#     [
#         [0.0, 0.0, 1.0],
#     ]
# )
# gt_particle_widths = jnp.array([0.1])
# gt_particle_colors = jnp.array(
#     [
#         [1.0, 0.0, 0.0],
#     ]
# )

# # 3 unit tests on 3 arguments
# gt_image = render_jit(gt_particle_centers, gt_particle_widths, gt_particle_colors)

# rr.log('image', rr.Image(gt_image[...,:3]))


# # Set up 3 squares oriented toward the camera, with different colors
# particle_centers = jnp.array(
#     [
#         [0.2, 0.1, 1.0],
#     ]
# )

# # 3 unit tests on 3 arguments
# image = render_jit(particle_centers, gt_particle_widths, gt_particle_colors)
# rr.log('image/reconstruction', rr.Image(image[...,:3]))

# loss_func = lambda a,b,c,gt: jnp.mean(jnp.abs(render_jit(a,b,c) - gt))
# loss_func_grad = jax.jit(jax.value_and_grad(loss_func, argnums=(0,)))
# loss_func_grad(particle_centers, gt_particle_widths, gt_particle_colors, gt_image)


# def all_pairs(X, Y):
#     return jnp.stack(jnp.meshgrid(jnp.arange(X), jnp.arange(Y)), axis=-1).reshape(-1, 2)
