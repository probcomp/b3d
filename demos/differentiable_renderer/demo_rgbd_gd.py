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

# Set up 3 squares oriented toward the camera, with different colors
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

# Get triangle "mesh" for the scene:
(
    vertices, faces, vertex_colors, triangle_colors,
    triangle_to_particle_index, color_image
) = utils.get_mesh_and_gt_render(
    renderer, particle_centers, particle_widths, particle_colors
)
rgb_image_opengl, depth_image_opengl = renderer.render_attribute(
    Pose.identity()[None, ...],
    vertices, faces, jnp.array([[0, len(faces)]]),
    vertex_colors
)

rr.init("differentiable_rendering--rgbd_gd")
rr.connect("127.0.0.1:8812")
rr.log("/scene/ground_truth", rr.Mesh3D(vertex_positions=vertices, indices=faces, vertex_colors=vertex_colors), timeless=True)
rr.log("/scene/camera", rr.Pinhole(focal_length=fx, width=image_width, height=image_height), timeless=True)
rr.log("/img/opengl_rendering/rgb", rr.Image(rgb_image_opengl), timeless=True)
rr.log("/img/opengl_rendering/depth", rr.Image(depth_image_opengl), timeless=True)

WINDOW = 3
SIGMA = 5e-5
GAMMA = 0.25
EPSILON = -1
hyperparams = rendering.DifferentiableRendererHyperparams(WINDOW, SIGMA, GAMMA, EPSILON)

def get_img_logpdf(key, img, weights, colors):
    choicemap = genjax.vector_choice_map(genjax.vector_choice_map(genjax.choice(img)))
    tr, w = likelihoods.mixture_rgb_sensor_model.importance(key, choicemap, (weights, colors, 3.))
    return w

def render_to_dist_from_centers(new_particle_centers):
    particle_center_delta  = new_particle_centers - particle_centers
    new_vertices = vertices.reshape(3, 4, 3) + jnp.expand_dims(particle_center_delta, 1)
    weights, colors = rendering.render_to_dist_params(
        renderer,
        new_vertices.reshape(-1, 3), faces.reshape(-1, 3),
        vertex_colors, hyperparams
    )
    return weights, colors

def render_from_centers(new_particle_centers):
    particle_center_delta  = new_particle_centers - particle_centers
    new_vertices = vertices.reshape(3, 4, 3) + jnp.expand_dims(particle_center_delta, 1)
    return rendering.render_to_average(
        renderer,
        new_vertices.reshape(-1, 3), faces.reshape(-1, 3),
        vertex_colors,
        background_attribute=jnp.array([0., 0., 0.]),
        hyperparams=hyperparams
    )

def compute_logpdf(centers):
    weights, colors = render_to_dist_from_centers(centers)
    return get_img_logpdf(jax.random.PRNGKey(0), color_image, weights, colors)    


