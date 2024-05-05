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

import demos.differentiable_renderer.clean.utils as utils
import demos.differentiable_renderer.clean.rendering as rendering
import demos.differentiable_renderer.clean.likelihoods as likelihoods

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

#############################
# Visualize scene + renders #
#############################

# Get triangle "mesh" for the scene:
(
    vertices, faces, vertex_colors, triangle_colors,
    triangle_to_particle_index, color_image
) = utils.get_mesh_and_gt_render(
    particle_centers, particle_widths, particle_colors
)

# visualize the scene
rr.init("differentiable_rendering--scene")
rr.connect("127.0.0.1:8812")
rr.log("scene/triangles", rr.Mesh3D(vertex_positions=vertices, indices=faces, vertex_colors=vertex_colors), timeless=True)
rr.log("scene/camera", rr.Pinhole(focal_length=rendering.fx, width=rendering.image_width, height=rendering.image_height), timeless=True)
rr.log("img/opengl_rendering", rr.Image(color_image), timeless=True)

# Stochastic rendering of scene
def get_render(key, weights, colors):
    lab_color_space_noise_scale = 3.0
    return likelihoods.mixture_rgb_sensor_model.simulate(
        key, (weights, colors, lab_color_space_noise_scale)
    ).get_retval().reshape(100, 100, 3)

SIGMA = 1e-4
GAMMA = 0.4
EPSILON = 1e-5
hyperparams = (SIGMA, GAMMA, EPSILON)
(weights, colors) = rendering.render_to_dist_parameters(
    vertices, faces, triangle_colors, hyperparams
)

# Generate + visualize 100 stochastic renders
keys = jax.random.split(jax.random.PRNGKey(0), 100)
renders = jax.vmap(get_render, in_axes=(0, None, None))(
    keys, weights, colors
)
for t in range(100):
    rr.set_time_sequence("stochastic_render", t)
    rr.log("img/stochastic_render", rr.Image(renders[t, ...]))
    rr.log("scene/camera", rr.Image(renders[t, ...]))

### Visualize how the likelihood changes as we move a square around ###
def get_img_logpdf(key, img, weights, colors):
    choicemap = genjax.vector_choice_map(genjax.choice(img.reshape(-1, 3)))
    tr, w = likelihoods.mixture_rgb_sensor_model.importance(key, choicemap, (weights, colors, 3.))
    return w

def render_to_dist_from_centers(new_particle_centers):
    particle_center_delta  = new_particle_centers - particle_centers
    new_vertices = vertices.reshape(3, 4, 3) + jnp.expand_dims(particle_center_delta, 1)
    weights, colors = rendering.render_to_dist_parameters(
        new_vertices.reshape(-1, 3), faces.reshape(-1, 3),
        particle_colors[triangle_to_particle_index], hyperparams
    )
    return weights, colors

def compute_logpdf(centers):
    weights, colors = render_to_dist_from_centers(centers)
    return get_img_logpdf(jax.random.PRNGKey(0), color_image, weights, colors)    

@jax.jit
def square2_pos_to_logpdf(xy):
    x, y = xy
    return compute_logpdf(jnp.array([[0.0, 0.0, 1.0],
                                     [x, y, 2.0],
                                     [0., 0., 5.]]))
x = jnp.linspace(-0.4, 0.6, 140)
y = jnp.linspace(-0.4, 0.6, 140)
xy = jnp.stack(jnp.meshgrid(x, y), axis=-1).reshape(-1, 2)
# This may take a couple minutes -- we can't vmap this call yet
# due to unfinished paths in the opengl renderer
logpdfs = jnp.array([square2_pos_to_logpdf(_xy) for _xy in xy])
logpdfs = logpdfs.reshape(140, 140)
rr.log("logpdfs_of_opengl_rendering_as_green_square_moves/logpdfs", rr.DepthImage(logpdfs), timeless=True)
lt = jnp.linalg.norm(xy - jnp.array([0.2, 0.2]), axis=-1) < 0.01

# mark the true position of this square
idx = jnp.where(lt)[0]
ij = jnp.unravel_index(idx, (140, 140))
ij2 = jnp.array([jnp.mean(ij[0]), jnp.mean(ij[0])])
rr.log("logpdfs_of_opengl_rendering_as_green_square_moves/true_object_position", rr.Points2D(jnp.array([ij2]), radii=1.0, colors=jnp.array([0, 0, 0])), timeless=True)

#####################
### Scene fitting ###
#####################