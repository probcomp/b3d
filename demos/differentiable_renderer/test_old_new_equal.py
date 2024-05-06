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

import b3d._differentiable_renderer_old as rendering_old
import b3d.differentiable_renderer as rendering
import b3d.likelihoods as likelihoods
import demos.differentiable_renderer.utils as utils

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
rr.init("differentiable_rendering--test_old_new_renderer")
rr.connect("127.0.0.1:8812")
rr.log("scene/triangles", rr.Mesh3D(vertex_positions=vertices, indices=faces, vertex_colors=vertex_colors), timeless=True)
rr.log("scene/camera", rr.Pinhole(focal_length=rendering.fx, width=rendering.image_width, height=rendering.image_height), timeless=True)
rr.log("img/opengl_rendering", rr.Image(color_image), timeless=True)

# Soft rendering of scene
SIGMA = 1e-4
GAMMA = 0.4
EPSILON = 1e-5
hyperparams = (SIGMA, GAMMA, EPSILON)
soft_img_old = rendering_old.render(
    vertices, faces, triangle_colors, hyperparams
)
soft_img_new = rendering.render_to_averaged_attributes(
    vertices, faces, vertex_colors, hyperparams,
    background_attribute=jnp.array([0.1, 0.1, 0.1])
)

# Check that the old and new renderer do the same thing
rr.log("img/soft_rendering_old", rr.Image(soft_img_old), timeless=True)
rr.log("img/soft_rendering_new", rr.Image(soft_img_new), timeless=True)
rr.log("img/err", rr.Image(jnp.abs(soft_img_new - soft_img_old)), timeless=True)
assert jnp.all(jnp.abs(soft_img_new - soft_img_old) < 1e-4)