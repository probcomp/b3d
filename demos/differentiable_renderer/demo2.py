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

from demos.differentiable_renderer.utils import (
    image_width, image_height, fx, fy, cx, cy, near, far, renderer,
    center_and_width_to_vertices_faces_colors, ray_from_ij, WINDOW,
    get_pixel_color_from_ij, render, rr_log_gt
)

rr.init("diff_rendering6")
rr.connect("127.0.0.1:8812")

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
rr_log_gt("gt", particle_centers, particle_widths, particle_colors)

rendered_soft = render(particle_centers, particle_widths, particle_colors)
rr.log("gt/gt_rendered_soft", rr.Image(rendered_soft))

particle_centers_shifted = jnp.array(
    [
        [0.05, 0.0, 1.0],
        [0.15, 0.2, 1.8],
    ]
)
rendered_shifted = render(particle_centers_shifted, particle_widths, particle_colors)
rr.log("shifted", rr.Image(rendered_shifted))

def compute_error(centers):
    rendered = render(centers, particle_widths, particle_colors)
    return jnp.sum((rendered - rendered_soft) ** 2)

print("ERROR:")
print(compute_error(particle_centers_shifted))
print("GRAD:")
g = jax.grad(compute_error)(particle_centers_shifted)
print(g)