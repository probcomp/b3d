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
from tqdm import tqdm
import jax
import jax.numpy as jnp
import optax
import b3d.differentiable_renderer as rendering
import b3d.likelihoods as likelihoods
import demos.differentiable_renderer.utils as utils

rr.init("gradients")
rr.connect("127.0.0.1:8812")

# Load date
path = os.path.join(
    b3d.get_root_path(),
    "assets/shared_data_bucket/input_data/shout_on_desk.r3d.video_input.npz",
)
video_input = b3d.VideoInput.load(path)

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

hyperparams = rendering.DEFAULT_HYPERPARAMS
def render(particle_centers, particle_widths, particle_colors):
    particle_colors *= 10
    particle_colors = jnp.clip(particle_colors, 0.0, 1.0)
    particle_widths = jnp.abs(particle_widths)
    # Get triangle "mesh" for the scene:
    vertices, faces, vertex_colors, triangle_index_to_particle_index = jax.vmap(
        b3d.square_center_width_color_to_vertices_faces_colors
    )(jnp.arange(len(particle_centers)), particle_centers, particle_widths / 2, particle_colors)
    vertices = vertices.reshape(-1, 3)
    faces = faces.reshape(-1, 3)
    vertex_rgbs = vertex_colors.reshape(-1, 3)
    triangle_index_to_particle_index = triangle_index_to_particle_index.reshape(-1)
    image = rendering.render_to_average_rgbd(
        renderer,
        vertices,
        faces,
        vertex_rgbs,
        background_attribute=jnp.array([0.0, 0.0, 0.0, 0])
    )
    return jnp.clip(image, 0.0, 1.0)

def render2(particle_centers, particle_widths, particle_colors):
    particle_colors *= 10
    particle_colors = jnp.clip(particle_colors, 0.0, 1.0)
    particle_widths = jnp.abs(particle_widths)
    # Get triangle "mesh" for the scene:
    vertices, faces, vertex_colors, triangle_index_to_particle_index = jax.vmap(
        b3d.square_center_width_color_to_vertices_faces_colors
    )(jnp.arange(len(particle_centers)), particle_centers, particle_widths / 2, particle_colors)
    vertices = vertices.reshape(-1, 3)
    faces = faces.reshape(-1, 3)
    vertex_rgbs = vertex_colors.reshape(-1, 3)
    triangle_index_to_particle_index = triangle_index_to_particle_index.reshape(-1)
    image = rendering.render_to_average_rgbd(
        renderer,
        vertices,
        faces,
        vertex_rgbs,
        background_attribute=jnp.array([0.0, 0.0, 0.0, 0])
    )
    return image

render_jit = jax.jit(render)
# Set up 3 squares oriented toward the camera, with different colors

particle_centers = jnp.array([[0.0, 0.0, 0.4]])
particle_colors = jnp.array([[.1, 0.0, 0.0]])
particle_widths = jnp.array([[0.05]])
gt_image = render_jit(particle_centers, particle_widths, particle_colors)
rr.set_time_sequence("frame", 0)
rr.log("img", rr.Image(gt_image[...,:3]))

def loss_func(a,b,c, gt):
    image = render(a,b,c)
    return jnp.mean(jnp.abs(image - gt))

loss_func_grad = jax.jit(jax.value_and_grad(loss_func, argnums=(0,1,2)))
loss_func_grad(particle_centers, particle_widths, particle_colors, gt_image)

def loss_function_2(c, gt):
    image = render(particle_centers, particle_widths, c)
    return jnp.mean(jnp.abs(image - gt))

loss_func_grad2 = jax.jit(jax.value_and_grad(loss_function_2, argnums=(0,)))
loss_func_grad2(particle_colors, gt_image)

particle_centers = jnp.array([[0.0, 0.0, 0.4]])
particle_colors = jnp.array([[0.0, .1, 0.0]])
particle_widths = jnp.array([[0.05]])
image = render_jit(particle_centers, particle_widths, particle_colors)

rr.log('image', rr.Image(gt_image[...,:3]),timeless=True)
rr.log('image/reconstruction', rr.Image(image[...,:3]))

pbar = tqdm(range(100))

from jax.example_libraries import optimizers
opt_init, opt_update, get_params = optimizers.adam(1e-2)
# opt_state = opt_init(particle_colors)
opt_state = opt_init((particle_centers, particle_widths, particle_colors))

for t in pbar:
    rr.set_time_sequence("frame2", t)
    particle_centers, particle_widths, particle_colors = get_params(opt_state)
    # particle_colors = get_params(opt_state)
    # loss, (grads, ) = loss_func_grad2(particle_colors, gt_image)
    loss, grads = loss_func_grad(particle_centers, particle_widths, particle_colors, gt_image)
    opt_state = opt_update(t, grads, opt_state)
    pbar.set_description(f"Loss: {loss}")
    particle_centers, particle_widths, particle_colors = get_params(opt_state)
    # particle_colors = get_params(opt_state)
    image = render_jit(particle_centers, particle_widths, particle_colors)
    rr.log('image/reconstruction', rr.Image(image[...,:3]))