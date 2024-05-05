"""
Test: GD w.r.t. stochastic likelihood.
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
import genjax


from demos.differentiable_renderer.utils import (
    center_and_width_to_vertices_faces_colors, rr_log_gt, ray_from_ij,
    fx, fy, cx, cy
)
from demos.differentiable_renderer.rendering import all_pairs, render_to_dist_parameters, render, renderer, project_pixel_to_plane

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

rr.init("softras_gd_1")
rr.connect("127.0.0.1:8812")

rr_log_gt("gt", particle_centers, particle_widths, particle_colors)

SIGMA = 1e-4
GAMMA = 0.4
EPSILON = 1e-5
hyperparams = (SIGMA, GAMMA, EPSILON)
(weights, colors) = render_to_dist_parameters(vertices, faces, triangle_colors, hyperparams)

from demos.differentiable_renderer.likelihoods import mixture_rgb_sensor_model

# key = jax.random.PRNGKey(0)
# trace = mixture_rgb_sensor_model.simulate(key, (weights, colors, 3.))

rr.log("c/gt", rr.Image(color_image), timeless=True)
# rr.log("c/rendered", rr.Image(trace.get_retval().reshape(100, 100, 3)), timeless=True)

def get_render(key, weights, colors):
    return mixture_rgb_sensor_model.simulate(key, (weights, colors, 3.)).get_retval().reshape(100, 100, 3)

def get_img_logpdf(key, img, weights, colors):
    choicemap = genjax.vector_choice_map(genjax.choice(img.reshape(-1, 3)))
    tr, w = mixture_rgb_sensor_model.importance(key, choicemap, (weights, colors, 3.))
    return w

keys = jax.random.split(jax.random.PRNGKey(0), 100)
renders = jax.vmap(get_render, in_axes=(0, None, None))(
    keys, weights, colors
)
for t in range(100):
    rr.set_time_sequence("stochastic_render", t)
    rr.log("c/rendered", rr.Image(renders[t, ...]))

mean_render = jnp.mean(renders, axis=0)
rr.log("c/mean_render", rr.Image(mean_render), timeless=True)

# g = jnp.where(
#     weights == 0.,
#     0.,
#     jax.grad(get_img_logpdf, argnums=(2,))(jax.random.PRNGKey(0), color_image, weights, colors)[0]
# )

jax.grad(get_img_logpdf, argnums=(2,))(jax.random.PRNGKey(0), color_image, weights, colors)[0]

###
def render_to_dist_from_centers(new_particle_centers):
    particle_center_delta  = new_particle_centers - particle_centers
    new_vertices = vertices_og + jnp.expand_dims(particle_center_delta, 1)
    weights, colors = render_to_dist_parameters(new_vertices.reshape(-1, 3), faces.reshape(-1, 3), particle_colors[triangle_to_particle_index], hyperparams)
    return weights, colors

def render_from_centers(new_particle_centers):
    particle_center_delta  = new_particle_centers - particle_centers
    new_vertices = vertices_og + jnp.expand_dims(particle_center_delta, 1)
    return render(new_vertices.reshape(-1, 3), faces.reshape(-1, 3), particle_colors[triangle_to_particle_index], hyperparams)

def compute_logpdf(centers):
    weights, colors = render_to_dist_from_centers(centers)
    return get_img_logpdf(jax.random.PRNGKey(0), color_image, weights, colors)    

particle_centers_shifted = jnp.array(
    [
        [0.05, 0.0, 1.0],
        [0.15, 0.2, 2.0],
        [0., 0., 5.]
    ]
)


rr.log("gt_mesh", rr.Mesh3D(
    vertex_positions=vertices,
    indices=faces,
    vertex_colors=colors),
    timeless=True
)

# assumes some uniform prior on the particle centers
@jax.jit
def ULA_step(key, current_centers, sigma):
    g = jax.grad(compute_logpdf)(current_centers)
    stepped = current_centers + sigma**2 * g
    noised = genjax.normal.sample(key, stepped, jnp.sqrt(2) * sigma)
    return noised

@jax.jit
def MALA_step(key, current_centers, sigma):
    g = jax.grad(compute_logpdf)(current_centers)
    stepped = current_centers + sigma**2 * g
    noised = genjax.normal.sample(key, stepped, jnp.sqrt(2) * sigma)
    
    # compute acceptance probability
    logpdf_current = compute_logpdf(current_centers)
    logpdf_stepped = compute_logpdf(noised)
    p_ratio = logpdf_stepped - logpdf_current

    q_fwd = genjax.normal.logpdf(noised, stepped, jnp.sqrt(2) * sigma)
    g_bwd = jax.grad(compute_logpdf)(noised)
    bwd_stepped = noised + sigma**2 * g_bwd
    q_bwd = genjax.normal.logpdf(current_centers, bwd_stepped, jnp.sqrt(2) * sigma)
    q_ratio = q_bwd - q_fwd
    alpha = jnp.minimum(jnp.exp(p_ratio + q_ratio), 1.0)
    accept = genjax.bernoulli.sample(key, alpha)
    return (accept, jnp.where(accept, noised, current_centers))

current_centers = particle_centers_shifted
eps = 1e-5
key = jax.random.PRNGKey(10)
n_acc = 0
N_steps = 300
for i in range(N_steps):
    print(f"i = {i}")
    # current_centers = current_centers + eps * g
    key, subkey = jax.random.split(key)
    # (acc, current_centers) = MALA_step(subkey, current_centers, 1e-3)
    current_centers = ULA_step(subkey, current_centers, 1e-3)
    # n_acc += acc
    rr.set_time_sequence("gd", i)
    rendered = render_from_centers(current_centers)
    rr.log("gd", rr.Image(rendered))
    v, f, c, t2pidx = jax.vmap(
        center_and_width_to_vertices_faces_colors
    )(jnp.arange(len(current_centers)), current_centers, particle_widths, particle_colors)
    v = v.reshape(-1, 3)
    f = f.reshape(-1, 3)
    c = c.reshape(-1, 3)
    rr.log("gd_mesh", rr.Mesh3D(
        vertex_positions=v,
        indices=f)
    )
print(f"Acceptance fraction: {n_acc / N_steps}")


def step(carry, x):
    key, centers = carry
    key, subkey = jax.random.split(key)
    c2 = ULA_step(subkey, centers, 1e-3)
    return ((key, c2), x)

@jax.jit
def do_inference():
    return jax.lax.scan(step, (jax.random.PRNGKey(0), particle_centers_shifted), xs=None, length=300)

do_inference()

import time
start = time.time()
do_inference()
end = time.time()
print(f"Time: {end - start}")

##### Grid of likelihood scores ###

def value_to_centers(x, y):
    particle_centers = jnp.array(
        [
            [0.0, 0.0, 1.0],
            [x, y, 2.0],
            [0., 0., 5.]
        ]
    )
    return particle_centers

@jax.jit
def value_to_logpdf(xy):
    x, y = xy
    particle_centers = value_to_centers(x, y)
    return compute_logpdf(particle_centers)

x = jnp.linspace(-0.5, 0.5, 200)
y = jnp.linspace(-0.5, 0.5, 200)
xy = jnp.stack(jnp.meshgrid(x, y), axis=-1).reshape(-1, 2)
logpdfs = jnp.array([value_to_logpdf(_xy) for _xy in xy])
logpdfs = logpdfs.reshape(200, 200)
rr.log("logpdfs/logpdfs", rr.DepthImage(logpdfs), timeless=True)
# find index with [0.2, 0.2] in xy
lt = jnp.linalg.norm(xy - jnp.array([0.2, 0.2]), axis=-1) < 0.01
idx = jnp.where(lt)[0]
ij = jnp.unravel_index(idx, (200, 200))
ij2 = jnp.array([jnp.mean(ij[0]), jnp.mean(ij[0])])
rr.log("logpdfs/true_position", rr.Points2D(jnp.array([ij2]), radii=1.0, colors=jnp.array([0, 0, 0])), timeless=True)