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

import b3d._differentiable_renderer_old as rendering
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
rr.init("differentiable_rendering--scene_and_renders")
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

def render_from_centers(new_particle_centers):
    particle_center_delta  = new_particle_centers - particle_centers
    new_vertices = vertices.reshape(3, 4, 3) + jnp.expand_dims(particle_center_delta, 1)
    return rendering.render(new_vertices.reshape(-1, 3), faces.reshape(-1, 3), particle_colors[triangle_to_particle_index], hyperparams)

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
rr.init("differentiable_rendering--scene_fitting4")
rr.connect("127.0.0.1:8812")
rr.log("/scene/ground_truth", rr.Mesh3D(vertex_positions=vertices, indices=faces, vertex_colors=vertex_colors), timeless=True)
rr.log("/scene/camera", rr.Pinhole(focal_length=rendering.fx, width=rendering.image_width, height=rendering.image_height), timeless=True)
rr.log("/img/opengl_rendering", rr.Image(color_image), timeless=True)

# ULA & MALA inference moves
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
grad_jit = jax.jit(jax.grad(compute_logpdf))

# Set up variant of scene misaligned with the real image
particle_centers_shifted = jnp.array(
    [
        [-0.07, -0.08, 1.02],
        [0., 0.23, 1.98],
        [0., 0., 5.]
    ]
)

current_centers_gd = particle_centers_shifted
current_centers_ula = particle_centers_shifted
current_centers_mala = particle_centers_shifted

sigma = .004
key = jax.random.PRNGKey(10)
n_acc_mala = 0
N_steps = 100 # 10 steps should be enough to fit it pretty well --
              # but I'm showing 100 to show how it progresses after this initial fit
# Note that this is slow due to all the logging, but last time I checked
# we can run ULA at about 170 fps.  (Possibly closer to 450fps on this scene
# with some optimizations I think I know how to implement, judging by a quick experiment I ran.)
for i in range(100):
    print(f"i = {i}")
    key, subkey = jax.random.split(key)
    if i != 0:
        current_centers_gd = current_centers_gd + sigma**2 * grad_jit(current_centers_gd)
        current_centers_ula = ULA_step(subkey, current_centers_ula, sigma)
        (acc, current_centers_mala) = MALA_step(subkey, current_centers_mala, sigma)
        n_acc_mala += acc

    rr.set_time_sequence("scene_fitting_step", i)
    rendered_gd = render_from_centers(current_centers_gd)
    rendered_ula = render_from_centers(current_centers_ula)
    rendered_mala = render_from_centers(current_centers_mala)
    rr.log("/rerendered/gd", rr.Image(rendered_gd))
    rr.log("/rerendered/ula", rr.Image(rendered_ula))
    rr.log("/rerendered/mala", rr.Image(rendered_mala))

    rr.log("/l1_error_imgs/gd", rr.DepthImage(jnp.abs(rendered_gd - color_image).sum(axis=-1)))
    rr.log("/l1_error_imgs/ula", rr.DepthImage(jnp.abs(rendered_ula - color_image).sum(axis=-1)))
    rr.log("/l1_error_imgs/mala", rr.DepthImage(jnp.abs(rendered_mala - color_image).sum(axis=-1)))

    for string, centers in [("gd", current_centers_gd), ("ula", current_centers_ula), ("mala", current_centers_mala)]:
        v, f, c, t2pidx = jax.vmap(
            utils.center_and_width_to_vertices_faces_colors
        )(jnp.arange(len(centers)), centers, particle_widths, particle_colors)
        v, f, c = v.reshape(-1, 3), f.reshape(-1, 3), c.reshape(-1, 3)
        rr.log(
            f"scene/{string}",
            rr.Mesh3D(vertex_positions=v, indices=f)
        )

    rr.log("/logpdf/gd", rr.Scalar(compute_logpdf(current_centers_gd)))
    rr.log("/logpdf/ula", rr.Scalar(compute_logpdf(current_centers_ula)))
    rr.log("/logpdf/mala", rr.Scalar(compute_logpdf(current_centers_mala)))

    rr.log("/l1_errors/gd", rr.Scalar(jnp.abs(rendered_gd - color_image).sum()))
    rr.log("/l1_errors/ula", rr.Scalar(jnp.abs(rendered_ula - color_image).sum()))
    rr.log("/l1_errors/mala", rr.Scalar(jnp.abs(rendered_mala - color_image).sum()))

print(f"MALA acceptance fraction: {n_acc_mala / N_steps}")

### Timing experiment:

def step(carry, x):
    key, centers = carry
    key, subkey = jax.random.split(key)
    c2 = ULA_step(subkey, centers, 1e-3)
    return ((key, c2), x)
@jax.jit
def do_inference():
    return jax.lax.scan(step, (jax.random.PRNGKey(0), particle_centers_shifted), xs=None, length=100)

do_inference()

import time
start = time.time()
do_inference()
end = time.time()
print(f"Time to run 100 steps of ULA: {end - start}s")