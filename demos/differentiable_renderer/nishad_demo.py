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

# Load date
path = os.path.join(
    b3d.get_root_path(),
    "assets/shared_data_bucket/input_data/shout_on_desk.r3d.video_input.npz",
)
video_input = b3d.VideoInput.load(path)

# Get intrinsics
image_width, image_height, fx, fy, cx, cy, near, far = np.array(
    video_input.camera_intrinsics_depth
)
image_width, image_height = int(image_width), int(image_height)
fx, fy, cx, cy, near, far = (
    float(fx),
    float(fy),
    float(cx),
    float(cy),
    float(near),
    float(far),
)


# Get RGBS and Depth
rgbs = video_input.rgb[::4] / 255.0
xyzs = video_input.xyz[::4]

# Resize rgbs to be same size as depth.
rgbs_resized = jnp.clip(
    jax.vmap(jax.image.resize, in_axes=(0, None, None))(
        rgbs, (video_input.xyz.shape[1], video_input.xyz.shape[2], 3), "linear"
    ),
    0.0,
    1.0,
)

num_layers = 2048
renderer = b3d.Renderer(image_width, image_height, fx, fy, cx, cy, near, far)

WINDOW = 5

###########
@functools.partial(
    jnp.vectorize,
    signature="(m)->(k),()",
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
    particle_depth_in_window = particle_centers_in_window[..., 2]

    particle_centers_on_image_plane = particle_centers_in_window[..., :2] / particle_centers_in_window[..., 2:]
    particle_pixel_coordinate = particle_centers_on_image_plane * jnp.array([fx, fy]) + jnp.array([cx, cy])

    distance_on_image_plane = jnp.linalg.norm(particle_pixel_coordinate - jnp.array([ij[1], ij[0]]) + 1e-5, axis=-1)

    particle_width_on_image_plane = particle_widths_in_window / (particle_centers_in_window[..., 2] + 1e-10) + 1e-10
    # return jnp.array([jnp.sum(particle_width_on_image_plane, axis=[0,1])]) # checks out

    scaled_distance_on_image_plane = distance_on_image_plane / particle_width_on_image_plane / 20.0
    scaled_distance_on_image_plane_masked = scaled_distance_on_image_plane * (triangle_intersected_padded_in_window > 0)
    # return jnp.array([jnp.sum(scaled_distance_on_image_plane_masked, axis=[0,1])])
    # rr.log("influence", rr.DepthImage(scaled_distance_on_image_plane))
    # rr.log("influence", rr.DepthImage(scaled_distance_on_image_plane_masked))
    influence = jnp.exp(- (scaled_distance_on_image_plane_masked ** 2)) * (triangle_intersected_padded_in_window > 0)
    # rr.log("influence", rr.DepthImage(influence))

    particle_colors_in_window = particle_colors[particle_indices_in_window]
    particle_colors_in_window_masked = particle_colors_in_window 
    # rr.log("particle_colors_in_window_masked", rr.Image(particle_colors_in_window_masked))
    
    color_image = jnp.sum(particle_colors_in_window_masked * influence[...,None], axis=[0,1]) / (influence.sum() + 1e-10)
    depth_image = jnp.sum(particle_depth_in_window * influence, axis=[0,1]) / (influence.sum() + 1e-10)

    return color_image, depth_image


def render(particle_centers, particle_widths, particle_colors):
    particle_colors = jax.nn.sigmoid(particle_colors)
    particle_widths = jnp.abs(particle_widths)
    # Get triangle "mesh" for the scene:
    vertices, faces, vertex_colors, triangle_index_to_particle_index = jax.vmap(
        b3d.particle_center_width_color_to_vertices_faces_colors
    )(jnp.arange(len(particle_centers)), particle_centers, particle_widths / 2, particle_colors)
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
    rgb_image, depth_image = get_pixel_color(ijs, triangle_intersected_padded, particle_centers, particle_widths, particle_colors, triangle_index_to_particle_index, vertices, faces, vertex_colors)

    return jnp.clip(rgb_image, 0.0, 1.0), depth_image, triangle_id_image

render_jit = jax.jit(render)
# Set up 3 squares oriented toward the camera, with different colors

# particle_centers = jnp.array([[0.0, 0.0, 0.3]])
# particle_widths = jnp.array([0.02])
# particle_colors = jnp.array([[1.0, 0.0, 0.0]])
N = 1000
particle_centers = jax.random.uniform(
    jax.random.PRNGKey(10), (N, 3),
    minval=jnp.array([-2.0, -2.0, 2.0]),
    maxval=jnp.array([2.0, 2.0, 2.5]),
)
particle_widths = jax.random.uniform(jax.random.PRNGKey(0), (N,), minval=0.001, maxval=0.1)
particle_colors = jax.random.uniform(jax.random.PRNGKey(0), (N,3), minval=-1.0, maxval=1.0)


ij = jnp.array([157, 85])
# 3 unit tests on 3 arguments
rgb_image, depth_image, triangle_id_image = render_jit(particle_centers, particle_widths, particle_colors)
rr.set_time_sequence("frame", 0)

gt_image = rgbs_resized[0]
gt_depth = xyzs[0][..., 2]

rr.log('image', rr.Image(gt_image),timeless=True)
rr.log('depth', rr.DepthImage(gt_depth),timeless=True)
rr.log('image/reconstruction', rr.Image(rgb_image[...,:3]))
rr.log('depth/reconstruction', rr.DepthImage(depth_image))
rr.log('image/triangles', rr.DepthImage(triangle_id_image))
image.sum()

def loss_func(a,b,c,gt_rgb, gt_depth):
    rgb,depth,_ = render(a,b,c)
    return jnp.mean(jnp.abs(depth - gt_depth))
loss_func_grad = jax.jit(jax.value_and_grad(loss_func, argnums=(0,1,2)))
loss_func_grad(particle_centers, particle_widths, particle_colors, gt_image, gt_depth)

for t in range(100):
    rr.set_time_sequence("frame", t)
    loss, (grad_centers,grad_widths, grad_colors) = loss_func_grad(particle_centers, particle_widths, particle_colors, gt_image, gt_depth)
    particle_centers -= 1 * grad_centers
    # particle_widths -= 1 * grad_widths
    particle_colors -= 1000 * grad_colors
    rgb_image, depth_image = render_jit(particle_centers, particle_widths, particle_colors)[:2]
    rr.log('image/reconstruction', rr.Image(rgb_image[...,:3]))
    rr.log('depth/reconstruction', rr.DepthImage(depth_image))



