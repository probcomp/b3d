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

rr.init("diff_rendering")
rr.connect("127.0.0.1:8812")

image_width=100
image_height=100
fx=50.0
fy=50.0
cx=50.0
cy=50.0
near=0.001
far=16.0
renderer = b3d.Renderer(
    image_width, image_height, fx, fy, cx, cy, near, far
)

def center_and_width_to_vertices_faces_colors(
    i, center, width, color
):
    vertices = jnp.array([
        [-0.5, -0.5, 0.0],
        [0.5, -0.5, 0.0],
        [0.5, 0.5, 0.0],
        [-0.5, 0.5, 0.0],
    ]) * width + center
    faces = jnp.array([
        [0, 1, 2],
        [0, 2, 3],
    ]) + 4*i
    colors = jnp.ones((4,3)) * color
    return vertices, faces, colors, jnp.ones(len(faces), dtype=jnp.int32) * i

particle_centers = jnp.array([
    [0.0, 0.0, 1.0],
    [0.2, 0.2, 1.0],
])
particle_widths = jnp.array([0.1, 0.3])
particle_colors = jnp.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])

vertices, faces, colors, triangle_to_particle_index = jax.vmap(center_and_width_to_vertices_faces_colors)(jnp.arange(len(centers)), particle_centers, particle_widths, particle_colors)
vertices = vertices.reshape(-1, 3)
faces = faces.reshape(-1, 3)
colors = colors.reshape(-1, 3)
triangle_to_particle_index = triangle_to_particle_index.reshape(-1)
_, _, triangle_id_image, depth_image = renderer.rasterize(Pose.identity()[None,...], vertices, faces, jnp.array([[0, len(faces)]]))

point_of_intersection = b3d.xyz_from_depth(depth, fx,fy,cx,cy)
particle_intersected = triangle_to_particle_index[triangle_id_image - 1]


# Load date
path = os.path.join(b3d.get_root_path(),
"assets/shared_data_bucket/input_data/shout_on_desk.r3d.video_input.npz")
video_input = b3d.VideoInput.load(path)


# Get intrinsics
image_width, image_height, fx,fy, cx,cy,near,far = np.array(video_input.camera_intrinsics_depth)
image_width, image_height = int(image_width), int(image_height)
fx,fy, cx,cy,near,far = float(fx),float(fy), float(cx),float(cy),float(near),float(far)

# Get RGBS and Depth
rgbs = video_input.rgb[::3] / 255.0
xyzs = video_input.xyz[::3]

# Resize rgbs to be same size as depth.
rgbs_resized = jnp.clip(jax.vmap(jax.image.resize, in_axes=(0, None, None))(
    rgbs, (video_input.xyz.shape[1], video_input.xyz.shape[2], 3), "linear"
), 0.0, 1.0)

renderer = b3d.Renderer(
    image_width, image_height, fx, fy, cx, cy, near, far
)

particle_centers = xyzs[0].reshape(-1,3)
particle_widths = xyzs[...,2].reshape(-1) / fx
vertices, faces, colors, triangle_to_particle_index = jax.vmap(center_and_width_to_vertices_faces_colors)(jnp.arange(len(centers)), particle_centers, particle_widths, particle_colors)
vertices = vertices.reshape(-1, 3)
faces = faces.reshape(-1, 3)
colors = colors.reshape(-1, 3)
triangle_to_particle_index = triangle_to_particle_index.reshape(-1)
_, _, triangle_id_image, depth_image = renderer.rasterize(Pose.identity()[None,...], vertices, faces, jnp.array([[0, len(faces)]]))




def render_particles(particle_centers, particle_widths, particle_colors):
    vertices, faces, colors = jax.vmap(center_and_width_to_vertices_faces_colors)(jnp.arange(len(centers)), particle_centers, particle_widths, particle_colors)
    vertices = vertices.reshape(-1, 3)
    faces = faces.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    image, depth = renderer.render_attribute(Pose.identity()[None,...], vertices, faces, jnp.array([[0, len(faces)]]), colors)
    return image
render_particles_jit = jax.jit(render_particles)




from tqdm import tqdm
for _ in tqdm(range(1000)):
    image = render_particles_jit(particle_centers, particle_widths, particle_colors)

rr.log("/rgb", rr.Image(image), timeless=True)




