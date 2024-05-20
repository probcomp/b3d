import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np
import os
import trimesh
import b3d
from jax.scipy.spatial.transform import Rotation as Rot
from b3d import Pose
#from b3d.utils import unproject_depth
import rerun as rr
import genjax
from tqdm import tqdm
import demos.differentiable_renderer.tracking_v1.utils as utils
import demos.differentiable_renderer.tracking_v1.model as m
import b3d.likelihoods as likelihoods
import b3d.differentiable_renderer as r

rr.init("test6")
rr.connect("127.0.0.1:8812")

width = 100
height = 128
fx=64.0
fy=64.0
cx=64.0
cy=64.0
near=0.001
far=16.0

renderer = b3d.Renderer(
    width, height, fx, fy, cx, cy, near, far
)

mesh_path = os.path.join(b3d.get_root_path(),
    "assets/shared_data_bucket/ycb_video_models/models/003_cracker_box/textured_simple.obj")
mesh = trimesh.load(mesh_path)
object_library = b3d.MeshLibrary.make_empty_library()
object_library.add_trimesh(mesh)
rots = utils.vec_transform_axis_angle(jnp.array([0,0,1]), jnp.linspace(jnp.pi/4, 3*jnp.pi/4, 30))
in_place_rots = b3d.Pose.from_matrix(rots)
cam_pose = b3d.Pose.from_position_and_target(
    jnp.array([0.15, 0.15, 0.0]),
    jnp.array([0.0, 0.0, 0.0])
)
compound_pose = cam_pose.inv() @ in_place_rots
rgbs, depths = renderer.render_attribute_many(
    compound_pose[:,None,...],
    object_library.vertices,
    object_library.faces,
    jnp.array([[0, len(object_library.faces)]]),
    object_library.attributes
)

vertices = object_library.vertices
faces = object_library.faces
vertex_colors = object_library.attributes

def get_render(key, weights, colors):
    lab_color_space_noise_scale = 3.0
    depth_noise_scale = 0.07
    return likelihoods.mixture_rgbd_sensor_model.simulate(
        key,
        (weights, colors, lab_color_space_noise_scale, depth_noise_scale, 0., 10.)
    ).get_retval().reshape(height, width, 4)

hyperparams = r.DifferentiableRendererHyperparams(
    3, 1e-5, 1e-2, -2
)
(weights, colors) = r.render_to_rgbd_dist_params(
    renderer, compound_pose[0].apply(vertices), faces, vertex_colors, hyperparams
)

# Generate + visualize 100 stochastic renders
keys = jax.random.split(jax.random.PRNGKey(0), 100)
renders = jax.vmap(get_render, in_axes=(0, None, None))(
    keys, weights, colors
)
rr.log("/img/opengl", rr.Image(rgbs[0, :, :, :3]), timeless=True)
for t in range(100):
    rr.set_time_sequence("stochastic_render", t)
    rr.log("/img/rendered", rr.Image(renders[t, :, :, :3]))
