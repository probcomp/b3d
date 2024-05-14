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
import demos.differentiable_renderer.tracking.utils as utils
import demos.differentiable_renderer.tracking.model as m
import b3d.likelihoods as likelihoods
import b3d.differentiable_renderer as r

rr.init("track_test-2")
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
cheezit_object_library = b3d.MeshLibrary.make_empty_library()
cheezit_object_library.add_trimesh(mesh)
rots = utils.vec_transform_axis_angle(jnp.array([0,0,1]), jnp.linspace(jnp.pi/4, 3*jnp.pi/4, 30))
in_place_rots = b3d.Pose.from_matrix(rots)
cam_pose = b3d.Pose.from_position_and_target(
    jnp.array([0.15, 0.15, 0.0]),
    jnp.array([0.0, 0.0, 0.0])
)
X_WC = cam_pose
compound_pose = X_WC.inv() @ in_place_rots
rgbs, depths = renderer.render_attribute_many(
    compound_pose[:,None,...],
    cheezit_object_library.vertices,
    cheezit_object_library.faces,
    jnp.array([[0, len(cheezit_object_library.faces)]]),
    cheezit_object_library.attributes
)
xyzs_C = utils.unproject_depth_vec(depths, renderer)
xyzs_W = X_WC.apply(xyzs_C)

rr.log("/gt_pointcloud", rr.Points3D(
    positions=xyzs_W[0].reshape(-1,3),
    colors=rgbs[0].reshape(-1,3),
    radii = 0.001*np.ones(xyzs_W[0].reshape(-1,3).shape[0])),
    timeless=True
)

####
center_x, center_y = 40, 45
del_pix = 5
patch_points_C = jax.lax.dynamic_slice(xyzs_C[0], (center_x-del_pix,center_y-del_pix,0), (2*del_pix,2*del_pix,3)).reshape(-1,3)
patch_rgbs = jax.lax.dynamic_slice(rgbs[0], (center_x-del_pix,center_y-del_pix,0), (2*del_pix,2*del_pix,3)).reshape(-1,3)
patch_vertices_C, patch_faces, patch_vertex_colors, patch_face_colors = b3d.make_mesh_from_point_cloud_and_resolution(
    patch_points_C, patch_rgbs, patch_points_C[:,2] / fx * 2.0
)

# W = world frame; P = patch frame; C = camera frame
X_CP = Pose.from_translation(patch_vertices_C.mean(0))
patch_vertices_P = X_CP.inv().apply(patch_vertices_C)
patch_vertices_W = X_WC.apply(patch_vertices_C)

###
hyperparams = r.DifferentiableRendererHyperparams(
    3, 1e-5, 1e-2, -1
)
# rendered = r.render_to_average_rgbd(
#     renderer,
#     patch_vertices_C,
#     patch_faces,
#     patch_vertex_colors,
#     background_attribute=jnp.array([0.1, 0.1, 0.1, 10.]),
#     hyperparams=hyperparams
# )
# rr.log("/img/rendered/avg_rgb", rr.Image(rendered[:, :, :3]), timeless=True)

rgbs2, depths2 = renderer.render_attribute(
    Pose.identity()[None, ...],
    patch_vertices_C,
    patch_faces,
    jnp.array([[0, len(patch_faces)]]),
    patch_vertex_colors
)
# rr.log("/trace/opengl_patch", rr.Image(rgbs2), timeless=True)

model = m.model_singleobject_gl_factory(renderer, hyperparams=hyperparams, depth_noise_scale=1e-5, lab_noise_scale = 0.1)
X_WP = X_WC @ X_CP

observed_rgbd = jnp.concatenate([rgbs[0], depths[0, :, :, None]], axis=-1)
vcm = genjax.vector_choice_map(genjax.vector_choice_map(genjax.choice(observed_rgbd)))
constraints = genjax.choice_map({
                             "pose": X_WP,
                             "camera_pose": X_WC,
                             "observed_rgbd": vcm
                         })
# trace, weight = model.importance(jax.random.PRNGKey(0), constraints, (patch_vertices_P, patch_faces, patch_vertex_colors))
# m.rr_viz_trace(trace, renderer)

###
def pose_to_trace_and_weight(pose):
    constraints = genjax.choice_map({
                             "pose": pose,
                             "camera_pose": X_WC,
                             "observed_rgbd": vcm
                         })
    trace, weight = model.importance(jax.random.PRNGKey(0), constraints, (patch_vertices_P, patch_faces, patch_vertex_colors))
    return trace, weight

def position_to_trace_and_weight(position):
    pose = b3d.Pose(position, X_WP._quaternion)
    return pose_to_trace_and_weight(pose)

def position_to_weight(position):
    return position_to_trace_and_weight(position)[1]
def position_to_trace(position):
    return position_to_trace_and_weight(position)[0]

position = X_WP.pos
step = 1e-7
for i in range(20):
    print("i = ", i)
    if i > 0:
        position = position + step * jax.grad(position_to_weight)(position)
    trace, weight = position_to_trace_and_weight(position)
    rr.set_time_sequence("gradient_ascent", i)
    rr.log("logpdf", rr.Scalar(weight))
    m.rr_viz_trace(trace, renderer)
    
    rr.log("/patchimg/observed_img", rr.Image(rgbs[0]))
    rr.log("/patchimg/opengl_patch", rr.Image(rgbs2))
    rr.log("/patchimg/rendered_patch", rr.Image(trace.get_retval()[1][:, :, :3]))