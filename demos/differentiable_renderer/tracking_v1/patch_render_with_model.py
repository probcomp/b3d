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

rr.init("test10")
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
for t in range(rgbs.shape[0]):
    rr.set_time_sequence("frame", t)
    rr.log("/img/gt/rgb", rr.Image(rgbs[t, ...]))
    rr.log("/img/gt/depth", rr.Image(depths[t, ...]))
    rr.log("/gt_pointcloud", rr.Points3D(
        positions=xyzs_W[t].reshape(-1,3),
        colors=rgbs[t].reshape(-1,3),
        radii = 0.001*np.ones(xyzs_W[t].reshape(-1,3).shape[0]))
    )
rr.log("camera",
       rr.Pinhole(
           focal_length=renderer.fx,
           width=renderer.width,
           height=renderer.height,
           principal_point=jnp.array([renderer.cx, renderer.cy]),
        ), timeless=True
    )
rr.log("camera", rr.Transform3D(translation=cam_pose.pos, mat3x3=cam_pose.rot.as_matrix()), timeless=True)

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

rr.set_time_sequence("frame", 0)
rr.log("patch", rr.Mesh3D(
    vertex_positions=patch_vertices_W,
    indices=patch_faces,
    vertex_colors=patch_vertex_colors
))

###
hyperparams = r.DifferentiableRendererHyperparams(
    3, 1e-5, 1e-2, -1
)
rendered = r.render_to_average_rgbd(
    renderer,
    patch_vertices_C,
    patch_faces,
    patch_vertex_colors,
    background_attribute=jnp.array([0.1, 0.1, 0.1, 10.]),
    hyperparams=hyperparams
)
rr.log("/img/rendered/avg_rgb", rr.Image(rendered[:, :, :3]), timeless=True)

rgbs2, depths2 = renderer.render_attribute(
    Pose.identity()[None, ...],
    patch_vertices_C,
    patch_faces,
    jnp.array([[0, len(patch_faces)]]),
    patch_vertex_colors
)
rr.log("/trace/opengl_patch", rr.Image(rgbs2), timeless=True)

model = m.model_singleobject_gl_factory(renderer, hyperparams=hyperparams)
X_WP = X_WC @ X_CP
def get_render(key):
    trace, _ = model.importance(key,
                                genjax.choice_map({
                                    "pose": X_WP,
                                    "camera_pose": X_WC
                                }),
                                (patch_vertices_P, patch_faces, patch_vertex_colors))
    return trace.get_retval()[0]

# Generate + visualize 100 stochastic renders
keys = jax.random.split(jax.random.PRNGKey(0), 100)
renders = jax.vmap(get_render, in_axes=(0,))(keys)
for t in range(100):
    rr.set_time_sequence("stochastic_render", t)
    rr.log("/img/rendered/stochastic_rgb", rr.Image(renders[t, :, :, :3]))

###
observed_rgbd = jnp.concatenate([rgbs[0], depths[0, :, :, None]], axis=-1)
vcm = genjax.vector_choice_map(genjax.vector_choice_map(genjax.choice(observed_rgbd)))
constraints = genjax.choice_map({
                             "pose": X_WP,
                             "camera_pose": X_WC,
                             "observed_rgbd": vcm
                         })
trace, weight = model.importance(jax.random.PRNGKey(0), constraints, (patch_vertices_P, patch_faces, patch_vertex_colors))
m.rr_viz_trace(trace, renderer)

###
model = m.model_singleobject_gl_factory(renderer, hyperparams=r.DEFAULT_HYPERPARAMS)
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
        position = position + step * jax.grad(position_to_weight)(X_WP.pos)
    trace, weight = position_to_trace_and_weight(position)
    rr.set_time_sequence("gradient_ascent", i)
    rr.log("logpdf", rr.Scalar(weight))
    m.rr_viz_trace(trace, renderer)