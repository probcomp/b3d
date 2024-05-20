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

# W = world frame; P = patch frame; C = camera frame

rr.init("track_test-3")
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

### Rotating box data ###
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
observed_rgbds = jnp.concatenate([rgbs, depths[...,None]], axis=-1)
xyzs_C = utils.unproject_depth_vec(depths, renderer)
xyzs_W = X_WC.apply(xyzs_C)
for t in range(rgbs.shape[0]):
    rr.set_time_sequence("frame", t)
    rr.log("/img/gt/rgb", rr.Image(rgbs[t, ...]))
    rr.log("/img/gt/depth", rr.Image(depths[t, ...]))
    rr.log("/3D/gt_pointcloud", rr.Points3D(
        positions=xyzs_W[t].reshape(-1,3),
        colors=rgbs[t].reshape(-1,3),
        radii = 0.001*np.ones(xyzs_W[t].reshape(-1,3).shape[0]))
    )
rr.log("/3D/camera",
       rr.Pinhole(
           focal_length=renderer.fx,
           width=renderer.width,
           height=renderer.height,
           principal_point=jnp.array([renderer.cx, renderer.cy]),
        ), timeless=True
    )
rr.log("/3D/camera", rr.Transform3D(translation=cam_pose.pos, mat3x3=cam_pose.rot.as_matrix()), timeless=True)

### Get patch ###
center_x, center_y = 40, 45
del_pix = 5
patch_points_C = jax.lax.dynamic_slice(xyzs_C[0], (center_x-del_pix,center_y-del_pix,0), (2*del_pix,2*del_pix,3)).reshape(-1,3)
patch_rgbs = jax.lax.dynamic_slice(rgbs[0], (center_x-del_pix,center_y-del_pix,0), (2*del_pix,2*del_pix,3)).reshape(-1,3)
patch_vertices_C, patch_faces, patch_vertex_colors, patch_face_colors = b3d.make_mesh_from_point_cloud_and_resolution(
    patch_points_C, patch_rgbs, patch_points_C[:,2] / fx * 2.0
)
X_CP = Pose.from_translation(patch_vertices_C.mean(0))
patch_vertices_P = X_CP.inv().apply(patch_vertices_C)
patch_vertices_W = X_WC.apply(patch_vertices_C)
X_WP = X_WC @ X_CP

### Set up diff rend ###
hyperparams = r.DifferentiableRendererHyperparams(
    3, 1e-5, 1e-2, -1
)
model = m.model_singleobject_gl_factory(
    renderer,
    hyperparams=hyperparams,
    depth_noise_scale=1e-2,
    lab_noise_scale = 0.5,
    mindepth=-1.,
    maxdepth=1.0
)

#                          b3d.Pose    (H, W, 4) array
def pose_to_trace_and_weight(pose, observed_rgbd):
    vcm = genjax.vector_choice_map(genjax.vector_choice_map(genjax.choice(observed_rgbd)))
    constraints = genjax.choice_map({
                             "pose": pose,
                             "camera_pose": X_WC,
                             "observed_rgbd": vcm
                         })
    trace, weight = model.importance(jax.random.PRNGKey(0), constraints, (patch_vertices_P, patch_faces, patch_vertex_colors))
    return trace, weight

@jax.jit
def pos_quat_to_trace_and_weight(position, quaternion, observed_rgbd):
    pose = b3d.Pose(position, quaternion)
    return pose_to_trace_and_weight(pose, observed_rgbd)

def pos_quat_to_weight(position, quaternion, observed_rgbd):
    return pos_quat_to_trace_and_weight(position, quaternion, observed_rgbd)[1]

pos_quat_grad_jitted = jax.grad(pos_quat_to_weight, argnums=(0, 1))
step_pos = 1e-7
step_quat = 1e-7
CLIP_STEP = 0.001
CLIP_QUAT = 0.001
@jax.jit
def step_pose(position, quaternion, observed_rgbd):
    gp, gq = pos_quat_grad_jitted(position, quaternion, observed_rgbd)
    p_inc = step_pos * gp
    p_inc = jnp.where(jnp.linalg.norm(p_inc) > CLIP_STEP, p_inc / jnp.linalg.norm(p_inc) * CLIP_STEP, p_inc)
    q_inc = step_quat * gq
    q_inc = jnp.where(jnp.linalg.norm(q_inc) > CLIP_QUAT, q_inc / jnp.linalg.norm(q_inc) * CLIP_QUAT, q_inc)    
    position = position + p_inc
    quaternion = quaternion + q_inc
    quaternion = quaternion / jnp.linalg.norm(quaternion)
    return position, quaternion

position = X_WP.pos + jnp.array([0.0001, 0.002, -0.0031])
quaternion = X_WP._quaternion
STEPS_PER_FRAME=50
key = jax.random.PRNGKey(12)
for fr in tqdm(range(observed_rgbds.shape[0])):
    for step in tqdm(range(STEPS_PER_FRAME)):
        if step > 0:
            # position, quaternion = step_pose(position, quaternion, observed_rgbds[fr])
            key, subkey = jax.random.split(key)
            (position, quaternion) = step_pose(position, quaternion, observed_rgbds[fr])
        trace, weight = pos_quat_to_trace_and_weight(position, quaternion, observed_rgbds[fr])
        rr.set_time_sequence("gradient_ascent_step1", fr*STEPS_PER_FRAME + step)
        m.rr_viz_trace(trace, renderer)
        rr.log("/trace/logpdf", rr.Scalar(trace.get_score()))

    rr.set_time_sequence("frames_gradient_tracking", fr)
    m.rr_viz_trace(trace, renderer)

# position = X_WP.pos
# quaternion = X_WP._quaternion
# STEPS_PER_FRAME = 50
# for fr in tqdm(range(observed_rgbds.shape[0])):
#     for step in range(STEPS_PER_FRAME):
#         if step > 0:
#             position, quaternion = step_pose(position, quaternion, observed_rgbds[fr])

#     trace, weight = pos_quat_to_trace_and_weight(position, quaternion, observed_rgbds[fr])
#     rr.set_time_sequence("frame", fr)
#     m.rr_viz_trace(trace, renderer)

# position = X_WP.pos
# quaternion = X_WP._quaternion
# STEPS_PER_FRAME=10
# for fr in tqdm(range(observed_rgbds.shape[0])):
#     for step in tqdm(range(STEPS_PER_FRAME)):
#         if step > 0:
#             position, quaternion = step_pose(position, quaternion, observed_rgbds[fr])
#         trace, weight = pos_quat_to_trace_and_weight(position, quaternion, observed_rgbds[fr])
#         rr.set_time_sequence("gradient_ascent_step", fr*STEPS_PER_FRAME + step)
#         rr_viz_trace(trace, renderer)


# fr = 0
# for step in tqdm(range(200)):
#     if step > 0:
#         position, quaternion = step_pose(position, quaternion, observed_rgbds[fr])
#     trace, weight = pos_quat_to_trace_and_weight(position, quaternion, observed_rgbds[fr])
#     rr.set_time_sequence("gradient_ascent_step3", step)
#     rr_viz_trace(trace, renderer)
#     rr.log("/trace/logpdf", rr.Scalar(trace.get_score()))



# ########
# fr = 1

# for step in tqdm(range(STEPS_PER_FRAME)):
#     if step > 0:
#         position, quaternion = step_pose(position, quaternion, observed_rgbds[fr])
#     trace, weight = pos_quat_to_trace_and_weight(position, quaternion, observed_rgbds[fr])
#     rr.set_time_sequence("gradient_ascent_step", fr*STEPS_PER_FRAME + step)
#     rr_viz_trace(trace, renderer)

# fr = 2
# for step in tqdm(range(STEPS_PER_FRAME)):
#     if step > 0:
#         position, quaternion = step_pose(position, quaternion, observed_rgbds[fr])
#     trace, weight = pos_quat_to_trace_and_weight(position, quaternion, observed_rgbds[fr])
#     rr.set_time_sequence("gradient_ascent_step", fr*STEPS_PER_FRAME + step)
#     rr_viz_trace(trace, renderer)

# fr = 3
# for step in tqdm(range(STEPS_PER_FRAME)):
#     if step > 0:
#         position, quaternion = step_pose(position, quaternion, observed_rgbds[fr])
#     trace, weight = pos_quat_to_trace_and_weight(position, quaternion, observed_rgbds[fr])
#     rr.set_time_sequence("gradient_ascent_step", fr*STEPS_PER_FRAME + step)
#     rr_viz_trace(trace, renderer)
