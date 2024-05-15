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
import time

# W = world frame; P = patch frame; C = camera frame

rr.init("multitrack_test-1")
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

def all_pairs_2(X, Y):
    return jnp.swapaxes(
        jnp.stack(jnp.meshgrid(X, Y), axis=-1),
        0, 1
    ).reshape(-1, 2)

width_gradations = jnp.arange(44, 84, 3)
height_gradations = jnp.arange(38, 96, 3)
centers = all_pairs_2(height_gradations, width_gradations)

def get_patches(center):
    center_x, center_y = center[0], center[1]
    del_pix = 3
    patch_points_C = jax.lax.dynamic_slice(xyzs_C[0], (center_x-del_pix,center_y-del_pix,0), (2*del_pix,2*del_pix,3)).reshape(-1,3)
    patch_rgbs = jax.lax.dynamic_slice(rgbs[0], (center_x-del_pix,center_y-del_pix,0), (2*del_pix,2*del_pix,3)).reshape(-1,3)
    patch_vertices_C, patch_faces, patch_vertex_colors, patch_face_colors = b3d.make_mesh_from_point_cloud_and_resolution(
        patch_points_C, patch_rgbs, patch_points_C[:,2] / fx * 2.0
)
    X_CP = Pose.from_translation(patch_vertices_C.mean(0))
    X_WP = X_WC @ X_CP
    patch_vertices_P = X_CP.inv().apply(patch_vertices_C)
    return (patch_vertices_P, patch_faces, patch_vertex_colors, X_WP)

(patch_vertices_P, patch_faces, patch_vertex_colors, Xs_WP) = jax.vmap(get_patches, in_axes=(0,))(centers)

rr.set_time_sequence("frame", 0)
for i in range(patch_vertices_P.shape[0]):
    rr.log("/3D/patch/{}".format(i), rr.Mesh3D(
        vertex_positions=Xs_WP[i].apply(patch_vertices_P[i]),
        indices=patch_faces[i],
        vertex_colors=patch_vertex_colors[i]
    ))

### Set up diff rend ###
hyperparams = r.DifferentiableRendererHyperparams(
    3, 1e-5, 1e-2, -1
)
importlib.reload(m)
model = m.model_multiobject_gl_factory(
    renderer,
    hyperparams=hyperparams,
    depth_noise_scale=1e-2,
    lab_noise_scale = 0.5,
    mindepth=-1.,
    maxdepth=1.0
)
# trace = model.simulate(key, (patch_vertices_P, patch_faces, patch_vertex_colors))

key = jax.random.PRNGKey(2)
trace, weight = model.importance(
    key,
    genjax.choice_map({
        "camera_pose": X_WC,
        "poses": genjax.vector_choice_map(genjax.choice(Xs_WP)),
        "observed_rgbd": genjax.vector_choice_map(genjax.vector_choice_map(genjax.choice(observed_rgbds[0])))
    }),
    (patch_vertices_P, patch_faces, patch_vertex_colors)
)

rr.set_time_sequence("frame", 0)
m.rr_viz_multiobject_trace(trace, renderer)

##### Tracking ######
#                          b3d.Pose[N]    (H, W, 4) array
def poses_to_trace_and_weight(poses, observed_rgbd):
    vcm = genjax.vector_choice_map(genjax.vector_choice_map(genjax.choice(observed_rgbd)))
    constraints = genjax.choice_map({
                             "poses": genjax.vector_choice_map(genjax.choice(poses)),
                             "camera_pose": X_WC,
                             "observed_rgbd": vcm
                         })
    trace, weight = model.importance(jax.random.PRNGKey(0), constraints, (patch_vertices_P, patch_faces, patch_vertex_colors))
    return trace, weight

@jax.jit
def poss_quats_to_trace_and_weight(positions, quaternions, observed_rgbd):
    pose = b3d.Pose(positions, quaternions)
    return poses_to_trace_and_weight(pose, observed_rgbd)

def poss_quats_to_weight(positions, quaternions, observed_rgbd):
    return poss_quats_to_trace_and_weight(positions, quaternions, observed_rgbd)[1]

pos_quat_grad_jitted = jax.grad(poss_quats_to_weight, argnums=(0, 1))
step_pos = 2e-7
step_quat = 2e-7
CLIP_STEP = 0.001
CLIP_QUAT = 0.001
@jax.jit
def step_pose(position, quaternion, observed_rgbd):
    gp, gq = pos_quat_grad_jitted(position, quaternion, observed_rgbd)
    p_inc = step_pos * gp
    p_inc = jnp.where((jnp.linalg.norm(p_inc, axis=-1) > CLIP_STEP)[:, None], p_inc / jnp.linalg.norm(p_inc, axis=-1)[:, None] * CLIP_STEP, p_inc)
    q_inc = step_quat * gq
    q_inc = jnp.where((jnp.linalg.norm(q_inc, axis=-1) > CLIP_QUAT)[:, None], q_inc / jnp.linalg.norm(q_inc, axis=-1)[:, None] * CLIP_QUAT, q_inc)
    position = position + p_inc
    quaternion = quaternion + q_inc
    quaternion = quaternion / jnp.linalg.norm(quaternion, axis=-1)[:, None]
    return position, quaternion

# positions = Xs_WP.pos
# quaternions = Xs_WP._quaternion
# STEPS_PER_FRAME=20
# for fr in tqdm(range(observed_rgbds.shape[0])):
#     for step in tqdm(range(STEPS_PER_FRAME)):
#         if step > 0:
#             # position, quaternion = step_pose(position, quaternion, observed_rgbds[fr])
#             key, subkey = jax.random.split(key)
#             (positions, quaternions) = step_pose(positions, quaternions, observed_rgbds[fr])
#         trace, weight = poss_quats_to_trace_and_weight(positions, quaternions, observed_rgbds[fr])
#         rr.set_time_sequence("gradient_ascent_step1", fr*STEPS_PER_FRAME + step)
#         m.rr_viz_multiobject_trace(trace, renderer)
#         rr.log("/trace/logpdf", rr.Scalar(trace.get_score()))

#     rr.set_time_sequence("frames_gradient_tracking", fr)
#     m.rr_viz_multiobject_trace(trace, renderer)

@jax.jit
def run_gd_step(carry, fr):
    positions, quaternions = carry
    (positions, quaternions) = step_pose(positions, quaternions, observed_rgbds[fr])
    return ((positions, quaternions), (positions, quaternions))

@jax.jit
def run_frame(carry, fr):
    positions, quaternions = carry
    (positions, quaternions), _ = jax.lax.scan(run_gd_step, (positions, quaternions), jnp.repeat(fr, STEPS_PER_FRAME))
    return (positions, quaternions), (positions, quaternions)

@jax.jit
def run_inference(p, q):
    return jax.lax.scan(run_frame, (p, q), jnp.arange(observed_rgbds.shape[0]))[1]

# carry, seq = jax.lax.scan(run_gd_step, (positions, quaternions), jnp.arange(STEPS_PER_FRAME))
positions = Xs_WP.pos
quaternions = Xs_WP._quaternion
STEPS_PER_FRAME=20
step_pos = 1e-7
step_quat = 1e-7
t = time.time()
seq = run_inference(positions, quaternions)
print(time.time() - t)
for i in range(observed_rgbds.shape[0]):
    rr.set_time_sequence("frame", i)
    rr.log("/3D/tracked_points", rr.Points3D(
        positions = seq[0][i],
        radii=0.0075*np.ones(seq[0][i].shape[0]),
        colors=np.repeat(np.array([0,0,255])[None,...], seq[0][i].shape[0], axis=0))
    )
