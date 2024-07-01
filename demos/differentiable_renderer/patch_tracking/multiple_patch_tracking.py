### Preliminaries ###

import jax.numpy as jnp
import jax
from b3d import Pose
import b3d
import rerun as rr
import genjax
from tqdm import tqdm
import demos.differentiable_renderer.patch_tracking.demo_utils as du
import demos.differentiable_renderer.patch_tracking.model as m
import b3d.chisight.dense.likelihoods as l
import b3d.chisight.dense.differentiable_renderer as r
import os
import trimesh
import matplotlib.pyplot as plt
import numpy as np
import optax

rr.init("multiple_patch_tracking")
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
rots = du.vec_transform_axis_angle(jnp.array([0,0,1]), jnp.linspace(jnp.pi/4, 3*jnp.pi/4, 30))
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
xyzs_C = b3d.xyz_from_depth_vectorized(depths, renderer.fx, renderer.fy, renderer.cx, renderer.cy)
xyzs_W = X_WC.apply(xyzs_C)

### Get patches ###

def all_pairs_2(X, Y):
    return jnp.swapaxes(
        jnp.stack(jnp.meshgrid(X, Y), axis=-1),
        0, 1
    ).reshape(-1, 2)

width_gradations = jnp.arange(44, 84, 6)
height_gradations = jnp.arange(38, 96, 6)
centers = all_pairs_2(height_gradations, width_gradations)

def get_patches(center):
    center_x, center_y = center[0], center[1]
    del_pix = 3
    patch_points_C = jax.lax.dynamic_slice(xyzs_C[0], (center_x-del_pix,center_y-del_pix,0), (2*del_pix-1,2*del_pix-1,3)).reshape(-1,3)
    patch_rgbs = jax.lax.dynamic_slice(rgbs[0], (center_x-del_pix,center_y-del_pix,0), (2*del_pix-1,2*del_pix-1,3)).reshape(-1,3)
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
        triangle_indices=patch_faces[i],
        vertex_colors=patch_vertex_colors[i]
    ))


###
hyperparams = r.DifferentiableRendererHyperparams(
    3, 1e-5, 1e-2, -1
)

depth_scale = 0.0001
color_scale = 0.002
mindepth = -1.0
maxdepth = 2.0
likelihood = l.get_uniform_multilaplace_image_dist_with_fixed_params(
    renderer.height, renderer.width, depth_scale, color_scale, mindepth, maxdepth
)

model = m.multiple_object_model_factory(
    renderer,
    likelihood,
    hyperparams
)

key = jax.random.PRNGKey(3)

trace, weight = model.importance(
    key,
    genjax.choice_map({
        "poses": genjax.vector_choice_map(genjax.choice(Xs_WP)),
        "camera_pose": X_WC,
    }),
    (patch_vertices_P, patch_faces, patch_vertex_colors, ())
)


###

@jax.jit
def importance_from_pos_quat_v3(positions, quaternions, timestep):
    poses = jax.vmap(lambda pos, quat: Pose.from_vec(jnp.concatenate([pos, quat])), in_axes=(0, 0))(positions, quaternions)
    trace, weight = model.importance(
        key,
        genjax.choice_map({
            "poses": genjax.vector_choice_map(genjax.choice(poses)),
            "camera_pose": X_WC,
            "observed_rgbd": observed_rgbds[timestep]
        }),
        (patch_vertices_P, patch_faces, patch_vertex_colors, ())
    )
    return trace, weight

trace, wt = importance_from_pos_quat_v3(Xs_WP._position, Xs_WP._quaternion, 0)
assert jnp.all(trace["poses"].inner.value._position == Xs_WP._position)
assert jnp.all(trace["poses"].inner.value._quaternion == Xs_WP._quaternion)
m.rr_log_multiobject_trace(trace, renderer)

def weight_from_pos_quat_v3(pos, quat, timestep):
    return importance_from_pos_quat_v3(pos, quat, timestep)[1]

grad_jitted_3 = jax.jit(jax.grad(weight_from_pos_quat_v3, argnums=(0, 1,)))

# V1: try having all positions share an optimizer state,
# and have all quaternions share an optimizer state

optimizer_pos = optax.adam(learning_rate=1e-4, b1=0.7)
optimizer_quat = optax.adam(learning_rate=4e-3)

@jax.jit
def optimizer_kernel(st, i):
    opt_state_pos, opt_state_quat, pos, quat, timestep = st
    grad_pos, grad_quat = grad_jitted_3(pos, quat, timestep)
    updates_pos, opt_state_pos = optimizer_pos.update(-grad_pos, opt_state_pos)
    updates_quat, opt_state_quat = optimizer_quat.update(-grad_quat, opt_state_quat)
    pos = optax.apply_updates(pos, updates_pos)
    quat = optax.apply_updates(quat, updates_quat)
    return (opt_state_pos, opt_state_quat, pos, quat, timestep), (pos, quat)

@jax.jit
def unfold_300_steps(st):
    ret_st, _ = jax.lax.scan(optimizer_kernel, st, jnp.arange(300))
    return ret_st

@jax.jit
def unfold_600_steps(st):
    ret_st, _ = jax.lax.scan(optimizer_kernel, st, jnp.arange(600))
    return ret_st

opt_state_pos = optimizer_pos.init(Xs_WP._position)
opt_state_quat = optimizer_quat.init(Xs_WP._quaternion)
pos = Xs_WP._position
quat = Xs_WP._quaternion
positions = []
quaternions = []
for timestep in tqdm(range(30)):
    opt_state_pos = optimizer_pos.init(pos)
    opt_state_quat = optimizer_quat.init(quat)
    (opt_state_pos, opt_state_quat, pos, quat, _) = unfold_300_steps(
        (opt_state_pos, opt_state_quat, pos, quat, timestep)
    )
    tr, weight = importance_from_pos_quat_v3(pos, quat, timestep)
    positions.append(pos)
    quaternions.append(quat)
    rr.set_time_sequence("frame--tracking", timestep)
    m.rr_log_multiobject_trace(tr, renderer)

for i in range(observed_rgbds.shape[0]):
    rr.set_time_sequence("frame--tracking", i)

    rr.log("/3D/tracked_points", rr.Points3D(
        positions = positions[i],
        radii=0.0075*np.ones(positions[i].shape[0]),
        colors=np.repeat(np.array([0,0,255])[None,...], positions[i].shape[0], axis=0))
    )
