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

### Some stuff for debugging: ###

# observed_rgbd = jnp.concatenate([rgbs[0], depths[0, :, :, None]], axis=-1)


# hyperparams = r.DifferentiableRendererHyperparams(
#     3, 1e-5, 1e-2, -1
# )
# weights, attributes = r.render_to_rgbd_dist_params(
#     renderer, patch_vertices_C, patch_faces, patch_vertex_colors, hyperparams
# )

# weight = likelihoods.mixture_rgbd_pixel_model.logpdf(
#     observed_rgbd[0, 0], weights[0, 0], attributes[0, 0], 3.0, 0.07, 0., 20.
# )

# v = jax.jit(jax.value_and_grad(likelihoods.mixture_rgbd_pixel_model.logpdf,
#                        argnums=(1, 2)
#                        ))(
#     observed_rgbd[0, 0], weights[0, 0], attributes[0, 0], 3.0, 0.07, 0., 20.
# )

# ###
# v2 = jax.value_and_grad(likelihoods.laplace_rgb_pixel_model.logpdf,
#                         argnums=(1,)
#                         )(observed_rgbd[0, 0, :3], attributes[0, 0, 0, :3], 3.0)







### Some stuff I used for debugging: ###

# vertices, faces, vertex_rgbs = patch_vertices_C, patch_faces, patch_vertex_colors

# def model_weight_comp_manual(observed_rgbd, vertices_O, faces, vertex_colors, X_WO, X_WC, hyperparams=r.DEFAULT_HYPERPARAMS):
#         vertices_W = X_WO.apply(vertices_O)
#         vertices_C = X_WC.inv().apply(vertices_W)
#         weights, attributes = r.render_to_rgbd_dist_params(
#             renderer, vertices_C, faces, vertex_colors, hyperparams
#         )
#         weight = likelihoods.mixture_rgbd_pixel_model.logpdf(
#             observed_rgbd, weights[0, 0], attributes[0, 0], 3.0, 0.07, 0., 20.
#         )
#         # tr, weight = likelihoods.mixture_rgbd_sensor_model.importance(
#         #     jax.random.PRNGKey(0),
#         #     genjax.vector_choice_map(genjax.vector_choice_map(genjax.choice(observed_rgbd))),
#         #     (weights, attributes, 3.0, 0.07, 0., 10.)
#         # )
#         return weight

# def vertices_to_score_manual(vertices):
#     return model_weight_comp_manual(
#             observed_rgbd, vertices, faces, vertex_rgbs, X_WC, X_WC
#         )
# jax.grad(vertices_to_score_manual)(vertices)



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

model = m.model_singleobject_gl_factory(renderer, hyperparams=hyperparams, depth_noise_scale=1e-5, lab_noise_scale = 0.5)
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

# position = X_WP.pos
# step = 1e-7
# for i in range(10):
#     print("i = ", i)
#     if i > 0:
#         position = position + step * jax.grad(position_to_weight)(position)
#     trace, weight = position_to_trace_and_weight(position)
#     rr.set_time_sequence("gradient_ascent", i)
#     rr.log("logpdf", rr.Scalar(weight))
#     m.rr_viz_trace(trace, renderer)
    
#     rr.log("/patchimg/observed_img", rr.Image(rgbs[0]))
#     rr.log("/patchimg/opengl_patch", rr.Image(rgbs2))
#     rr.log("/patchimg/rendered_patch", rr.Image(trace.get_retval()[1][:, :, :3]))

###

@jax.jit
def pos_quat_to_trace_and_weight(position, quaternion):
    pose = b3d.Pose(position, quaternion)
    return pose_to_trace_and_weight(pose)

def pos_quat_to_weight(position, quaternion):
    return pos_quat_to_trace_and_weight(position, quaternion)[1]
def pos_quat_to_trace(position, quaternion):
    return pos_quat_to_trace_and_weight(position, quaternion)[0]

pos_quat_grad_jitted = jax.grad(pos_quat_to_weight, argnums=(0, 1))
position = X_WP.pos
quaternion = X_WP._quaternion
step_pos = 1e-3
step_quat = 1e-3
CLIP = 0.002
for i in range(200):
    print("i = ", i)
    if i > 0:
        gp, gq = pos_quat_grad_jitted(position, quaternion)
        print("gp: ", gp)
        print("gq: ", gq)
        if jnp.logical_or(jnp.any(jnp.isnan(gp)), jnp.any(jnp.isnan(gq))):
            break
        
        p_inc = step_pos * gp
        p_inc = jnp.where(jnp.linalg.norm(p_inc) > CLIP, p_inc / jnp.linalg.norm(p_inc) * CLIP, p_inc)
        q_inc = step_quat * gq
        q_inc = jnp.where(jnp.linalg.norm(q_inc) > CLIP, q_inc / jnp.linalg.norm(q_inc) * CLIP, q_inc)

        position = position + p_inc
        quaternion = quaternion + q_inc
        quaternion = quaternion / jnp.linalg.norm(quaternion)
    trace, weight = pos_quat_to_trace_and_weight(position, quaternion)
    if jnp.isinf(weight):
        break
    print("weight: ", weight)
    rr.set_time_sequence("gradient_ascent_pos_rot", i)
    rr.log("logpdf", rr.Scalar(weight))
    m.rr_viz_trace(trace, renderer)
    
    rr.log("/patchimg/observed_img", rr.Image(rgbs[0]))
    rr.log("/patchimg/opengl_patch", rr.Image(rgbs2))
    rr.log("/patchimg/rendered_patch", rr.Image(trace.get_retval()[1][:, :, :3]))