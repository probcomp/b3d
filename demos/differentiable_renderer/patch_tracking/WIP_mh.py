### Preliminaries ###

import jax.numpy as jnp
import jax
from b3d import Pose
import rerun as rr
import genjax
from tqdm import tqdm
import demos.differentiable_renderer.patch_tracking.demo_utils as du
import demos.differentiable_renderer.patch_tracking.model as m
import b3d.likelihoods as l
import b3d.differentiable_renderer as r
import matplotlib.pyplot as plt
import numpy as np
import b3d
import optax

rr.init("single_patch_tracking")
rr.connect("127.0.0.1:8812")

(
    renderer,
    (observed_rgbds, gt_rots),
    ((patch_vertices_P, patch_faces, patch_vertex_colors), X_WP),
    X_WC
) = du.get_renderer_boxdata_and_patch()

hyperparams = r.DifferentiableRendererHyperparams(
    3, 1e-5, 1e-2, -1
)

depth_scale = 0.0001
color_scale = 0.002
mindepth = -1.0
maxdepth = 2.0
likelihood = l.ArgMap(
    l.ImageDistFromPixelDist(
        l.uniform_multilaplace_mixture,
        [True, True, False, False, False, False]
    ),
    lambda weights, rgbds: ( renderer.height, renderer.width,
                            weights, rgbds, (depth_scale,), (color_scale,), mindepth, maxdepth )
)

model = m.single_object_model_factory(
    renderer,
    likelihood,
    hyperparams
)

key = jax.random.PRNGKey(0)

### Generate image samples from the observation model ###

def generate_image(key):
    trace, weight = model.importance(
        key,
        genjax.choice_map({ "pose": X_WP, "camera_pose": X_WC }),
        (patch_vertices_P, patch_faces, patch_vertex_colors, ())
    )
    return trace.get_retval()[0]
images = jax.vmap(generate_image)(jax.random.split(key, 100))
for i, image in enumerate(images):
    rr.set_time_sequence("image_sample", i)
    rr.log(f"/image_sample/rgb", rr.Image(image[:, :, :3]))
    rr.log(f"/image_sample/depth", rr.DepthImage(image[:, :, 3]))

### Patch tracking ###

def importance_from_pos_quat_v3(pos, quat, timestep):
    pose = Pose.from_vec(jnp.concatenate([pos, quat]))
    trace, weight = model.importance(
        key,
        genjax.choice_map({
            "pose": pose,
            "camera_pose": X_WC,
            "observed_rgbd": observed_rgbds[timestep]
        }),
        (patch_vertices_P, patch_faces, patch_vertex_colors, ())
    )
    return trace, weight

trace, wt = importance_from_pos_quat_v3(X_WP._position, X_WP._quaternion, 0)

def weight_from_pos_quat_v3(pos, quat, timestep):
    return importance_from_pos_quat_v3(pos, quat, timestep)[1]

grad_jitted_3 = jax.jit(jax.grad(weight_from_pos_quat_v3, argnums=(0, 1,)))

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
def unfold_100_steps(st):
    ret_st, _ = jax.lax.scan(optimizer_kernel, st, jnp.arange(100))
    return ret_st

# opt_state_pos = optimizer_pos.init(X_WP._position)
# opt_state_quat = optimizer_quat.init(X_WP._quaternion)
# pos = X_WP._position
# quat = X_WP._quaternion
# for timestep in tqdm(range(30)):
#     opt_state_pos = optimizer_pos.init(pos)
#     opt_state_quat = optimizer_quat.init(quat)
#     (opt_state_pos, opt_state_quat, pos, quat, _) = unfold_100_steps(
#         (opt_state_pos, opt_state_quat, pos, quat, timestep)
#     )
#     tr, weight = importance_from_pos_quat_v3(pos, quat, timestep)
#     rr.set_time_sequence("frame--tracking", timestep)
#     m.rr_log_trace(tr, renderer)

# ### The following code is slow, but can be used to visualize the whole GD sequence.
# # opt_state_pos = optimizer_pos.init(X_WP._position)
# # opt_state_quat = optimizer_quat.init(X_WP._quaternion)
# # pos = X_WP._position
# # quat = X_WP._quaternion
# # N_STEPS = 80
# # for timestep in range(10):
# #     opt_state_pos = optimizer_pos.init(pos)
# #     opt_state_quat = optimizer_quat.init(quat)
# #     for i in range(N_STEPS):
# #         (opt_state_pos, opt_state_quat, pos, quat, _), _ = optimizer_kernel(
# #             (opt_state_pos, opt_state_quat, pos, quat, timestep), i
# #         )
# #         tr, weight = importance_from_pos_quat_v3(pos, quat, timestep)
# #         rr.set_time_sequence("full_seq-10", i + timestep * N_STEPS)
# #         m.rr_log_trace(tr, renderer)
# #         rr.log("weight", rr.Scalar(weight))
# #     tr, weight = importance_from_pos_quat_v3(pos, quat, timestep)
# #     rr.set_time_sequence("tracking-frame-10", timestep)
# #     m.rr_log_trace(tr, renderer)

# ### Gaussian VMF MH patch tracking ###
# def mh_drift_update(key, tr, pos_std, rot_concentration):
#     key, subkey = jax.random.split(key)
#     newpose = b3d.model.gaussian_vmf_pose.sample(
#         subkey, tr["pose"], pos_std, rot_concentration
#     )
#     q_fwd = b3d.model.gaussian_vmf_pose.logpdf(
#         newpose, tr["pose"], pos_std, rot_concentration
#     )
#     q_bwd = b3d.model.gaussian_vmf_pose.logpdf(
#         tr["pose"], newpose, pos_std, rot_concentration
#     )

#     key, subkey = jax.random.split(key)
#     proposed_trace, p_ratio, _, _ = tr.update(
#         subkey,
#         genjax.choice_map({"pose": newpose}),
#         genjax.Diff.tree_diff_no_change(tr.get_args())
#     )

#     log_full_ratio = p_ratio + q_bwd - q_fwd
#     alpha = jnp.minimum(jnp.exp(log_full_ratio), 1.0)
#     key, subkey = jax.random.split(key)
#     accept = jax.random.bernoulli(subkey, alpha)

#     new_trace = jax.lax.cond(
#         accept,
#         lambda _: proposed_trace,
#         lambda _: tr,
#         None
#     )

#     metadata = {
#         "accept": accept,
#         "log_p_ratio": p_ratio,
#         "log_q_fwd": q_fwd,
#         "log_q_bwd": q_bwd,
#         "log_full_ratio": log_full_ratio,
#         "alpha": alpha
#     }

#     return (new_trace, metadata)

# @jax.jit
# def mh_kernel(st, i):
#     key, tr, pos_std, rot_conc = st
#     key, subkey = jax.random.split(key)
#     new_tr, metadata = mh_drift_update(subkey, tr, pos_std, rot_conc)
#     return (key, new_tr, pos_std, rot_conc), metadata

# (_, new_tr, _, _), metadata = mh_kernel((key, trace, 0.01, 0.1), 0)

# @jax.jit
# def unfold_mh_100_steps(st):
#     ret_st, metadata = jax.lax.scan(mh_kernel, st, jnp.arange(100))
#     return ret_st, metadata

# @jax.jit
# def unfold_mh_400_steps(st):
#     ret_st, metadata = jax.lax.scan(mh_kernel, st, jnp.arange(400))
#     return ret_st, metadata

# @jax.jit
# def unfold_mh_1000_steps(st):
#     ret_st, metadata = jax.lax.scan(mh_kernel, st, jnp.arange(1000))
#     return ret_st, metadata

# tr = trace
# for timestep in tqdm(range(10)):
#     key, subkey = jax.random.split(key)
#     tr, wt, _, _ = tr.update(
#         subkey,
#         genjax.choice_map({"observed_rgbd": observed_rgbds[timestep]}),
#         genjax.Diff.tree_diff_no_change(trace.get_args())
#     )

#     key, subkey = jax.random.split(key)
#     (_, tr, _, _), metadata = unfold_mh_1000_steps((subkey, tr, 0.00005, 0.00005))
#     n_accepted = jnp.sum(metadata["accept"])
#     rr.set_time_sequence("frame--tracking-joint-mh", timestep)
#     m.rr_log_trace(tr, renderer)
#     rr.log("n_accepted", rr.Scalar(n_accepted))

### MH patch tracking, separately on position and rotation ###

def position_drift_mh_update(key, tr, pos_std):
    key, subkey = jax.random.split(key)
    new_pos = genjax.normal.sample(
        subkey, tr["pose"]._position, pos_std
    )
    q_fwd = genjax.normal.logpdf(new_pos, tr["pose"].pos, pos_std)
    q_bwd = genjax.normal.logpdf(tr["pose"].pos, new_pos, pos_std)

    key, subkey = jax.random.split(key)
    proposed_trace, p_ratio, _, _ = tr.update(subkey,
        genjax.choice_map({"pose": b3d.Pose.from_vec(jnp.concatenate([new_pos, tr["pose"]._quaternion]))}),
        genjax.Diff.tree_diff_no_change(tr.get_args())
    )

    log_full_ratio = p_ratio + q_bwd - q_fwd
    alpha = jnp.minimum(jnp.exp(log_full_ratio), 1.0)
    key, subkey = jax.random.split(key)
    accept = jax.random.bernoulli(subkey, alpha)

    tr = jax.lax.cond(accept, lambda _: proposed_trace, lambda _: tr, None)
    metadata = {
        "accept": accept, "log_p_ratio": p_ratio, "log_q_fwd": q_fwd,
        "log_q_bwd": q_bwd, "log_full_ratio": log_full_ratio, "alpha": alpha
    }

    return (tr, metadata)

def rotation_drift_mh_update(key, tr, rot_conc):
    key, subkey = jax.random.split(key)
    new_quat = b3d.model.vmf.sample(subkey, tr["pose"]._quaternion, rot_conc)
    q_fwd = b3d.model.vmf.logpdf(new_quat, tr["pose"]._quaternion, rot_conc)
    q_bwd = b3d.model.vmf.logpdf(tr["pose"]._quaternion, new_quat, rot_conc)

    key, subkey = jax.random.split(key)
    proposed_trace, p_ratio, _, _ = tr.update(subkey,
        genjax.choice_map({"pose": b3d.Pose.from_vec(jnp.concatenate([tr["pose"].pos, new_quat]))}),
        genjax.Diff.tree_diff_no_change(tr.get_args())
    )

    log_full_ratio = p_ratio + q_bwd - q_fwd
    alpha = jnp.minimum(jnp.exp(log_full_ratio), 1.0)
    key, subkey = jax.random.split(key)
    accept = jax.random.bernoulli(subkey, alpha)

    tr = jax.lax.cond(accept, lambda _: proposed_trace, lambda _: tr, None)
    metadata = {
        "accept": accept, "log_p_ratio": p_ratio, "log_q_fwd": q_fwd,
        "log_q_bwd": q_bwd, "log_full_ratio": log_full_ratio, "alpha": alpha
    }

    return (tr, metadata)

def pos_then_rot_mh_drift_update(key, tr, pos_std, rot_concentration):
    key, subkey = jax.random.split(key)
    tr, metadata_pos = position_drift_mh_update(subkey, tr, pos_std)
    key, subkey = jax.random.split(key)
    tr, metadata_rot = rotation_drift_mh_update(subkey, tr, rot_concentration)
    metadata = {"position": metadata_pos, "rotation": metadata_rot}
    return (tr, metadata)

trace_test, metadata = pos_then_rot_mh_drift_update(key, trace, 0.01, 0.1)

@jax.jit
def pose_then_rot_mh_kernel(st, i):
    key, tr, pos_std, rot_conc = st
    key, subkey = jax.random.split(key)
    new_tr, metadata = pos_then_rot_mh_drift_update(subkey, tr, pos_std, rot_conc)
    return (key, new_tr, pos_std, rot_conc), metadata

@jax.jit
def unfold_pos_then_rot_mh_1000_steps(st):
    ret_st, metadata = jax.lax.scan(pose_then_rot_mh_kernel, st, jnp.arange(1000))
    return ret_st, metadata

tr = trace
for timestep in tqdm(range(10)):
    key, subkey = jax.random.split(key)
    tr, wt, _, _ = tr.update(
        subkey,
        genjax.choice_map({"observed_rgbd": observed_rgbds[timestep]}),
        genjax.Diff.tree_diff_no_change(trace.get_args())
    )

    key, subkey = jax.random.split(key)
    (_, tr, _, _), metadata = unfold_pos_then_rot_mh_1000_steps((subkey, tr, 5e-5, 5e-3))
    n_accepted_pos = jnp.sum(metadata["position"]["accept"])
    n_accepted_rot = jnp.sum(metadata["rotation"]["accept"])
    rr.set_time_sequence("frame--tracking-separate-mh-5", timestep)
    m.rr_log_trace(tr, renderer)
    rr.log("n_accepted_pos", rr.Scalar(n_accepted_pos))
    rr.log("n_accepted_rot", rr.Scalar(n_accepted_rot))