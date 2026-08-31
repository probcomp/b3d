### Preliminaries ###

import time

import genjax
import jax
import jax.numpy as jnp
import rerun as rr
from tqdm import tqdm

import b3d
import b3d.chisight.dense.differentiable_renderer as r
import b3d.chisight.dense.likelihoods as l
import demos.differentiable_renderer.patch_tracking.demo_utils as du
import demos.differentiable_renderer.patch_tracking.model as m
from b3d import Pose

rr.init("single_patch_tracking-mh")
rr.connect("127.0.0.1:8812")

(
    renderer,
    (observed_rgbds, gt_rots),
    ((patch_vertices_P, patch_faces, patch_vertex_colors), X_WP),
    X_WC,
) = du.get_renderer_boxdata_and_patch()

hyperparams = r.DifferentiableRendererHyperparams(3, 1e-5, 1e-2, -1)

depth_scale = 0.0001
color_scale = 0.002
mindepth = -1.0
maxdepth = 2.0
likelihood = l.ArgMap(
    l.ImageDistFromPixelDist(
        l.uniform_multilaplace_mixture, [True, True, False, False, False, False]
    ),
    lambda weights, rgbds: (
        renderer.height,
        renderer.width,
        weights,
        rgbds,
        (depth_scale,),
        (color_scale,),
        mindepth,
        maxdepth,
    ),
)

model = m.single_object_model_factory(renderer, likelihood, hyperparams)

key = jax.random.PRNGKey(0)

### Generate image samples from the observation model ###


def generate_image(key):
    trace, _weight = model.importance(
        key,
        genjax.choice_map({"pose": X_WP, "camera_pose": X_WC}),
        (patch_vertices_P, patch_faces, patch_vertex_colors, ()),
    )
    return trace.get_retval()[0]


images = jax.vmap(generate_image)(jax.random.split(key, 100))
for i, image in enumerate(images):
    rr.set_time_sequence("image_sample", i)
    rr.log("/image_sample/rgb", rr.Image(image[:, :, :3]))
    rr.log("/image_sample/depth", rr.DepthImage(image[:, :, 3]))

### Patch tracking ###


def importance_from_pos_quat_v3(pos, quat, timestep):
    pose = Pose.from_vec(jnp.concatenate([pos, quat]))
    trace, weight = model.importance(
        key,
        genjax.choice_map(
            {
                "pose": pose,
                "camera_pose": X_WC,
                "observed_rgbd": observed_rgbds[timestep],
            }
        ),
        (patch_vertices_P, patch_faces, patch_vertex_colors, ()),
    )
    return trace, weight


trace, wt = importance_from_pos_quat_v3(X_WP._position, X_WP._quaternion, 0)


def weight_from_pos_quat_v3(pos, quat, timestep):
    return importance_from_pos_quat_v3(pos, quat, timestep)[1]


# ### Gaussian VMF MH patch tracking ###
def mh_drift_update(key, tr, pos_std, rot_concentration):
    key, subkey = jax.random.split(key)
    newpose = b3d.model.gaussian_vmf_pose.sample(
        subkey, tr["pose"], pos_std, rot_concentration
    )
    q_fwd = b3d.model.gaussian_vmf_pose.logpdf(
        newpose, tr["pose"], pos_std, rot_concentration
    )
    q_bwd = b3d.model.gaussian_vmf_pose.logpdf(
        tr["pose"], newpose, pos_std, rot_concentration
    )

    key, subkey = jax.random.split(key)
    proposed_trace, p_ratio, _, _ = tr.update(
        subkey,
        genjax.choice_map({"pose": newpose}),
        genjax.Diff.tree_diff_no_change(tr.get_args()),
    )

    log_full_ratio = p_ratio + q_bwd - q_fwd
    alpha = jnp.minimum(jnp.exp(log_full_ratio), 1.0)
    key, subkey = jax.random.split(key)
    accept = jax.random.bernoulli(subkey, alpha)

    new_trace = jax.lax.cond(accept, lambda _: proposed_trace, lambda _: tr, None)

    metadata = {
        "accept": accept,
        "log_p_ratio": p_ratio,
        "log_q_fwd": q_fwd,
        "log_q_bwd": q_bwd,
        "log_full_ratio": log_full_ratio,
        "alpha": alpha,
    }

    return (new_trace, metadata)


@jax.jit
def mh_kernel(st, i):
    key, tr, pos_std, rot_conc = st
    key, subkey = jax.random.split(key)
    new_tr, metadata = mh_drift_update(subkey, tr, pos_std, rot_conc)
    return (key, new_tr, pos_std, rot_conc), metadata


(_, new_tr, _, _), metadata = mh_kernel((key, trace, 0.01, 0.1), 0)


@jax.jit
def unfold_mh_100_steps(st):
    ret_st, metadata = jax.lax.scan(mh_kernel, st, jnp.arange(100))
    return ret_st, metadata


@jax.jit
def unfold_mh_400_steps(st):
    ret_st, metadata = jax.lax.scan(mh_kernel, st, jnp.arange(400))
    return ret_st, metadata


@jax.jit
def unfold_mh_1000_steps(st):
    ret_st, metadata = jax.lax.scan(mh_kernel, st, jnp.arange(1000))
    return ret_st, metadata


tr = trace
for timestep in tqdm(range(10)):
    key, subkey = jax.random.split(key)
    tr, wt, _, _ = tr.update(
        subkey,
        genjax.choice_map({"observed_rgbd": observed_rgbds[timestep]}),
        genjax.Diff.tree_diff_no_change(trace.get_args()),
    )

    key, subkey = jax.random.split(key)
    (_, tr, _, _), metadata = unfold_mh_1000_steps((subkey, tr, 1e-4, 1e4))
    n_accepted = jnp.sum(metadata["accept"])
    rr.set_time_sequence("frame--tracking-joint-mh", timestep)
    m.rr_log_trace(tr, renderer)
    rr.log("n_accepted", rr.Scalar(n_accepted))

### MH patch tracking, separately on position and rotation ###


def position_drift_mh_update(key, tr, pos_std, get_proposed_trace=False):
    key, subkey = jax.random.split(key)
    new_pos = genjax.normal.sample(subkey, tr["pose"]._position, pos_std)
    q_fwd = genjax.normal.logpdf(new_pos, tr["pose"].pos, pos_std)
    q_bwd = genjax.normal.logpdf(tr["pose"].pos, new_pos, pos_std)

    key, subkey = jax.random.split(key)
    proposed_trace, p_ratio, _, _ = tr.update(
        subkey,
        genjax.choice_map(
            {
                "pose": b3d.Pose.from_vec(
                    jnp.concatenate([new_pos, tr["pose"]._quaternion])
                )
            }
        ),
        genjax.Diff.tree_diff_no_change(tr.get_args()),
    )

    log_full_ratio = p_ratio + q_bwd - q_fwd
    alpha = jnp.minimum(jnp.exp(log_full_ratio), 1.0)
    key, subkey = jax.random.split(key)
    accept = jax.random.bernoulli(subkey, alpha)

    tr = jax.lax.cond(accept, lambda _: proposed_trace, lambda _: tr, None)
    metadata = {
        "accept": accept,
        "log_p_ratio": p_ratio,
        "log_q_fwd": q_fwd,
        "log_q_bwd": q_bwd,
        "log_full_ratio": log_full_ratio,
        "alpha": alpha,
    }
    if get_proposed_trace:
        metadata["proposed_trace"] = proposed_trace

    return (tr, metadata)


def rotation_drift_mh_update(key, tr, rot_conc, get_proposed_trace=False):
    key, subkey = jax.random.split(key)
    new_quat = b3d.model.vmf.sample(subkey, tr["pose"]._quaternion, rot_conc)
    q_fwd = b3d.model.vmf.logpdf(new_quat, tr["pose"]._quaternion, rot_conc)
    q_bwd = b3d.model.vmf.logpdf(tr["pose"]._quaternion, new_quat, rot_conc)

    key, subkey = jax.random.split(key)
    proposed_trace, p_ratio, _, _ = tr.update(
        subkey,
        genjax.choice_map(
            {"pose": b3d.Pose.from_vec(jnp.concatenate([tr["pose"].pos, new_quat]))}
        ),
        genjax.Diff.tree_diff_no_change(tr.get_args()),
    )

    log_full_ratio = p_ratio + q_bwd - q_fwd
    alpha = jnp.minimum(jnp.exp(log_full_ratio), 1.0)
    key, subkey = jax.random.split(key)
    accept = jax.random.bernoulli(subkey, alpha)

    tr = jax.lax.cond(accept, lambda _: proposed_trace, lambda _: tr, None)
    metadata = {
        "accept": accept,
        "log_p_ratio": p_ratio,
        "log_q_fwd": q_fwd,
        "log_q_bwd": q_bwd,
        "log_full_ratio": log_full_ratio,
        "alpha": alpha,
    }
    if get_proposed_trace:
        metadata["proposed_trace"] = proposed_trace

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
def unfold_pos_then_rot_mh_100_steps(st):
    ret_st, metadata = jax.lax.scan(pose_then_rot_mh_kernel, st, jnp.arange(200))
    return ret_st, metadata


@jax.jit
def unfold_pos_then_rot_mh_1000_steps(st):
    ret_st, metadata = jax.lax.scan(pose_then_rot_mh_kernel, st, jnp.arange(1000))
    return ret_st, metadata


@jax.jit
def multiple_mh_for_100_steps(key, tr, pos_stds, rot_concs):
    metadata = []
    for pos_std, rot_conc in zip(pos_stds, rot_concs):
        st = (key, tr, pos_std, rot_conc)
        ret_st, metadatum = jax.lax.scan(pose_then_rot_mh_kernel, st, jnp.arange(100))
        metadata.append(metadatum)
        (key, tr, _, _) = ret_st

    return ret_st, metadata


t = time.time()
((_, tr, _, _), metadata) = multiple_mh_for_100_steps(
    subkey, tr, [1e-3, 5e-4, 2e-4], [1e4, 4e5, 8e4]
)
print(time.time() - t)

tr = trace
for timestep in tqdm(range(30)):
    key, subkey = jax.random.split(key)
    tr, wt, _, _ = tr.update(
        subkey,
        genjax.choice_map({"observed_rgbd": observed_rgbds[timestep]}),
        genjax.Diff.tree_diff_no_change(trace.get_args()),
    )

    key, subkey = jax.random.split(key)
    ((_, tr, _, _), metadata) = multiple_mh_for_100_steps(
        subkey, tr, [1e-3, 5e-4, 2e-4], [1e4, 4e5, 8e4]
    )
    metadata1, metadata2, metadata3 = metadata
    # (_, tr, _, _), metadata1 = unfold_pos_then_rot_mh_100_steps((subkey, tr, 5e-4, 1e4))
    # (_, tr, _, _), metadata2 = unfold_pos_then_rot_mh_100_steps((subkey, tr, 3e-4, 3e5))
    # (_, tr, _, _), metadata3 = unfold_pos_then_rot_mh_100_steps((subkey, tr, 1e-4, 1e5))
    n_accepted_pos_1 = jnp.sum(metadata1["position"]["accept"])
    n_accepted_pos_2 = jnp.sum(metadata2["position"]["accept"])
    n_accepted_pos_3 = jnp.sum(metadata3["position"]["accept"])
    n_accepted_rot_1 = jnp.sum(metadata1["rotation"]["accept"])
    n_accepted_rot_2 = jnp.sum(metadata2["rotation"]["accept"])
    n_accepted_rot_3 = jnp.sum(metadata3["rotation"]["accept"])
    rr.set_time_sequence("frame--tracking-separate-mh-5", timestep)
    m.rr_log_trace(tr, renderer)
    rr.log("n_accepted_pos_1", rr.Scalar(n_accepted_pos_1))
    rr.log("n_accepted_pos_2", rr.Scalar(n_accepted_pos_2))
    rr.log("n_accepted_pos_3", rr.Scalar(n_accepted_pos_3))
    rr.log("n_accepted_rot_1", rr.Scalar(n_accepted_rot_1))
    rr.log("n_accepted_rot_2", rr.Scalar(n_accepted_rot_2))
    rr.log("n_accepted_rot_3", rr.Scalar(n_accepted_rot_3))


### Turn on the below to visualize running MH at one frame for a bit:
# timestep = 2
# tr, wt, _, _ = trace.update(
#     key,
#     genjax.choice_map({"observed_rgbd": observed_rgbds[timestep]}),
#     genjax.Diff.tree_diff_no_change(trace.get_args())
# )
# N_STEPS = 100
# pos_drift_mh = jax.jit(lambda *args: position_drift_mh_update(*args, get_proposed_trace=True))
# rot_drift_mh = jax.jit(lambda *args: rotation_drift_mh_update(*args, get_proposed_trace=True))
# for iter in range(N_STEPS):
#     key, subkey = jax.random.split(key)
#     old_tr = tr
#     tr, metadata_pos = pos_drift_mh(subkey, tr, 5e-4,)
#     key, subkey = jax.random.split(key)
#     tr, metadata_rot = rot_drift_mh(subkey, tr, 1e4)

#     rr.set_time_sequence("step4", iter)
#     m.rr_log_trace(metadata_pos["proposed_trace"], renderer, "proposed_trace_pos")
#     m.rr_log_trace(metadata_rot["proposed_trace"], renderer, "proposed_trace_rot")
#     m.rr_log_trace(tr, renderer, "post_update_trace")
#     m.rr_log_trace(old_tr, renderer, "pre_update_trace")

#     rr.log("accept_pos", rr.Scalar(metadata_pos["accept"]))
#     rr.log("accept_rot", rr.Scalar(metadata_rot["accept"]))


### Turn on the below to visualize orientation proposals:
# tr = trace
# for iter in range(10):
#     key, subkey = jax.random.split(key)
#     tr, metadata_rot = rot_drift_mh(subkey, tr, 1e4)
#     rr.set_time_sequence("proposal", iter)
#     m.rr_log_trace(metadata_rot["proposed_trace"], renderer, "proposed_trace_rot")
#     m.rr_log_trace(trace, renderer, "starting_trace")
