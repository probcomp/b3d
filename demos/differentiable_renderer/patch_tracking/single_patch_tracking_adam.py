### Preliminaries ###

import genjax
import jax
import jax.numpy as jnp
import optax
import rerun as rr
from tqdm import tqdm

import b3d.chisight.dense.differentiable_renderer as r
import b3d.chisight.dense.likelihoods as l
import demos.differentiable_renderer.patch_tracking.demo_utils as du
import demos.differentiable_renderer.patch_tracking.model as m
from b3d import Pose

rr.init("single_patch_tracking")
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
likelihood = l.get_uniform_multilaplace_image_dist_with_fixed_params(
    renderer.height, renderer.width, depth_scale, color_scale, mindepth, maxdepth
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


grad_jitted_3 = jax.jit(
    jax.grad(
        weight_from_pos_quat_v3,
        argnums=(
            0,
            1,
        ),
    )
)

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


opt_state_pos = optimizer_pos.init(X_WP._position)
opt_state_quat = optimizer_quat.init(X_WP._quaternion)
pos = X_WP._position
quat = X_WP._quaternion
for timestep in tqdm(range(30)):
    opt_state_pos = optimizer_pos.init(pos)
    opt_state_quat = optimizer_quat.init(quat)
    (opt_state_pos, opt_state_quat, pos, quat, _) = unfold_100_steps(
        (opt_state_pos, opt_state_quat, pos, quat, timestep)
    )
    tr, weight = importance_from_pos_quat_v3(pos, quat, timestep)
    rr.set_time_sequence("frame--tracking", timestep)
    m.rr_log_trace(tr, renderer)

### The following code is slow, but can be used to visualize the whole GD sequence.
# opt_state_pos = optimizer_pos.init(X_WP._position)
# opt_state_quat = optimizer_quat.init(X_WP._quaternion)
# pos = X_WP._position
# quat = X_WP._quaternion
# N_STEPS = 80
# for timestep in range(10):
#     opt_state_pos = optimizer_pos.init(pos)
#     opt_state_quat = optimizer_quat.init(quat)
#     for i in range(N_STEPS):
#         (opt_state_pos, opt_state_quat, pos, quat, _), _ = optimizer_kernel(
#             (opt_state_pos, opt_state_quat, pos, quat, timestep), i
#         )
#         tr, weight = importance_from_pos_quat_v3(pos, quat, timestep)
#         rr.set_time_sequence("full_seq-10", i + timestep * N_STEPS)
#         m.rr_log_trace(tr, renderer)
#         rr.log("weight", rr.Scalar(weight))
#     tr, weight = importance_from_pos_quat_v3(pos, quat, timestep)
#     rr.set_time_sequence("tracking-frame-10", timestep)
#     m.rr_log_trace(tr, renderer)
