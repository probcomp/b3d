import rerun as rr
import genjax
import os
import numpy as np
import jax.numpy as jnp
import jax
from b3d import Pose
import b3d
from tqdm import tqdm
import trimesh
import genjax

PORT = 8812
rr.init("real")
rr.connect(addr=f"127.0.0.1:{PORT}")

data = jnp.load(b3d.get_root_path() / "assets/shared_data_bucket/datasets/posterior_uncertainty_mug_handle_w_0.02.npz")

scaling_factor = 3
image_width, image_height, fx, fy, cx, cy, near, far = (
    jnp.array(data["camera_intrinsics"]) / scaling_factor
)
image_width, image_height = int(image_width), int(image_height)
fx, fy, cx, cy, near, far = (
    float(fx),
    float(fy),
    float(cx),
    float(cy),
    float(near),
    float(far),
)

_rgb = data["rgb"]
_depth = data["depth"]
rgbs = jnp.clip(
    jax.image.resize(_rgb, (len(_rgb), image_height, image_width, 3), "nearest"), 0.0, 1.0
)
depths = jax.image.resize(_depth, (len(_rgb), image_height, image_width), "nearest")


# rr.log(
#     "point_cloud", rr.Points3D(point_cloud.reshape(-1, 3), colors=rgb.reshape(-1, 3))
# )
# table_pose, table_dims = b3d.Pose.fit_table_plane(point_cloud, 0.01, 0.01, 100, 1000)
# b3d.rr_log_pose("table", table_pose)

object_library = b3d.MeshLibrary.make_empty_library()
mesh_path = os.path.join(
    b3d.get_root_path(),
    "assets/shared_data_bucket/ycb_video_models/models/025_mug/textured_simple.obj",
)
object_library.add_trimesh(trimesh.load(mesh_path))


color_error, depth_error = (60.0, 0.01)
inlier_score, outlier_prob = (5.0, 0.00001)
color_multiplier, depth_multiplier = (10000.0, 500.0)
model_args = b3d.ModelArgs(
    color_error,
    depth_error,
    inlier_score,
    outlier_prob,
    color_multiplier,
    depth_multiplier,
)

key = jax.random.PRNGKey(1000)
renderer = b3d.Renderer(image_width, image_height, fx, fy, cx, cy, 0.01, 10.0)

model = b3d.model_multiobject_gl_factory(renderer, b3d.rgbd_sensor_model)



importance_jit = jax.jit(model.importance)
key = jax.random.PRNGKey(110)



T = 0
rgb = rgbs[T]
depth = depths[T]
point_cloud = b3d.xyz_from_depth(depth, fx, fy, cx, cy).reshape(-1, 3)

gt_trace, _ = model.importance(
    jax.random.PRNGKey(0),
    genjax.choice_map(
        {
            "camera_pose": Pose.identity(),
            "object_pose_0": Pose(data["object_positions"][T], data["object_quaternions"][T]),
            "object_0": 0,
            "observed_rgb_depth": (rgb, depth),
        }
    ),
    (jnp.arange(1), model_args, object_library),
)
b3d.rerun_visualize_trace_t(gt_trace, 0)

vertex_colors = object_library.attributes
rgb_object_samples = vertex_colors[
    jax.random.choice(jax.random.PRNGKey(0), jnp.arange(len(vertex_colors)), (10,))
]
distances = jnp.abs(rgb[..., None] - rgb_object_samples.T).sum([-1, -2])
# rr.log("image/distances", rr.DepthImage(distances))
# rr.log("img", rr.Image(rgb))

object_center_hypothesis = point_cloud[distances.argmin()]



key = jax.random.split(key, 2)[-1]
trace, _ = model.importance(
    jax.random.PRNGKey(0),
    genjax.choice_map(
        {
            "camera_pose": Pose.identity(),
            "object_pose_0": Pose.sample_gaussian_vmf_pose(
                key, Pose.from_translation(object_center_hypothesis), 0.001, 0.01
            ),
            "object_0": 0,
            "observed_rgb_depth": (rgb, depth),
        }
    ),
    (jnp.arange(1), model_args, object_library),
)
b3d.rerun_visualize_trace_t(trace, 0)


params = jnp.array([0.02, 1.0])
skips = 0
for i in range(30):
    key = jax.random.split(key, 2)[-1]
    (
        trace2,
        key,
    ) = b3d.gvmf_and_sample(
        trace, key, params[0], params[1], genjax.Pytree.const("object_pose_0"), 10000
    )
    if trace2.get_score() > trace.get_score():
        trace = trace2
        # b3d.rerun_visualize_trace_t(trace, 0)
    else:
        params = jnp.array([params[0] * 0.5, params[1] * 2.0])
        skips += 1
        print(f"shrinking")
        if skips > 5:
            print(f"skip {i}")
            break

b3d.rerun_visualize_trace_t(trace, 0)


trace_after_gvmf = trace
key = jax.random.split(key, 2)[-1]

delta_cps = jnp.stack(
    jnp.meshgrid(
        jnp.linspace(-0.02, 0.02, 31),
        jnp.linspace(-0.02, 0.02, 31),
        jnp.linspace(-jnp.pi, jnp.pi, 71),
    ),
    axis=-1,
).reshape(-1, 3)
cp_delta_poses = jax.vmap(b3d.contact_parameters_to_pose)(delta_cps)

test_poses = trace["object_pose_0"] @ cp_delta_poses
test_poses_batches = test_poses.split(10)

scores = jnp.concatenate(
    [
        b3d.enumerate_choices_get_scores_jit(
            gt_trace, key, genjax.Pytree.const(["object_pose_0"]), poses
        )
        for poses in test_poses_batches
    ]
)
samples = jax.random.categorical(key, scores, shape=(50,))

samples_deg_range = jnp.rad2deg(
    (
        jnp.max(delta_cps[samples], axis=0)
        - jnp.min(delta_cps[samples], axis=0)
    )[2]
)

trace = b3d.update_choices_jit(
    trace,
    key,
    genjax.Pytree.const(["object_pose_0"]),
    test_poses[samples[0]],
)
b3d.rerun_visualize_trace_t(trace, 0)


print("Sampled Angle Range:", samples_deg_range)


for t in range(len(samples)):
    trace = b3d.update_choices_jit(
        trace,
        key,
        genjax.Pytree.const(["object_pose_0"]),
        test_poses[samples[t]],
    )
    b3d.rerun_visualize_trace_t(trace, t)

b3d.rerun_visualize_trace_t(trace_after_gvmf, 1)