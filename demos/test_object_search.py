#!/usr/bin/env python
import jax.numpy as jnp
import jax
import numpy as np
import os
import b3d
from b3d import Pose
import genjax
import rerun as rr
from tqdm import tqdm

PORT = 8812
rr.init("mug sm2c inference")
rr.connect(addr=f"127.0.0.1:{PORT}")


# Load date
path = os.path.join(
    b3d.get_root_path(),
    "assets/shared_data_bucket/input_data/object_search.r3d.video_input.npz",
)
video_input = b3d.io.VideoInput.load(path)

# Get intrinsics
image_width, image_height, fx, fy, cx, cy, near, far = np.array(
    video_input.camera_intrinsics_depth
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

# Get RGBS and Depth
rgbs = video_input.rgb[::4] / 255.0
xyzs = video_input.xyz[::4]

# Resize rgbs to be same size as depth.
rgbs_resized = jnp.clip(
    jax.vmap(jax.image.resize, in_axes=(0, None, None))(
        rgbs, (video_input.xyz.shape[1], video_input.xyz.shape[2], 3), "linear"
    ),
    0.0,
    1.0,
)

num_layers = 2048
renderer = b3d.Renderer(image_width, image_height, fx, fy, cx, cy, near, far)
model = b3d.model_multiobject_gl_factory(renderer)
importance_jit = jax.jit(model.importance)
update_jit = jax.jit(model.update)

# Arguments of the generative model.
# These control the inlier / outlier decision boundary for color error and depth error.
color_error, depth_error = (60.0, 0.02)
inlier_score, outlier_prob = (5.0, 0.01)
color_multiplier, depth_multiplier = (10000.0, 500.0)
model_args = b3d.ModelArgs(
    color_error,
    depth_error,
    inlier_score,
    outlier_prob,
    color_multiplier,
    depth_multiplier,
)


# Defines the enumeration schedule.
key = jax.random.PRNGKey(0)
# Gridding on translation only.
translation_deltas = Pose.concatenate_poses(
    [
        jax.vmap(lambda p: Pose.from_translation(p))(
            jnp.stack(
                jnp.meshgrid(
                    jnp.linspace(-0.01, 0.01, 11),
                    jnp.linspace(-0.01, 0.01, 11),
                    jnp.linspace(-0.01, 0.01, 11),
                ),
                axis=-1,
            ).reshape(-1, 3)
        ),
        Pose.identity()[None, ...],
    ]
)
# Sample orientations from a VMF to define a "grid" over orientations.
rotation_deltas = Pose.concatenate_poses(
    [
        jax.vmap(Pose.sample_gaussian_vmf_pose, in_axes=(0, None, None, None))(
            jax.random.split(jax.random.PRNGKey(0), 11 * 11 * 11),
            Pose.identity(),
            0.00001,
            1000.0,
        ),
        Pose.identity()[None, ...],
    ]
)
all_deltas = Pose.stack_poses([translation_deltas, rotation_deltas])

# Make empty library
object_library = b3d.MeshLibrary.make_empty_library()

# Take point cloud at frame 0
point_cloud = jax.image.resize(
    xyzs[0], (xyzs[0].shape[0], xyzs[0].shape[1], 3), "linear"
).reshape(-1, 3)
colors = jax.image.resize(
    rgbs_resized[0], (xyzs[0].shape[0], xyzs[0].shape[1], 3), "linear"
).reshape(-1, 3)

rr.log(
    "xyz/",
    rr.Points3D(point_cloud.reshape(-1, 3), colors=(colors * 255).astype(jnp.uint8)),
)
table_pose, table_dims = b3d.Pose.fit_table_plane(point_cloud, 0.01, 0.02, 1000, 1000)
b3d.rr_log_pose("table_pose", table_pose)

# `make_mesh_from_point_cloud_and_resolution` takes a 3D positions, colors, and sizes of the boxes that we want
# to place at each position and create a mesh
vertices, faces, vertex_colors, face_colors = (
    b3d.make_mesh_from_point_cloud_and_resolution(
        point_cloud,
        colors,
        point_cloud[:, 2]
        / fx
        * 3.0,  # This is scaling the size of the box to correspond to the effective size of the pixel in 3D. It really should be multiplied by 2.
        # and the 6 makes it larger
    )
)
object_library.add_object(vertices, faces, vertex_colors)

vertices, faces, vertex_colors, face_colors = (
    b3d.make_mesh_from_point_cloud_and_resolution(
        jnp.zeros((1, 3)),
        jnp.array([[0.0, 1.0, 0.0]]),
        jnp.array([0.05]),
    )
)
object_library.add_object(vertices, faces, vertex_colors)


# Defines the enumeration schedule.
key = jax.random.PRNGKey(0)
# Gridding on translation only.
translation_deltas = Pose.concatenate_poses(
    [
        jax.vmap(lambda p: Pose.from_translation(p))(
            jnp.stack(
                jnp.meshgrid(
                    jnp.linspace(-0.01, 0.01, 11),
                    jnp.linspace(-0.01, 0.01, 11),
                    jnp.linspace(-0.01, 0.01, 11),
                ),
                axis=-1,
            ).reshape(-1, 3)
        ),
        Pose.identity()[None, ...],
    ]
)
# Sample orientations from a VMF to define a "grid" over orientations.
rotation_deltas = Pose.concatenate_poses(
    [
        jax.vmap(Pose.sample_gaussian_vmf_pose, in_axes=(0, None, None, None))(
            jax.random.split(jax.random.PRNGKey(0), 11 * 11 * 11),
            Pose.identity(),
            0.00001,
            1000.0,
        ),
        Pose.identity()[None, ...],
    ]
)
all_deltas = Pose.stack_poses([translation_deltas, rotation_deltas])


# Initial trace for timestep 0
START_T = 0
trace, _ = importance_jit(
    jax.random.PRNGKey(0),
    genjax.choice_map(
        dict(
            [
                ("camera_pose", Pose.identity()),
                ("object_pose_0", Pose.identity()),
                ("object_0", 0),
                (
                    "observed_rgb_depth",
                    (rgbs_resized[START_T], xyzs[START_T, ..., 2]),
                ),
                ("object_pose_1", Pose.from_translation(jnp.array([0.0, 0.0, 0.5]))),
                ("object_1", -1),
            ]
        )
    ),
    (jnp.arange(2), model_args, object_library),
)
poses = []

for T_observed_image in tqdm(range(len(rgbs))):
    # Constrain on new RGB and Depth data.
    trace = b3d.update_choices_jit(
        trace,
        key,
        genjax.Pytree.const(["observed_rgb_depth"]),
        (rgbs_resized[T_observed_image], xyzs[T_observed_image, ..., 2]),
    )
    trace, key = b3d.enumerate_and_select_best_move_jit(
        trace, genjax.Pytree.const(["camera_pose"]), key, all_deltas
    )
    b3d.rerun_visualize_trace_t(trace, T_observed_image)
    poses.append(trace["camera_pose"])


trace = b3d.update_choices_jit(trace, key, genjax.Pytree.const(["object_1"]), 1)


test_poses = jax.vmap(
    lambda i: table_pose @ Pose.from_translation(jnp.array([i[0], i[1], 0.0]))
)(
    jnp.stack(
        jnp.meshgrid(
            jnp.linspace(-0.3, 0.3, 50), jnp.linspace(-0.3, 0.3, 50), indexing="ij"
        ),
        axis=-1,
    ).reshape(-1, 2)
)
T_observed_image = 0


for T_observed_image in tqdm(range(len(rgbs))):
    # Constrain on new RGB and Depth data.
    trace = b3d.update_choices_jit(
        trace,
        key,
        genjax.Pytree.const(["observed_rgb_depth", "camera_pose", "object_1"]),
        (rgbs_resized[T_observed_image], xyzs[T_observed_image, ..., 2]),
        poses[T_observed_image],
        1,
    )
    scores = b3d.enumerate_choices_get_scores_jit(
        trace,
        key,
        genjax.Pytree.const(["object_pose_1"]),
        test_poses,
    )
    samples = jax.random.categorical(key, scores, shape=(300,))

    trace = b3d.update_choices_jit(trace, key, genjax.Pytree.const(["object_1"]), -1)
    b3d.rerun_visualize_trace_t(trace, T_observed_image, modes=["rgb", "3d"])

    t = T_observed_image
    rendered_images = jax.vmap(
        lambda pose: b3d.update_choices(
            trace,
            key,
            genjax.Pytree.const(["object_pose_1", "object_0", "object_1"]),
            pose,
            -1,
            1,
        ).get_retval()[0][1]
    )(test_poses[samples])
    rendered_images = rendered_images.max(0)
    rr.log("/image/hidden_object", rr.Image(rendered_images))
    test_poses = test_poses[samples]


trace = b3d.update_choices_jit(
    trace, key, genjax.Pytree.const(["object_1", "object_pose_1"]), 1, test_poses[0]
)
b3d.rerun_visualize_trace_t(trace, t, modes=["rgb", "3d"])
