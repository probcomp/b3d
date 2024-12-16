import os
from functools import partial

import b3d
import b3d.bayes3d as bayes3d
import genjax
import jax
import jax.numpy as jnp
import numpy as np
import rerun as rr
from b3d import Pose
from tqdm import tqdm

# Rerun setup
PORT = 8812
rr.init("online_learning")
rr.connect(addr=f"127.0.0.1:{PORT}")

# Load date
path = os.path.join(
    b3d.get_assets_path(),
    #  "shared_data_bucket/input_data/orange_mug_pan_around_and_pickup.r3d.video_input.npz")
    # "shared_data_bucket/input_data/shout_on_desk.r3d.video_input.npz")
    "shared_data_bucket/input_data/desk_ramen2_spray1.r3d.video_input.npz",
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
rgbs = video_input.rgb[::3] / 255.0
xyzs = video_input.xyz[::3]

# Resize rgbs to be same size as depth.
rgbs_resized = jnp.clip(
    jax.vmap(jax.image.resize, in_axes=(0, None, None))(
        rgbs, (video_input.xyz.shape[1], video_input.xyz.shape[2], 3), "linear"
    ),
    0.0,
    1.0,
)

# Arguments of the generative model.
# These control the inlier / outlier decision boundary for color error and depth error.
color_error, depth_error = (60.0, 0.01)
inlier_score, outlier_prob = (5.0, 0.00001)
color_multiplier, depth_multiplier = (10000.0, 500.0)
model_args = bayes3d.ModelArgs(
    color_error,
    depth_error,
    inlier_score,
    outlier_prob,
    color_multiplier,
    depth_multiplier,
)

# Make empty library
object_library = b3d.MeshLibrary.make_empty_library()

# Creating initial background mesh

# Take point cloud at frame 0
point_cloud = xyzs[0].reshape(-1, 3)
# Take RGB data at frame 0
colors = rgbs_resized[0].reshape(-1, 3)

# Select a subset of those points
sub = jax.random.choice(
    jax.random.PRNGKey(0),
    jnp.arange(len(point_cloud)),
    (len(point_cloud) // 6,),
    replace=False,
)
# Instead of subsampling randomly, it would make more sense to scale down the image before unprojecting.
point_cloud = point_cloud[sub]
colors = colors[sub]

# `make_mesh_from_point_cloud_and_resolution` takes a 3D positions, colors, and sizes of the boxes that we want
# to place at each position and create a mesh
vertices, faces, vertex_colors, face_colors = (
    b3d.make_mesh_from_point_cloud_and_resolution(
        point_cloud,
        colors,
        point_cloud[:, 2]
        / fx
        * 6.0,  # This is scaling the size of the box to correspond to the effective size of the pixel in 3D. It really should be multiplied by 2.
        # and the 6 makes it larger
    )
)

# Add background mesh object to the library
object_library.add_object(vertices, faces, vertex_colors)

# Creates renderer and generative model.
renderer = b3d.Renderer(image_width, image_height, fx, fy, cx, cy, near, far)
model = bayes3d.model_multiobject_gl_factory(renderer)

# Arguments of the generative model.
# These control the inlier / outlier decision boundary for color error and depth error.
color_error, depth_error = (jnp.float32(30.0), jnp.float32(0.02))
# TODO: explain
inlier_score, outlier_prob = (jnp.float32(5.0), jnp.float32(0.001))
# TODO: explain
color_multiplier, depth_multiplier = (jnp.float32(3000.0), jnp.float32(3000.0))

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

# Enumerative proposal function


@partial(jax.jit, static_argnames=["addressses"])
def enumerative_proposal(trace, addressses, key, all_deltas):
    addr = addressses.const[0]
    current_pose = trace.get_choices()[addr]
    for i in range(len(all_deltas)):
        test_poses = current_pose @ all_deltas[i]
        potential_scores = b3d.enumerate_choices_get_scores(
            trace, addressses, test_poses
        )
        current_pose = test_poses[potential_scores.argmax()]
    trace = b3d.update_choices(trace, addressses, current_pose)
    return trace, key


# We have fixed which iterations on which we will pause to acquire a new object.
REAQUISITION_TS = [0, 95, 222, 355, len(rgbs_resized)]

importance_jit = jax.jit(model.importance)
update_jit = jax.jit(model.update)

# Initial trace for timestep 0
START_T = 0
trace, _ = importance_jit(
    jax.random.PRNGKey(0),
    genjax.ChoiceMap.d(
        dict(
            [
                ("camera_pose", Pose.identity()),
                ("object_pose_0", Pose.identity()),
                ("object_0", 0),
                ("object_1", -1),  # For all the unused objects, set their ID to -1
                ("object_2", -1),
                ("object_3", -1),
                ("observed_rgb_depth", (rgbs_resized[START_T], xyzs[START_T, ..., 2])),
            ]
        )
    ),
    (jnp.arange(4), model_args, object_library),
)
# Visualize trace
bayes3d.rerun_visualize_trace_t(trace, 0)
key = jax.random.PRNGKey(0)

inference_data_over_time = []
for reaquisition_phase in range(len(REAQUISITION_TS) - 1):
    for T_observed_image in tqdm(
        range(
            REAQUISITION_TS[reaquisition_phase], REAQUISITION_TS[reaquisition_phase + 1]
        )
    ):
        # Constrain on new RGB and Depth data.
        trace = b3d.update_choices(
            trace,
            genjax.Pytree.const(["observed_rgb_depth"]),
            (rgbs_resized[T_observed_image], xyzs[T_observed_image, ..., 2]),
        )
        # Enumerate, score, and update  camera pose
        trace, key = enumerative_proposal(
            trace, genjax.Pytree.const(("camera_pose",)), key, all_deltas
        )
        for i in range(1, len(object_library.ranges)):
            # Enumerate, score, update each objects pose
            trace, key = enumerative_proposal(
                trace, genjax.Pytree.const((f"object_pose_{i}",)), key, all_deltas
            )
        bayes3d.rerun_visualize_trace_t(trace, T_observed_image)
        inference_data_over_time.append(
            (
                b3d.get_poses_from_trace(trace),
                b3d.get_object_ids_from_trace(trace),
                trace["camera_pose"],
                T_observed_image,
            )
        )

    # Now we acquire a new object.

    # Compute RGB and Depth outliers
    inliers, color_inliers, depth_inliers, outliers, undecided, valid_data_mask = (
        b3d.get_rgb_depth_inliers_from_trace(trace)
    )
    rr.set_time_sequence("frame", T_observed_image)
    # rr.log(
    #     "/rgb/rgb_outliers",
    #     rr.Image(jnp.tile((rgb_outliers * 1.0)[..., None], (1, 1, 3))),
    # )
    # rr.log(
    #     "/rgb/depth_outliers",
    #     rr.Image(jnp.tile((depth_outliers * 1.0)[..., None], (1, 1, 3))),
    # )

    # Outliers are AND of the RGB and Depth outlier masks
    outlier_mask = outliers
    rr.log("outliers", rr.Image(jnp.tile((outlier_mask * 1.0)[..., None], (1, 1, 3))))

    # Get the point cloud corresponding to the outliers
    point_cloud = b3d.xyz_from_depth(trace["observed_rgb_depth"][1], fx, fy, cx, cy)[
        outlier_mask
    ]
    point_cloud_colors = trace["observed_rgb_depth"][0][outlier_mask]

    # Segment the outlier cloud.
    assignment = b3d.segment_point_cloud(point_cloud)

    # Only keep the largers cluster in the outlier cloud.
    point_cloud = point_cloud.reshape(-1, 3)[assignment == 0]
    point_cloud_colors = point_cloud_colors.reshape(-1, 3)[assignment == 0]

    # Subsample to reduce the number of triangles.
    sub = jax.random.choice(
        jax.random.PRNGKey(0),
        jnp.arange(len(point_cloud)),
        (len(point_cloud) // 4,),
        replace=False,
    )
    point_cloud = point_cloud[sub]
    colors = point_cloud_colors[sub]

    # Create new mesh.
    vertices, faces, vertex_colors, face_colors = (
        b3d.make_mesh_from_point_cloud_and_resolution(
            point_cloud, colors, point_cloud[:, 2] / fx * 2.0
        )
    )

    # Choose the nominal pose of the newly contructed object to be place at the mean of the 3D points.
    object_pose = Pose.from_translation(vertices.mean(0))
    vertices = object_pose.inverse().apply(vertices)
    object_library.add_object(vertices, faces, vertex_colors)

    REAQUISITION_T = REAQUISITION_TS[reaquisition_phase + 1] - 1
    next_object_id = len(object_library.ranges) - 1
    trace = trace.update(
        key,
        genjax.ChoiceMap.d(
            {
                f"object_{next_object_id}": next_object_id,  # Add identity of new object to trace.
                f"object_pose_{next_object_id}": trace["camera_pose"]
                @ object_pose,  # Add pose of new object to trace.
                "observed_rgb_depth": (
                    rgbs_resized[REAQUISITION_T],
                    xyzs[REAQUISITION_T, ..., 2],
                ),
            }
        ),
        # genjax.Diff.tree_diff_unknown_change((jnp.arange(2), *trace.get_args()[1:]))
        genjax.Diff.tree_diff_unknown_change(
            (jnp.arange(4), model_args, object_library)
        ),
    )[0]
    bayes3d.rerun_visualize_trace_t(trace, REAQUISITION_T)
    inference_data_over_time.append(
        (
            b3d.get_poses_from_trace(trace),
            b3d.get_object_ids_from_trace(trace),
            trace["camera_pose"],
            T_observed_image,
        )
    )


for i in tqdm(range(len(inference_data_over_time))):
    poses, object_ids, camera_pose, t = inference_data_over_time[i]
    trace = update_jit(
        key,
        trace,
        genjax.choice_map(
            dict(
                [
                    *[(f"object_pose_{i}", poses[i]) for i in range(len(poses))],
                    *[(f"object_{i}", object_ids[i]) for i in range(len(object_ids))],
                    ("camera_pose", camera_pose),
                    ("observed_rgb_depth", (rgbs_resized[t], xyzs[t, ..., 2])),
                ]
            )
        ),
        genjax.Diff.tree_diff_unknown_change(
            (jnp.arange(4), model_args, object_library)
        ),
    )[0]
    bayes3d.rerun_visualize_trace_t(trace, t)
    rr.set_time_sequence("frame", t)

    rgb_inliers, rgb_outliers = b3d.get_rgb_inlier_outlier_from_trace(trace)
    depth_inliers, depth_outliers = b3d.get_depth_inlier_outlier_from_trace(trace)

    outlier_mask = jnp.logical_and(rgb_outliers, depth_outliers)

    rr.log("outliers", rr.Image(jnp.tile((outlier_mask * 1.0)[..., None], (1, 1, 3))))
