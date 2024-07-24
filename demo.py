#!/usr/bin/env python
import os

import b3d
import b3d.bayes3d as bayes3d
import fire
import genjax
import jax
import jax.numpy as jnp
import numpy as np
import rerun as rr
from b3d import Pose
from tqdm import tqdm


def test_demo():
    rr.init("demo")
    rr.connect("127.0.0.1:8812")

    # Load date
    path = os.path.join(
        b3d.get_root_path(),
        "assets/shared_data_bucket/input_data/shout_on_desk.r3d.video_input.npz",
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

    renderer = b3d.Renderer(image_width, image_height, fx, fy, cx, cy, near, far)
    model = bayes3d.model_multiobject_gl_factory(renderer)
    importance_jit = jax.jit(model.importance)

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
    object_library = bayes3d.MeshLibrary.make_empty_library()

    # Take point cloud at frame 0
    point_cloud = jax.image.resize(
        xyzs[0], (xyzs[0].shape[0] // 3, xyzs[0].shape[1] // 3, 3), "linear"
    ).reshape(-1, 3)
    colors = jax.image.resize(
        rgbs_resized[0], (xyzs[0].shape[0] // 3, xyzs[0].shape[1] // 3, 3), "linear"
    ).reshape(-1, 3)

    # `make_mesh_from_point_cloud_and_resolution` takes a 3D positions, colors, and sizes of the boxes that we want
    # to place at each position and create a mesh
    vertices, faces, vertex_colors, _face_colors = (
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
                    (
                        "observed_rgb_depth",
                        (rgbs_resized[START_T], xyzs[START_T, ..., 2]),
                    ),
                ]
            )
        ),
        (jnp.arange(1), model_args, object_library),
    )
    # Visualize trace
    bayes3d.rerun_visualize_trace_t(trace, 0)

    ACQUISITION_T = 90
    for T_observed_image in tqdm(range(ACQUISITION_T)):
        # Constrain on new RGB and Depth data.
        trace = b3d.update_choices_jit(
            trace,
            key,
            ("observed_rgb_depth",),
            (rgbs_resized[T_observed_image], xyzs[T_observed_image, ..., 2]),
        )
        trace, key = bayes3d.enumerate_and_select_best_move(
            trace, ("camera_pose",), key, all_deltas
        )
        bayes3d.rerun_visualize_trace_t(trace, T_observed_image)

    # Outliers are AND of the RGB and Depth outlier masks
    _inliers, _color_inliers, _depth_inliers, outliers, _undecided, _valid_data_mask = (
        bayes3d.get_rgb_depth_inliers_from_trace(trace)
    )
    outlier_mask = outliers
    rr.log("outliers", rr.Image(jnp.tile((outlier_mask * 1.0)[..., None], (1, 1, 3))))

    # Get the point cloud corresponding to the outliers
    rgb, depth = trace.get_choices()["observed_rgb_depth"]
    point_cloud = b3d.xyz_from_depth(depth, fx, fy, cx, cy)[outlier_mask]
    point_cloud_colors = rgb[outlier_mask]

    # Segment the outlier cloud.
    assignment = b3d.segment_point_cloud(point_cloud)

    # Only keep the largers cluster in the outlier cloud.
    point_cloud = point_cloud.reshape(-1, 3)[assignment == 0]
    point_cloud_colors = point_cloud_colors.reshape(-1, 3)[assignment == 0]

    # Create new mesh.
    vertices, faces, vertex_colors, _face_colors = (
        b3d.make_mesh_from_point_cloud_and_resolution(
            point_cloud, point_cloud_colors, point_cloud[:, 2] / fx * 2.0
        )
    )
    object_pose = Pose.from_translation(vertices.mean(0))
    vertices = object_pose.inverse().apply(vertices)
    object_library.add_object(vertices, faces, vertex_colors)

    single_object_trace = trace

    trace = single_object_trace

    trace, _ = importance_jit(
        jax.random.PRNGKey(0),
        genjax.ChoiceMap.d(
            dict(
                [
                    ("camera_pose", trace.get_choices()["camera_pose"]),
                    ("object_pose_0", trace.get_choices()["object_pose_0"]),
                    ("object_pose_1", trace.get_choices()["camera_pose"] @ object_pose),
                    ("object_0", 0),
                    ("object_1", 1),
                    (
                        "observed_rgb_depth",
                        (rgbs_resized[ACQUISITION_T], xyzs[ACQUISITION_T, ..., 2]),
                    ),
                ]
            )
        ),
        (jnp.arange(2), model_args, object_library),
    )
    # Visualize trace
    bayes3d.rerun_visualize_trace_t(trace, ACQUISITION_T)

    FINAL_T = len(xyzs)
    for T_observed_image in tqdm(range(ACQUISITION_T, FINAL_T)):
        # Constrain on new RGB and Depth data.
        trace = b3d.update_choices_jit(
            trace,
            key,
            ("observed_rgb_depth",),
            (rgbs_resized[T_observed_image], xyzs[T_observed_image, ..., 2]),
        )
        trace, key = bayes3d.enumerate_and_select_best_move(
            trace, ("camera_pose",), key, all_deltas
        )
        trace, key = bayes3d.enumerate_and_select_best_move(
            trace, ("object_pose_1",), key, all_deltas
        )
        bayes3d.rerun_visualize_trace_t(trace, T_observed_image)


if __name__ == "__main__":
    fire.Fire(test_demo)
