#!/usr/bin/env python
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np
import os
import trimesh
import b3d
from jax.scipy.spatial.transform import Rotation as Rot
from b3d import Pose
import genjax
import rerun as rr
from tqdm import tqdm
import fire


def test_demo():
    rr.init("demo")
    rr.connect("127.0.0.1:8812")

    # Load date
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

    num_layers = 2048
    renderer = b3d.Renderer(image_width, image_height, fx, fy, cx, cy, near, far)
    model = b3d.model_multiobject_gl_factory(renderer)
    importance_jit = jax.jit(model.importance)
    update_jit = jax.jit(model.update)

    # Arguments of the generative model.
    # These control the inlier / outlier decision boundary for color error and depth error.
    color_error, depth_error = (60.0, 0.02)
    inlier_score, outlier_prob = (5.0, 0.001)
    color_multiplier, depth_multiplier = (5000.0, 500.0)
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
                        jnp.linspace(-0.02, 0.02, 11),
                        jnp.linspace(-0.02, 0.02, 11),
                        jnp.linspace(-0.02, 0.02, 11),
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
                0.001,
                1000.0,
            ),
            Pose.identity()[None, ...],
        ]
    )
    all_deltas = Pose.stack_poses([translation_deltas, rotation_deltas])

    original_camera_pose = b3d.Pose(
        video_input.camera_positions[0],
        video_input.camera_quaternions[0]
    )

    # Make empty library
    object_library = b3d.MeshLibrary.make_empty_library()

    # Take point cloud at frame 0
    point_cloud = jax.image.resize(
        xyzs[0], (xyzs[0].shape[0] // 3, xyzs[0].shape[1] // 3, 3), "linear"
    ).reshape(-1, 3)
    colors = jax.image.resize(
        rgbs_resized[0], (xyzs[0].shape[0] // 3, xyzs[0].shape[1] // 3, 3), "linear"
    ).reshape(-1, 3)

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
    object_pose =  Pose.from_translation(vertices.mean(0))
    vertices = object_pose.inverse().apply(vertices)
    object_library.add_object(vertices, faces, vertex_colors)
    REAQUISITION_TS = [0, 95, 222, 355, len(rgbs_resized)]

    trace, _ = importance_jit(
        jax.random.PRNGKey(0),
        genjax.choice_map(
            dict(
                [
                    ("camera_pose", original_camera_pose),
                    ("object_pose_0", original_camera_pose @ object_pose),
                    ("object_0", 0),
                    (
                        "observed_rgb_depth",
                        (rgbs_resized[0], xyzs[0, ..., 2]),
                    ),
                ]
            )
        ),
        (jnp.arange(1), model_args, object_library),
    )
    b3d.rerun_visualize_trace_t(trace, 0)

    num_objects = 0
    data = []
    for acquisition_phase in range(len(REAQUISITION_TS)-1):

        # Visualize trace
        ACQUISITION_T = REAQUISITION_TS[acquisition_phase + 1]
        for T_observed_image in tqdm(range(REAQUISITION_TS[acquisition_phase], ACQUISITION_T)):
            # Constrain on new RGB and Depth data.
            trace = b3d.update_choices_jit(
                trace,
                key,
                genjax.Pytree.const(["observed_rgb_depth"]),
                (rgbs_resized[T_observed_image], xyzs[T_observed_image, ..., 2]),
            )
            trace, key = b3d.enumerate_and_select_best_move(
                trace, genjax.Pytree.const(["camera_pose"]), key, all_deltas
            )
            for i in range(num_objects):
                trace, key = b3d.enumerate_and_select_best_move(
                    trace, genjax.Pytree.const([f"object_pose_{i+1}"]), key, all_deltas
                )
            b3d.rerun_visualize_trace_t(trace, T_observed_image)
            data.append((b3d.get_poses_from_trace(trace), trace["camera_pose"]))

        # Outliers are AND of the RGB and Depth outlier masks
        inliers, color_inliers, depth_inliers, outliers, undecided, valid_data_mask = (
            b3d.get_rgb_depth_inliers_from_trace(trace)
        )
        outlier_mask = outliers
        rr.log("outliers", rr.Image(jnp.tile((outlier_mask * 1.0)[..., None], (1, 1, 3))))

        # Get the point cloud corresponding to the outliers
        rgb, depth = trace["observed_rgb_depth"]
        point_cloud = b3d.xyz_from_depth(depth, fx, fy, cx, cy)[outlier_mask]
        point_cloud_colors = rgb[outlier_mask]

        # Segment the outlier cloud.
        assignment = b3d.segment_point_cloud(point_cloud)

        # Only keep the largers cluster in the outlier cloud.
        point_cloud = point_cloud.reshape(-1, 3)[assignment == 0]
        point_cloud_colors = point_cloud_colors.reshape(-1, 3)[assignment == 0]

        # Create new mesh.
        vertices, faces, vertex_colors, face_colors = (
            b3d.make_mesh_from_point_cloud_and_resolution(
                point_cloud, point_cloud_colors, point_cloud[:, 2] / fx * 2.0
            )
        )
        object_pose = Pose.from_translation(vertices.mean(0))
        vertices = object_pose.inverse().apply(vertices)
        object_library.add_object(vertices, faces, vertex_colors)

        single_object_trace = trace

        trace = single_object_trace

        num_objects += 1

        # from IPython import embed; embed()

        trace, _ = importance_jit(
            jax.random.PRNGKey(0),
            genjax.choice_map(
                dict(
                    [
                        ("camera_pose", trace["camera_pose"]),
                        *[ (f"object_pose_{i}", trace[f"object_pose_{i}"]) for i in range(num_objects)],
                        *[ (f"object_{i}", trace[f"object_{i}"]) for i in range(num_objects)],
                        (f"object_pose_{num_objects}", trace["camera_pose"] @ object_pose),
                        (f"object_{num_objects}", num_objects),
                        (
                            "observed_rgb_depth",
                            trace["observed_rgb_depth"]
                        ),
                    ]
                )
            ),
            (jnp.arange(num_objects + 1), model_args, object_library),
        )
        # Visualize trace
        b3d.rerun_visualize_trace_t(trace, ACQUISITION_T)


    import pickle
    with open('demo_data.dat', 'wb') as f:
        pickle.dump((data, object_library), f)

    data2 = pickle.load(open('demo_data.dat', 'rb'))

if __name__ == "__main__":
    fire.Fire(test_demo)
