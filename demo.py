#!/usr/bin/env python
import fire


def test_demo():
    import os

    import genjax
    import jax
    import jax.numpy as jnp
    import numpy as np
    import rerun as rr
    from genjax import Pytree
    from tqdm import tqdm

    import b3d
    import b3d.bayes3d as bayes3d
    from b3d import Mesh, Pose

    rr.init("demo")
    rr.serve()
    # rr.connect("127.0.0.1:8812")

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
    rgbds = jnp.concatenate([rgbs_resized, xyzs[..., 2:3]], axis=-1)

    scaling_factor = 1.0
    renderer = b3d.renderer.renderer_original.RendererOriginal(
        image_width * scaling_factor,
        image_height * scaling_factor,
        fx * scaling_factor,
        fy * scaling_factor,
        cx * scaling_factor,
        cy * scaling_factor,
        0.01,
        2.0,
    )

    # Defines the enumeration schedule.
    key = jax.random.PRNGKey(0)
    # Gridding on translation only.
    translation_deltas = Pose.concatenate_poses(
        [
            jax.vmap(lambda p: Pose.from_translation(p))(
                jnp.stack(
                    jnp.meshgrid(
                        jnp.linspace(-0.01, 0.01, 9),
                        jnp.linspace(-0.01, 0.01, 9),
                        jnp.linspace(-0.01, 0.01, 9),
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
                jax.random.split(jax.random.PRNGKey(0), 9 * 9 * 9),
                Pose.identity(),
                0.00001,
                1000.0,
            ),
            Pose.identity()[None, ...],
        ]
    )
    all_deltas = Pose.stack_poses([translation_deltas, rotation_deltas])

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
    background_mesh = Mesh(vertices, faces, vertex_colors)

    import b3d.chisight.dense.dense_model
    import b3d.chisight.dense.likelihoods.laplace_likelihood

    b3d.reload(b3d.chisight.dense.dense_model)
    b3d.reload(b3d.chisight.dense.likelihoods.laplace_likelihood)
    likelihood_func = b3d.chisight.dense.likelihoods.laplace_likelihood.likelihood_func
    model, viz_trace, info_from_trace = (
        b3d.chisight.dense.dense_model.make_dense_multiobject_model(
            renderer, likelihood_func
        )
    )
    importance_jit = jax.jit(model.importance)

    likelihood_args = {
        "fx": renderer.fx,
        "fy": renderer.fy,
        "cx": renderer.cx,
        "cy": renderer.cy,
        "image_width": Pytree.const(renderer.width),
        "image_height": Pytree.const(renderer.height),
    }

    # Initial trace for timestep 0
    START_T = 0
    choicemap = genjax.ChoiceMap.d(
        {
            "rgbd": rgbds[START_T],
            "camera_pose": Pose.identity(),
            "object_pose_0": Pose.identity(),
            "object_0": 0,
            "depth_noise_variance": 0.005,
            "color_noise_variance": 0.05,
            "outlier_probability": 0.1,
        }
    )

    trace, _ = importance_jit(
        jax.random.PRNGKey(0),
        choicemap,
        (
            {
                "num_objects": Pytree.const(1),
                "meshes": [background_mesh],
                "likelihood_args": likelihood_args,
            },
        ),
    )

    # Visualize trace
    viz_trace(trace, 0)

    ACQUISITION_T = 90
    for T_observed_image in tqdm(range(ACQUISITION_T)):
        # Constrain on new RGB and Depth data.
        trace = b3d.update_choices(
            trace,
            Pytree.const(["rgbd"]),
            rgbds[T_observed_image],
        )
        trace, key = bayes3d.enumerate_and_select_best_move(
            trace, Pytree.const(("camera_pose",)), key, all_deltas
        )
        viz_trace(trace, T_observed_image)

    # Outliers are AND of the RGB and Depth outlier masks
    rgbd = trace.get_choices()["rgbd"]
    latent_rgbd = info_from_trace(trace)["latent_rgbd"]

    mismatch_depth = jnp.abs(rgbd[..., 3] - latent_rgbd[..., 3]) > 0.01
    mismatch_rgbd = (
        jnp.linalg.norm(rgbd[..., :3] - latent_rgbd[..., :3], axis=-1) > 0.05
    )
    mismatch_mask = mismatch_depth * mismatch_rgbd
    b3d.rr_log_depth(1.0 * mismatch_mask, "mismatch")

    point_cloud = b3d.xyz_from_depth(rgbd[..., 3], fx, fy, cx, cy)[mismatch_mask]
    point_cloud_colors = rgbd[mismatch_mask, :3]

    # Segment the outlier cloud.
    assignment = b3d.segment_point_cloud(point_cloud)

    # Only keep the largers cluster in the outlier cloud.
    point_cloud = point_cloud.reshape(-1, 3)[assignment == 0]
    point_cloud_colors = point_cloud_colors.reshape(-1, 3)[assignment == 0]

    mask = jax.random.choice(
        key, len(point_cloud), (len(point_cloud) // 2,), replace=False
    )
    point_cloud = point_cloud[mask]
    point_cloud_colors = point_cloud_colors[mask]

    # Create new mesh.
    vertices, faces, vertex_colors, _face_colors = (
        b3d.make_mesh_from_point_cloud_and_resolution(
            point_cloud, point_cloud_colors, point_cloud[:, 2] / fx * 2.0
        )
    )
    object_pose = Pose.from_translation(vertices.mean(0))
    vertices = object_pose.inverse().apply(vertices)
    object_mesh = Mesh(vertices, faces, vertex_colors)
    object_mesh.rr_visualize("object_mesh")

    trace_post_initial_tracking = trace

    new_choices = (
        genjax.ChoiceMap.d(
            {
                "object_pose_1": trace.get_choices()["camera_pose"] @ object_pose,
                "object_1": 1,
            }
        )
        ^ trace_post_initial_tracking.get_choices()
    )

    trace_post_acquisition, _ = importance_jit(
        jax.random.PRNGKey(0),
        new_choices,
        (
            {
                "num_objects": Pytree.const(2),
                "meshes": [background_mesh, object_mesh],
                "likelihood_args": likelihood_args,
            },
        ),
    )

    # Visualize trace
    viz_trace(trace_post_acquisition, ACQUISITION_T)
    trace = trace_post_acquisition

    FINAL_T = len(xyzs)
    for T_observed_image in tqdm(range(ACQUISITION_T, FINAL_T)):
        # Constrain on new RGB and Depth data.
        trace = b3d.update_choices(
            trace,
            Pytree.const(["rgbd"]),
            rgbds[T_observed_image],
        )
        trace, key = bayes3d.enumerate_and_select_best_move(
            trace, Pytree.const(("camera_pose",)), key, all_deltas
        )
        trace, key = bayes3d.enumerate_and_select_best_move(
            trace, Pytree.const(("object_pose_1",)), key, all_deltas
        )

        viz_trace(trace, T_observed_image)


if __name__ == "__main__":
    fire.Fire(test_demo)
