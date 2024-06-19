import rerun as rr
import genjax
import os
import numpy as np
import jax.numpy as jnp
import jax
from b3d import Pose
import b3d
import b3d.bayes3d as bayes3d
from tqdm import tqdm
import trimesh
import genjax



def test_renderer_full(renderer):
    PORT = 8812
    rr.init("mug smc inference")
    rr.connect(addr=f"127.0.0.1:{PORT}")

    for INPUT in ["real-occluded", "real-visible", "synthetic"]:
        print(f"Running with input {INPUT}")
        if INPUT == "synthetic":
            video_input = b3d.io.VideoInput.load(b3d.utils.get_root_path() / "assets/shared_data_bucket/datasets/posterior_uncertainty_mug_handle_w_0.02_video_input.npz")
            scaling_factor = 3
            T = 50
        elif INPUT == "real-occluded":
            video_input = b3d.io.VideoInput.load(
                os.path.join(
                    b3d.utils.get_root_path(),
                    "assets/shared_data_bucket/input_data/mug_handle_occluded.video_input.npz",
                    # "assets/shared_data_bucket/input_data/mug_handle_visible.video_input.npz"
                )
            )
            scaling_factor = 5
            T = 0
        elif INPUT == "real-visible":
            video_input = b3d.io.VideoInput.load(
                os.path.join(
                    b3d.utils.get_root_path(),
                    "assets/shared_data_bucket/input_data/mug_handle_visible.video_input.npz",
                )
            )
            scaling_factor = 5
            T = 0
        else:
            raise ValueError(f"Unknown input {INPUT}")

        image_width, image_height, fx, fy, cx, cy, near, far = (
            jnp.array(video_input.camera_intrinsics_depth) / scaling_factor
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

        _rgb = video_input.rgb[T].astype(jnp.float32) / 255.0
        _depth = video_input.xyz[T].astype(jnp.float32)[..., 2]
        rgb = jnp.clip(
            jax.image.resize(_rgb, (image_height, image_width, 3), "nearest"), 0.0, 1.0
        )
        depth = jax.image.resize(_depth, (image_height, image_width), "nearest")

        object_library = bayes3d.MeshLibrary.make_empty_library()
        mesh_path = os.path.join(
            b3d.utils.get_root_path(),
            "assets/shared_data_bucket/ycb_video_models/models/025_mug/textured_simple.obj",
        )
        object_library.add_trimesh(trimesh.load(mesh_path))


        color_error, depth_error = (60.0, 0.02)
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

        key = jax.random.PRNGKey(1000)
        if renderer is None:
            renderer = b3d.Renderer(image_width, image_height, fx, fy, cx, cy, 0.01, 10.0)
        else:
            renderer.set_intrinsics(image_width, image_height, fx, fy, cx, cy, 0.01, 10.0)
        model = bayes3d.model_multiobject_gl_factory(renderer, bayes3d.rgbd_sensor_model)



        importance_jit = jax.jit(model.importance)
        key = jax.random.PRNGKey(110)


        point_cloud = b3d.utils.xyz_from_depth(depth, fx, fy, cx, cy).reshape(-1, 3)

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
        bayes3d.rerun_visualize_trace_t(trace, 0)


        params = jnp.array([0.02, 1.0])
        skips = 0
        counter = 1
        for t in tqdm(range(30)):
            (
                trace2,
                key,
            ) = bayes3d.gvmf_and_sample(
                trace, key, params[0], params[1], genjax.Pytree.const("object_pose_0"), 10000
            )
            if trace2.get_score() > trace.get_score():
                trace = trace2
                bayes3d.rerun_visualize_trace_t(trace, counter)
                counter += 1
            else:
                params = jnp.array([params[0] * 0.5, params[1] * 2.0])
                skips += 1
                print(f"shrinking")
                if skips > 5:
                    print(f"skip {t}")
                    break



        trace_after_gvmf = trace
        trace = trace_after_gvmf


        delta_cps = jnp.stack(
            jnp.meshgrid(
                jnp.linspace(-0.02, 0.02, 31),
                jnp.linspace(-0.02, 0.02, 31),
                jnp.linspace(-jnp.pi, jnp.pi, 71),
            ),
            axis=-1,
        ).reshape(-1, 3)
        cp_delta_poses = jax.vmap(bayes3d.contact_parameters_to_pose)(delta_cps)


        for _ in range(2):
            key = jax.random.split(key, 2)[-1]


            test_poses = trace["object_pose_0"] @ cp_delta_poses
            test_poses_batches = test_poses.split(20)

            scores = jnp.concatenate(
                [
                    b3d.utils.enumerate_choices_get_scores_jit(
                        trace, key, genjax.Pytree.const(["object_pose_0"]), poses
                    )
                    for poses in test_poses_batches
                ]
            )
            print("Score Max", scores.max())
            samples = jax.random.categorical(key, scores, shape=(50,))

            samples_deg_range = jnp.rad2deg(
                (
                    jnp.max(delta_cps[samples], axis=0)
                    - jnp.min(delta_cps[samples], axis=0)
                )[2]
            )

            trace = b3d.utils.update_choices_jit(
                trace,
                key,
                genjax.Pytree.const(["object_pose_0"]),
                test_poses[samples[0]],
            )
            print(trace.get_score())

            bayes3d.rerun_visualize_trace_t(trace, counter)
            counter += 1
            print("Sampled Angle Range:", samples_deg_range)



        alternate_camera_pose = Pose.from_position_and_target(
            jnp.array([0.00, -0.3, -0.1]) + test_poses[samples[0]].pos, test_poses[samples[0]].pos
        )

        alternate_view_images, _ = renderer.render_attribute_many(
            (alternate_camera_pose.inv() @ test_poses[samples])[:, None, ...],
            object_library.vertices,
            object_library.faces,
            object_library.ranges[jnp.array([0])],
            object_library.attributes,
        )


        for t in range(len(samples)):
            trace = b3d.utils.update_choices_jit(
                trace,
                key,
                genjax.Pytree.const(["object_pose_0"]),
                test_poses[samples[t]],
            )
            bayes3d.rerun_visualize_trace_t(trace, counter)
            counter += 1
            rr.log("/alternate_view_image",rr.Image(alternate_view_images[t]))


