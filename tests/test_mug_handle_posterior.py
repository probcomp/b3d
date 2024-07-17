import rerun as rr
import genjax
import os
import jax.numpy as jnp
import jax
from b3d import Pose
import b3d
import b3d.bayes3d as bayes3d
import trimesh
from genjax import Pytree

PORT = 8812
rr.init("233")
rr.connect(addr=f"127.0.0.1:{PORT}")


class TestMugHandlePosterior:
    def test_gridding_posterior(self, renderer):
        image_width, image_height, fx, fy, cx, cy, near, far = (
            100,
            100,
            200.0,
            200.0,
            50.0,
            50.0,
            0.01,
            10.0,
        )
        renderer.set_intrinsics(image_width, image_height, fx, fy, cx, cy, near, far)

        mesh_path = os.path.join(
            b3d.get_root_path(), "assets/shared_data_bucket/025_mug/textured.obj"
        )
        mesh = trimesh.load(mesh_path)
        vertices = jnp.array(mesh.vertices)
        vertices = vertices - jnp.mean(vertices, axis=0)
        faces = jnp.array(mesh.faces)
        vertex_colors = vertices * 0.0 + jnp.array([1.0, 0.0, 0.0])
        vertex_colors = jnp.array(mesh.visual.to_color().vertex_colors)[..., :3] / 255.0
        print("Vertices dimensions :", vertices.max(0) - vertices.min(0))

        key = jax.random.PRNGKey(0)

        camera_pose = Pose.from_position_and_target(
            jnp.array([0.6, 0.0, 0.0]), jnp.array([0.0, 0.0, 0.0])
        )

        cp_to_pose = lambda cp: Pose(
            jnp.array([cp[0], cp[1], 0.0]),
            b3d.Rot.from_rotvec(jnp.array([0.0, 0.0, cp[2]])).as_quat(),
        )
        object_library = bayes3d.MeshLibrary.make_empty_library()
        object_library.add_object(vertices, faces, vertex_colors)

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

        cps_to_test = [
            jnp.array([0.0, 0.0, jnp.pi]),  # Hidden
            jnp.array([0.0, 0.0, -jnp.pi / 2]),  # Side
            jnp.array([0.0, 0.0, 0.0]),  # Front
            jnp.array([0.0, 0.0, +jnp.pi / 2]),  # Side
        ]

        sampled_degree_range_bounds = [
            (50.0, 80.0),
            (0.0, 20.0),
            (0.0, 20.0),
            (0.0, 20.0),
        ]

        model = bayes3d.model_multiobject_gl_factory(renderer)
        importance_jit = jax.jit(model.importance)

        for text_index in range(len(cps_to_test)):
            gt_cp = cps_to_test[text_index]

            object_pose = cp_to_pose(gt_cp)

            gt_trace, _ = model.importance(
                jax.random.PRNGKey(0),
                genjax.ChoiceMap.d(
                    {
                        "camera_pose": camera_pose,
                        "object_pose_0": object_pose,
                        "object_0": 0,
                        # "observed_rgb": gt_img,
                        # "observed_depth": gt_depth,
                    }
                ),
                (jnp.arange(1), model_args, object_library),
            )
            print("IMG Size :", gt_trace.get_choices()["observed_rgb_depth"][0].shape)

            delta_cps = jnp.stack(
                jnp.meshgrid(
                    jnp.linspace(-0.02, 0.02, 31),
                    jnp.linspace(-0.02, 0.02, 31),
                    jnp.linspace(-jnp.pi, jnp.pi, 71),
                ),
                axis=-1,
            ).reshape(-1, 3)
            cp_delta_poses = jax.vmap(cp_to_pose)(delta_cps)

            test_poses = gt_trace.get_choices()["object_pose_0"] @ cp_delta_poses
            test_poses_batches = test_poses.split(10)
            scores = jnp.concatenate(
                [
                    b3d.enumerate_choices_get_scores_jit(
                        gt_trace, key, Pytree.const(("object_pose_0",)), poses
                    )
                    for poses in test_poses_batches
                ]
            )

            samples = jax.random.categorical(key, scores, shape=(50,))
            print("GT Contact Parameter :", gt_cp)

            samples_deg_range = jnp.rad2deg(
                (
                    jnp.max(delta_cps[samples], axis=0)
                    - jnp.min(delta_cps[samples], axis=0)
                )[2]
            )

            print("Sampled Angle Range:", samples_deg_range)

            alternate_camera_pose = Pose.from_position_and_target(
                jnp.array([0.01, 0.000, 0.9]), object_pose.pos
            )
            alternate_view_images, _ = renderer.render_attribute_many(
                (alternate_camera_pose.inv() @ test_poses[samples])[:, None, ...],
                object_library.vertices,
                object_library.faces,
                object_library.ranges[jnp.array([0])],
                object_library.attributes,
            )

            for t in range(len(samples)):
                trace_ = b3d.update_choices_jit(
                    gt_trace,
                    key,
                    Pytree.const(("object_pose_0",)),
                    test_poses[samples[t]],
                )
                bayes3d.rerun_visualize_trace_t(trace_, t)
                rr.set_time_sequence("frame", t)
                rr.log("alternate_view_image", rr.Image(alternate_view_images[t, ...]))
                rr.log(
                    "text",
                    rr.TextDocument(f"{delta_cps[samples[t]]} \n {scores[samples[t]]}"),
                )

            assert (
                samples_deg_range >= sampled_degree_range_bounds[text_index][0]
            ), f"{samples_deg_range}, {sampled_degree_range_bounds[text_index]}"
            assert (
                samples_deg_range <= sampled_degree_range_bounds[text_index][1]
            ), f"{samples_deg_range}, {sampled_degree_range_bounds[text_index]}"

            bayes3d.rerun_visualize_trace_t(gt_trace, 0)
