#!/usr/bin/env python
import fire


def run_tracking(scene=None, object=None, debug=False):
    import importlib
    import os

    import genjax
    import jax
    import jax.numpy as jnp
    from genjax import Pytree
    from tqdm import tqdm

    import b3d
    from b3d import Mesh, Pose

    importlib.reload(b3d.mesh)
    importlib.reload(b3d.io.data_loader)
    importlib.reload(b3d.utils)
    importlib.reload(b3d.renderer.renderer_original)

    FRAME_RATE = 50

    ycb_dir = os.path.join(b3d.get_assets_path(), "bop/ycbv")

    b3d.rr_init("run_ycbv_evaluation")

    if scene is None:
        scenes = range(48, 60)
    elif isinstance(scene, int):
        scenes = [scene]
    elif isinstance(scene, list):
        scenes = scene

    for scene_id in scenes:
        print(f"Scene {scene_id}")
        num_scenes = b3d.io.data_loader.get_ycbv_num_test_images(ycb_dir, scene_id)

        # image_ids = [image] if image is not None else range(1, num_scenes, FRAME_RATE)
        image_ids = range(1, num_scenes + 1, FRAME_RATE)
        all_data = b3d.io.get_ycbv_test_images(ycb_dir, scene_id, image_ids)

        meshes = [
            Mesh.from_obj_file(
                os.path.join(ycb_dir, f'models/obj_{f"{id + 1}".rjust(6, "0")}.ply')
            ).scale(0.001)
            for id in all_data[0]["object_types"]
        ]

        image_height, image_width = all_data[0]["rgbd"].shape[:2]
        fx, fy, cx, cy = all_data[0]["camera_intrinsics"]
        scaling_factor = 1.0
        renderer = b3d.renderer.renderer_original.RendererOriginal(
            image_width * scaling_factor,
            image_height * scaling_factor,
            fx * scaling_factor,
            fy * scaling_factor,
            cx * scaling_factor,
            cy * scaling_factor,
            0.01,
            4.0,
        )

        def grid_outlier_prob(trace, values):
            return jax.vmap(
                lambda x: info_from_trace(
                    b3d.update_choices(trace, Pytree.const(("outlier_probability",)), x)
                )["scores"]
            )(values)

        @jax.jit
        def update_pose_and_color(trace, address, pose):
            trace = b3d.update_choices(trace, address, pose)
            info = info_from_trace(trace)
            current_outlier_probabilities = trace.get_choices()["outlier_probability"]
            model_rgbd, observed_rgbd = (
                info["model_rgbd"],
                info["corresponding_observed_rgbd"],
            )
            deltas = (observed_rgbd - model_rgbd)[..., :3]
            deltas_clipped = jnp.clip(deltas, -0.1, 0.1)

            mesh = trace.get_args()[0]["meshes"][0]
            is_inlier = current_outlier_probabilities == outlier_probability_sweep[0]
            mesh.vertex_attributes = (
                mesh.vertex_attributes + deltas_clipped * is_inlier[..., None]
            )

            trace, _ = importance_jit(
                jax.random.PRNGKey(2),
                trace.get_choices(),
                (
                    {
                        "num_objects": Pytree.const(1),
                        "meshes": [mesh],
                        "likelihood_args": likelihood_args,
                    },
                ),
            )
            return trace.get_score()

        def _gvmf_and_select_best_move(
            trace, key, variance, concentration, address, number
        ):
            test_poses = Pose.concatenate_poses(
                [
                    jax.vmap(
                        Pose.sample_gaussian_vmf_pose, in_axes=(0, None, None, None)
                    )(
                        jax.random.split(key, number),
                        trace.get_choices()[address.const],
                        variance,
                        concentration,
                    ),
                    trace.get_choices()[address.const][None, ...],
                ]
            )
            scores = jax.vmap(update_pose_and_color, in_axes=(None, None, 0))(
                trace, address, test_poses
            )
            trace = b3d.update_choices(
                trace,
                address,
                test_poses[scores.argmax()],
            )
            key = jax.random.split(key, 2)[-1]
            return trace, key

        gvmf_and_select_best_move = jax.jit(
            _gvmf_and_select_best_move, static_argnames=["number"]
        )

        from b3d.chisight.dense.likelihoods.simplified_rendering_laplace_likelihood import (
            simplified_rendering_laplace_likelihood,
        )

        model, viz_trace, info_from_trace = (
            b3d.chisight.dense.dense_model.make_dense_multiobject_model(
                None, simplified_rendering_laplace_likelihood
            )
        )
        importance_jit = jax.jit(model.importance)

        # initial_camera_pose = all_data[0]["camera_pose"]
        initial_object_poses = all_data[0]["object_poses"]

        likelihood_args = {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "image_height": Pytree.const(image_height),
            "image_width": Pytree.const(image_width),
        }

        object_indices = (
            [object] if object is not None else range(len(initial_object_poses))
        )
        for OBJECT_INDEX in object_indices:
            print(f"Object {OBJECT_INDEX} out of {len(initial_object_poses) - 1}")
            T = 0

            template_pose = (
                all_data[T]["camera_pose"].inv()
                @ all_data[T]["object_poses"][OBJECT_INDEX]
            )
            rendered_rgbd = renderer.render_rgbd_from_mesh(
                meshes[OBJECT_INDEX].transform(template_pose)
            )
            xyz_rendered = b3d.xyz_from_depth(rendered_rgbd[..., 3], fx, fy, cx, cy)

            fx, fy, cx, cy = all_data[T]["camera_intrinsics"]
            xyz_observed = b3d.xyz_from_depth(
                all_data[T]["rgbd"][..., 3], fx, fy, cx, cy
            )
            mask = (
                all_data[T]["masks"][OBJECT_INDEX]
                * (xyz_observed[..., 2] > 0)
                * (jnp.linalg.norm(xyz_rendered - xyz_observed, axis=-1) < 0.01)
            )
            mesh = Mesh(
                vertices=template_pose.inv().apply(xyz_rendered[mask]),
                faces=jnp.zeros((0, 3), dtype=jnp.int32),
                vertex_attributes=all_data[T]["rgbd"][..., :3][mask],
            )

            outlier_probability_sweep = jnp.array([0.05, 1.0])

            choicemap = genjax.ChoiceMap.d(
                {
                    "rgbd": all_data[T]["rgbd"],
                    "camera_pose": Pose.identity(),
                    "object_pose_0": template_pose,
                    "outlier_probability": jnp.ones(len(mesh.vertices))
                    * outlier_probability_sweep[0],
                    "color_noise_variance": 0.05,
                    "depth_noise_variance": 0.01,
                }
            )

            trace0, _ = importance_jit(
                jax.random.PRNGKey(2),
                choicemap,
                (
                    {
                        "num_objects": Pytree.const(1),
                        "meshes": [mesh],
                        "likelihood_args": likelihood_args,
                    },
                ),
            )
            key = jax.random.PRNGKey(100)

            trace = trace0
            tracking_results = {}
            for T in tqdm(range(len(all_data))):
                trace = b3d.update_choices(
                    trace,
                    Pytree.const(("rgbd",)),
                    all_data[T]["rgbd"],
                )

                for _ in range(5):
                    trace, key = gvmf_and_select_best_move(
                        trace,
                        key,
                        0.01,
                        1000.0,
                        Pytree.const(("object_pose_0",)),
                        10000,
                    )
                    trace, key = gvmf_and_select_best_move(
                        trace,
                        key,
                        0.005,
                        2000.0,
                        Pytree.const(("object_pose_0",)),
                        10000,
                    )
                # viz_trace(trace, T)

                if T % 1 == 0:
                    trace = b3d.bayes3d.enumerative_proposals.enumerate_and_select_best(
                        trace,
                        Pytree.const(("color_noise_variance",)),
                        jnp.linspace(0.05, 0.1, 10),
                    )
                    trace = b3d.bayes3d.enumerative_proposals.enumerate_and_select_best(
                        trace,
                        Pytree.const(("depth_noise_variance",)),
                        jnp.linspace(0.005, 0.01, 10),
                    )

                    current_outlier_probabilities = trace.get_choices()[
                        "outlier_probability"
                    ]
                    scores = grid_outlier_prob(
                        trace,
                        outlier_probability_sweep[..., None]
                        * jnp.ones_like(current_outlier_probabilities),
                    )
                    trace = b3d.update_choices(
                        trace,
                        Pytree.const(("outlier_probability",)),
                        outlier_probability_sweep[jnp.argmax(scores, axis=0)],
                    )

                    current_outlier_probabilities = trace.get_choices()[
                        "outlier_probability"
                    ]
                    # b3d.rr_log_cloud(
                    #     mesh.vertices,
                    #     colors=colors[1 * (current_outlier_probabilities == outlier_probability_sweep[0])],
                    #     channel="cloud/outlier_probabilities"
                    # )

                    info = info_from_trace(trace)
                    current_outlier_probabilities = trace.get_choices()[
                        "outlier_probability"
                    ]
                    model_rgbd, observed_rgbd = (
                        info["model_rgbd"],
                        info["corresponding_observed_rgbd"],
                    )
                    deltas = observed_rgbd - model_rgbd
                    deltas_clipped = jnp.clip(deltas, -0.05, 0.05)
                    new_model_rgbd = model_rgbd + deltas_clipped

                    mesh = trace.get_args()[0]["meshes"][0]
                    is_inlier = (
                        current_outlier_probabilities == outlier_probability_sweep[0]
                    )
                    mesh.vertex_attributes = mesh.vertex_attributes.at[is_inlier].set(
                        new_model_rgbd[is_inlier, :3]
                    )

                    trace, _ = importance_jit(
                        jax.random.PRNGKey(2),
                        trace.get_choices(),
                        (
                            {
                                "num_objects": Pytree.const(1),
                                "meshes": [mesh],
                                "likelihood_args": likelihood_args,
                            },
                        ),
                    )
                    tracking_results[T] = trace
                if debug:
                    viz_trace(trace, T)

            inferred_poses = Pose.stack_poses(
                [
                    tracking_results[t].get_choices()["object_pose_0"]
                    for t in range(len(all_data))
                ]
            )
            jnp.savez(
                f"SCENE_{scene_id}_OBJECT_INDEX_{OBJECT_INDEX}_POSES.npy",
                position=inferred_poses.position,
                quaternion=inferred_poses.quat,
            )

            trace = tracking_results[len(all_data) - 1]
            info = info_from_trace(trace)
            rendered_rgbd = info["latent_rgbd"]

            a = b3d.viz_rgb(
                trace.get_choices()["rgbd"][..., :3],
            )
            b = b3d.viz_rgb(
                rendered_rgbd[..., :3],
            )
            b3d.multi_panel([a, b, b3d.overlay_image(a, b)]).save(
                f"photo_SCENE_{scene_id}_OBJECT_INDEX_{OBJECT_INDEX}_POSES.png"
            )


if __name__ == "__main__":
    fire.Fire(run_tracking)
