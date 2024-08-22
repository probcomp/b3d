#!/usr/bin/env python
import fire


def run_tracking(scene=None, object=None, debug=False):
    import importlib
    import os

    import b3d
    import genjax
    import jax
    import jax.numpy as jnp
    from b3d import Mesh, Pose
    from genjax import Pytree
    from tqdm import tqdm

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
            b3d.rr_log_rgb(rendered_rgbd, "image/rgb/rendered_full_mesh")
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

            choicemap = genjax.ChoiceMap.d(
                {
                    "rgbd": all_data[T]["rgbd"],
                    "camera_pose": Pose.identity(),
                    "object_pose_0": template_pose,
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
                trace, key = (
                    b3d.bayes3d.enumerative_proposals.gvmf_and_select_best_move(
                        trace, key, 0.04, 700.0, "object_pose_0", 20000
                    )
                )
                trace, key = (
                    b3d.bayes3d.enumerative_proposals.gvmf_and_select_best_move(
                        trace, key, 0.03, 700.0, "object_pose_0", 20000
                    )
                )
                trace, key = (
                    b3d.bayes3d.enumerative_proposals.gvmf_and_select_best_move(
                        trace, key, 0.02, 1000.0, "object_pose_0", 20000
                    )
                )
                trace, key = (
                    b3d.bayes3d.enumerative_proposals.gvmf_and_select_best_move(
                        trace, key, 0.01, 1000.0, "object_pose_0", 20000
                    )
                )
                tracking_results[T] = trace

                if T > 0 and T % 10 == 0:
                    template_pose = trace.get_choices()["object_pose_0"]
                    rendered_rgbd = renderer.render_rgbd_from_mesh(
                        meshes[OBJECT_INDEX].transform(template_pose)
                    )
                    xyz_rendered = b3d.xyz_from_depth(
                        rendered_rgbd[..., 3], fx, fy, cx, cy
                    )

                    fx, fy, cx, cy = all_data[T]["camera_intrinsics"]
                    xyz_observed = b3d.xyz_from_depth(
                        all_data[T]["rgbd"][..., 3], fx, fy, cx, cy
                    )
                    mask = (xyz_observed[..., 2] > 0) * (
                        jnp.linalg.norm(xyz_rendered - xyz_observed, axis=-1) < 0.01
                    )
                    mesh = Mesh(
                        vertices=template_pose.inv().apply(xyz_rendered[mask]),
                        faces=jnp.zeros((0, 3), dtype=jnp.int32),
                        vertex_attributes=all_data[T]["rgbd"][..., :3][mask],
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
