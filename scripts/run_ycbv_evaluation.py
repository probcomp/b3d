#!/usr/bin/env python

import os

import b3d.chisight.gen3d.inference as inference
import fire
import genjax
import jax
import jax.numpy as jnp
from b3d import Mesh, Pose
from b3d.chisight.gen3d.model import (
    dynamic_object_generative_model,
    make_colors_choicemap,
    make_depth_nonreturn_prob_choicemap,
    make_visibility_prob_choicemap,
)
from genjax import Pytree
from tqdm import tqdm


def run_tracking(scene=None, object=None, debug=False):
    import b3d

    FRAME_RATE = 50

    b3d.utils.rr_init("run_ycbv_evaluation")

    ycb_dir = os.path.join(b3d.utils.get_assets_path(), "bop/ycbv")

    if scene is None:
        scenes = range(48, 60)
    elif isinstance(scene, int):
        scenes = [scene]
    elif isinstance(scene, list):
        scenes = scene

    import b3d.chisight.gen3d.settings

    hyperparams = b3d.chisight.gen3d.settings.hyperparams
    # inference_hyperparams = b3d.chisight.gen3d.settings.inference_hyperparams

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

        # initial_camera_pose = all_data[0]["camera_pose"]
        initial_object_poses = all_data[0]["object_poses"]

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
            model_vertices = template_pose.inv().apply(xyz_rendered[mask])
            model_colors = all_data[T]["rgbd"][..., :3][mask]

            subset = jax.random.permutation(jax.random.PRNGKey(0), len(model_vertices))[
                : min(10000, len(model_vertices))
            ]
            model_vertices = model_vertices[subset]
            model_colors = model_colors[subset]

            hyperparams["intrinsics"] = {
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "image_height": Pytree.const(image_height),
                "image_width": Pytree.const(image_width),
                "near": 0.01,
                "far": 3.0,
            }
            hyperparams["vertices"] = model_vertices

            num_vertices = model_vertices.shape[0]
            previous_state = {
                "pose": template_pose,
                "colors": model_colors,
                "visibility_prob": jnp.ones(num_vertices)
                * hyperparams["visibility_prob_kernel"].support[-1],
                "depth_nonreturn_prob": jnp.ones(num_vertices)
                * hyperparams["depth_nonreturn_prob_kernel"].support[0],
                "depth_scale": hyperparams["depth_scale_kernel"].support[0],
                "color_scale": hyperparams["color_scale_kernel"].support[0],
            }

            choicemap = (
                genjax.ChoiceMap.d(
                    {
                        "pose": previous_state["pose"],
                        "color_scale": previous_state["color_scale"],
                        "depth_scale": previous_state["depth_scale"],
                        "rgbd": all_data[T]["rgbd"],
                    }
                )
                ^ make_visibility_prob_choicemap(previous_state["visibility_prob"])
                ^ make_colors_choicemap(previous_state["colors"])
                ^ make_depth_nonreturn_prob_choicemap(
                    previous_state["depth_nonreturn_prob"]
                )
            )
            key = jax.random.PRNGKey(0)

            tracking_results = {}
            trace = dynamic_object_generative_model.importance(
                key, choicemap, (hyperparams, previous_state)
            )[0]

            ### Run inference ###
            for T in tqdm(range(len(all_data))):
                key = b3d.split_key(key)
                trace = inference.inference_step_c2f(
                    key,
                    2,  # number of sequential iterations of the parallel pose proposal to consider
                    3000,  # number of poses to propose in parallel
                    # So the total number of poses considered at each step of C2F is 5000 * 1
                    trace,
                    all_data[T]["rgbd"],
                    prev_color_proposal_laplace_scale=0.1,  # inference_hyperparams.prev_color_proposal_laplace_scale,
                    obs_color_proposal_laplace_scale=0.1,  # inference_hyperparams.obs_color_proposal_laplace_scale,
                    do_stochastic_color_proposals=False,
                )
                tracking_results[T] = trace

                if debug:
                    b3d.chisight.gen3d.model.viz_trace(
                        trace,
                        T,
                        ground_truth_vertices=meshes[OBJECT_INDEX].vertices,
                        ground_truth_pose=all_data[T]["camera_pose"].inv()
                        @ all_data[T]["object_poses"][OBJECT_INDEX],
                    )

            inferred_poses = Pose.stack_poses(
                [
                    tracking_results[t].get_choices()["pose"]
                    for t in range(len(all_data))
                ]
            )
            jnp.savez(
                f"SCENE_{scene_id}_OBJECT_INDEX_{OBJECT_INDEX}_POSES.npy",
                position=inferred_poses.position,
                quaternion=inferred_poses.quat,
            )

            import b3d.chisight.gen3d.visualization as viz

            viz.make_video_from_traces(
                [tracking_results[t] for t in range(len(all_data))],
                f"SCENE_{scene_id}_OBJECT_INDEX_{OBJECT_INDEX}.mp4",
                scale=0.25,
            )


if __name__ == "__main__":
    fire.Fire(run_tracking)
