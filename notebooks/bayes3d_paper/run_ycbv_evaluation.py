#!/usr/bin/env python

import copy

import b3d.chisight.gen3d.inference as inference
import b3d.chisight.gen3d.inference_old as inference_old
import b3d.chisight.gen3d.settings as settings
import fire
import jax
import jax.numpy as jnp
from b3d import Pose
from b3d.chisight.gen3d.dataloading import (
    get_initial_state,
    load_object_given_scene,
    load_scene,
)
from tqdm import tqdm


def run_tracking(scene=None, object=None, debug=False):
    import b3d

    FRAME_RATE = 50

    b3d.utils.rr_init("run_ycbv_evaluation")

    if scene is None:
        scenes = range(48, 60)
    elif isinstance(scene, int):
        scenes = [scene]
    elif isinstance(scene, list):
        scenes = scene

    hyperparams = copy.deepcopy(settings.hyperparams)

    for scene_id in scenes:
        all_data, meshes, renderer, intrinsics, initial_object_poses = load_scene(
            scene_id, FRAME_RATE
        )

        object_indices = (
            [object] if object is not None else range(len(initial_object_poses))
        )
        for OBJECT_INDEX in object_indices:
            print(f"Object {OBJECT_INDEX} out of {len(initial_object_poses) - 1}")
            template_pose, model_vertices, model_colors = load_object_given_scene(
                all_data, meshes, renderer, OBJECT_INDEX
            )

            hyperparams["intrinsics"] = intrinsics
            hyperparams["vertices"] = model_vertices
            initial_state = get_initial_state(
                template_pose, model_vertices, model_colors, hyperparams
            )

            tracking_results = {}
            inference_hyperparams = b3d.chisight.gen3d.settings.inference_hyperparams

            ### Run inference ###
            key = jax.random.PRNGKey(156)
            trace = inference.get_initial_trace(
                key, hyperparams, initial_state, all_data[0]["rgbd"]
            )

            for T in tqdm(range(len(all_data))):
                key = b3d.split_key(key)
                trace = inference.advance_time(key, trace, all_data[T]["rgbd"])
                trace = inference_old.inference_step(trace, key, inference_hyperparams)[
                    0
                ]
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

            trace = tracking_results[len(all_data) - 1]
            latent_rgb = b3d.chisight.gen3d.image_kernel.get_latent_rgb_image(
                trace.get_retval()["new_state"], trace.get_args()[0]
            )

            a = b3d.viz_rgb(
                trace.get_choices()["rgbd"][..., :3],
            )
            b = b3d.viz_rgb(
                latent_rgb[..., :3],
            )
            b3d.multi_panel([a, b, b3d.overlay_image(a, b)]).save(
                f"photo_SCENE_{scene_id}_OBJECT_INDEX_{OBJECT_INDEX}_POSES.png"
            )


if __name__ == "__main__":
    fire.Fire(run_tracking)
