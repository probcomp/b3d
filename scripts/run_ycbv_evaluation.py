#!/usr/bin/env python

import copy

import b3d.chisight.gen3d.inference as inference
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
            inference_hyperparams = b3d.chisight.gen3d.settings.inference_hyperparams  # noqa

            ### Run inference ###
            key = jax.random.PRNGKey(156)
            trace = inference.get_initial_trace(
                key, hyperparams, initial_state, all_data[0]["rgbd"]
            )

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
