#!/usr/bin/env python

import copy
import os
import pprint
from datetime import datetime
from pathlib import Path

import fire
import jax
import jax.numpy as jnp
import rerun as rr
from tqdm import tqdm

import b3d
import b3d.chisight.gen3d.inference.inference as inference
import b3d.chisight.gen3d.settings as settings
import b3d.chisight.gen3d.visualization as viz
from b3d import Pose
from b3d.chisight.gen3d.dataloading import (
    get_initial_state,
    load_object_given_scene,
    load_scene,
)
from b3d.chisight.gen3d.model import viz_trace as rr_viz_trace


def setup_save_directory():
    # Make a folder, stamped with the current time.
    current_time = datetime.now().strftime("%Y-%m-%d--%H:%M")
    folder_name = (
        b3d.get_root_path() / "test_results" / "gen3d" / f"gen3d_{current_time}"
    )
    Path(folder_name).mkdir(parents=True, exist_ok=True)
    video_folder_name = folder_name / "mp4"
    npy_folder_name = folder_name / "npy"
    rr_folder_name = folder_name / "rr"
    os.mkdir(rr_folder_name)
    os.mkdir(video_folder_name)
    os.mkdir(npy_folder_name)
    return folder_name, video_folder_name, npy_folder_name, rr_folder_name


def save_hyperparams(folder_name, hyperparams, inference_hyperparams):
    hyperparams_file = folder_name / "hyperparams.txt"
    with open(hyperparams_file, "w") as f:
        f.write("Hyperparameters:\n")
        f.write(pprint.pformat(hyperparams))
        f.write("\n\n\nInference Hyperparameters:\n")
        f.write(pprint.pformat(inference_hyperparams))


def run_tracking(scene=None, object=None, save_rerun=False, max_n_frames=None):
    folder_name, video_folder_name, npy_folder_name, rr_folder_name = (
        setup_save_directory()
    )

    hyperparams = copy.deepcopy(settings.hyperparams)
    inference_hyperparams = b3d.chisight.gen3d.settings.inference_hyperparams  # noqa
    save_hyperparams(folder_name, hyperparams, inference_hyperparams)

    FRAME_RATE = 50

    if scene is None:
        scenes = range(48, 60)
    elif isinstance(scene, int):
        scenes = [scene]
    elif isinstance(scene, list):
        scenes = scene

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

            ### Run inference ###
            key = jax.random.PRNGKey(156)
            trace = inference.get_initial_trace(
                key, hyperparams, initial_state, all_data[0]["rgbd"]
            )

            if save_rerun:
                rr.init(f"SCENE_{scene_id}_OBJECT_INDEX_{OBJECT_INDEX}")
                rr.save(
                    rr_folder_name / f"SCENE_{scene_id}_OBJECT_INDEX_{OBJECT_INDEX}.rrd"
                )

            if max_n_frames is not None:
                maxT = min(max_n_frames, len(all_data))
            else:
                maxT = len(all_data)

            for T in tqdm(range(maxT)):
                key = b3d.split_key(key)
                trace, _ = inference.inference_step(
                    key, trace, all_data[T]["rgbd"], inference_hyperparams
                )
                tracking_results[T] = trace

                if save_rerun:
                    rr_viz_trace(
                        trace,
                        T,
                        ground_truth_vertices=meshes[OBJECT_INDEX].vertices,
                        ground_truth_pose=all_data[T]["camera_pose"].inv()
                        @ all_data[T]["object_poses"][OBJECT_INDEX],
                    )

            inferred_poses = Pose.stack_poses(
                [tracking_results[t].get_choices()["pose"] for t in range(maxT)]
            )
            jnp.savez(
                npy_folder_name
                / f"SCENE_{scene_id}_OBJECT_INDEX_{OBJECT_INDEX}_POSES.npy",
                position=inferred_poses.position,
                quaternion=inferred_poses.quat,
            )

            viz.make_video_from_traces(
                [tracking_results[t] for t in range(maxT)],
                video_folder_name / f"SCENE_{scene_id}_OBJECT_INDEX_{OBJECT_INDEX}.mp4",
                scale=0.25,
            )

            if save_rerun:
                rr.disconnect()
                print("rerun disconnected")


if __name__ == "__main__":
    fire.Fire(run_tracking)
