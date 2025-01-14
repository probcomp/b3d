import argparse
import json
import os
import time
from os.path import join

import b3d
import b3d.chisight.dense.dense_model
import b3d.chisight.dense.likelihoods.laplace_likelihood
import b3d.chisight.gen3d.inference.inference as inference
import b3d.chisight.gen3d.settings as settings
import jax
import jax.numpy as jnp
import rerun as rr
import trimesh
from b3d.chisight.gen3d.dataloading import (
    calculate_relevant_objects,
    get_initial_state,
    load_trial,
    resize_rgbds_and_get_masks,
)
from b3d.chisight.gen3d.datawriting import write_json
from genjax import Pytree


def foreground_background(depth_map, area, val):
    zero_depth_map = jnp.full(depth_map.shape, val)
    zero_depth_map = zero_depth_map.at[area].set(depth_map[area])
    return zero_depth_map


def main(
    scenario,
    trial_name,
    mesh_file_path,
    pred_file_path,
    save_path,
    recording_id,
    viz_index,
    masked=True,
    debug=True,
):
    start_time = time.time()
    rr.init("demo", recording_id=recording_id)
    rr.connect("127.0.0.1:8813")

    if scenario == "dominoes":
        START_T = 14
    else:
        START_T = 0

    if scenario == "collide":
        FINAL_T = 15
    else:
        FINAL_T = 45

    near_plane = 0.1
    far_plane = 100
    im_width = 350
    im_height = 350
    width = 1024
    height = 1024

    with open(pred_file_path) as f:
        pred_file_all = json.load(f)
    pred_file = pred_file_all[trial_name]

    all_meshes = {}
    for path, dirs, files in os.walk(mesh_file_path):
        for name in files + dirs:
            if name.endswith(".obj"):
                mesh = trimesh.load(os.path.join(path, name))
                all_meshes[name[:-4]] = mesh

    scaling_factor = im_height / height
    vfov = 54.43222 / 180.0 * jnp.pi
    tan_half_vfov = jnp.tan(vfov / 2.0)
    tan_half_hfov = tan_half_vfov * width / float(height)
    fx = width / 2.0 / tan_half_hfov
    fy = height / 2.0 / tan_half_vfov

    renderer = b3d.renderer.renderer_original.RendererOriginal(
        width * scaling_factor,
        height * scaling_factor,
        fx * scaling_factor,
        fy * scaling_factor,
        (width / 2) * scaling_factor,
        (height / 2) * scaling_factor,
        near_plane,
        far_plane,
    )

    b3d.reload(b3d.chisight.dense.likelihoods.laplace_likelihood)
    likelihood_func = b3d.chisight.dense.likelihoods.laplace_likelihood.likelihood_func

    b3d.reload(b3d.chisight.dense.dense_model)
    dynamic_object_generative_model, viz_trace = (
        b3d.chisight.dense.dense_model.make_dense_multiobject_dynamics_model(
            renderer, likelihood_func
        )
    )
    importance_jit = jax.jit(dynamic_object_generative_model.importance)

    likelihood_args = {
        "fx": renderer.fx,
        "fy": renderer.fy,
        "cx": renderer.cx,
        "cy": renderer.cy,
        "image_width": Pytree.const(renderer.width),
        "image_height": Pytree.const(renderer.height),
        "masked": Pytree.const(masked),
        "check_interp": Pytree.const(True),
        "num_mc_sample": Pytree.const(500),
        "interp_penalty": Pytree.const(1e5),
    }

    inference_hyperparams = b3d.chisight.gen3d.settings.inference_hyperparams

    hdf5_file_path = join(
        "/home/haoliangwang/data/physion_hdf5",
        scenario + "_all_movies",
        f"{trial_name}.hdf5",
    )
    initalization_time = time.time()
    print(f"\t\t Initialization time: {initalization_time - start_time}")

    (
        rgbds_original,
        seg_arr_original,
        object_ids,
        object_segmentation_colors,
        background_areas,
        camera_pose,
    ) = load_trial(hdf5_file_path, FINAL_T)
    loading_time = time.time()
    print(f"\t\t Loading time: {loading_time - initalization_time}")

    hyperparams = settings.hyperparams
    hyperparams["camera_pose"] = camera_pose
    hyperparams["likelihood_args"] = likelihood_args

    initial_state, hyperparams = get_initial_state(
        pred_file,
        object_ids,
        object_segmentation_colors,
        all_meshes,
        seg_arr_original[START_T],
        rgbds_original[START_T],
        hyperparams,
    )
    first_state_time = time.time()
    print(f"\t\t First state time: {first_state_time - loading_time}")

    rgbds, all_areas, background_areas = resize_rgbds_and_get_masks(
        rgbds_original, seg_arr_original, background_areas, im_height, im_width
    )
    hyperparams["background"] = jnp.asarray(
        [
            foreground_background(rgbds[t], background_areas[t], jnp.inf)
            for t in range(rgbds.shape[0])
        ]
    )

    key = jax.random.PRNGKey(156)
    trace = inference.get_initial_trace(
        key,
        importance_jit,
        hyperparams,
        initial_state,
        foreground_background(rgbds[START_T], all_areas[START_T], 0.0),
    )
    viz_trace(trace, t=viz_index)
    first_trace_time = time.time()
    print(f"\t\t First trace time: {first_trace_time - first_state_time}")

    posterior_across_frames = {"pose": []}
    for i, T in enumerate(range(START_T, FINAL_T)):
        this_iteration_start_time = time.time()
        if i == 0:
            relevant_objects = object_ids
        else:
            # relevant_objects = object_ids
            relevant_objects = calculate_relevant_objects(
                rgbds_original[T],
                rgbds_original[T - 1],
                seg_arr_original[T],
                seg_arr_original[T - 1],
                object_ids,
                object_segmentation_colors,
            )
        key = b3d.split_key(key)
        trace, this_frame_posterior = inference.inference_step(
            key,
            trace,
            foreground_background(rgbds[T], all_areas[T], 0.0),
            inference_hyperparams,
            [Pytree.const(f"object_pose_{o_id}") for o_id in relevant_objects],
        )
        posterior_across_frames["pose"].append(this_frame_posterior)
        viz_trace(trace, t=viz_index + i + 1)
        this_iteration_end_time = time.time()
        print(f"\t\t frame {T}: {this_iteration_end_time - this_iteration_start_time}, relevant objects: {relevant_objects}")

    write_json(
        pred_file,
        hyperparams,
        posterior_across_frames,
        save_path,
        scenario,
        trial_name,
        debug=debug,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="collide", type=str)
    parser.add_argument("--trial_name", default="", type=str)
    parser.add_argument("--recording_id", default="", type=str)
    parser.add_argument("--viz_index", default="", type=int)
    args = parser.parse_args()
    scenario = args.scenario
    trial_name = args.trial_name
    recording_id = args.recording_id
    viz_index = args.viz_index

    mesh_file_path = "/home/haoliangwang/data/all_flex_meshes/core"
    save_path = "/home/haoliangwang/data/b3d_tracking_results/test"
    pred_file_path = "/home/haoliangwang/data/pred_files/gt_info/gt.json"

    main(
        scenario,
        trial_name,
        mesh_file_path,
        pred_file_path,
        save_path,
        recording_id,
        viz_index,
    )
