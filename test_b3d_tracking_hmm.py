import argparse
import collections
import json
import os
from os import listdir
from os.path import isfile, join
import rerun as rr
import trimesh

import b3d
import b3d.chisight.dense.dense_model
import b3d.chisight.dense.likelihoods.laplace_likelihood
import b3d.chisight.gen3d.inference.inference as inference
import b3d.chisight.gen3d.settings as settings
import jax
import jax.numpy as jnp
from b3d.chisight.gen3d.dataloading import (
    get_initial_state,
    load_trial,
    resize_rgbds_and_get_masks,
)
from b3d.chisight.gen3d.datawriting import write_json
from genjax import Pytree
from b3d.chisight.dense.dense_model import get_new_state


def foreground_background(depth_map, area, val):
    zero_depth_map = jnp.full(depth_map.shape, val)
    zero_depth_map = zero_depth_map.at[area].set(depth_map[area])
    return zero_depth_map


def main(
    hdf5_file_path,
    scenario,
    mesh_file_path,
    pred_file_path,
    save_path,
    masked=True,
    debug=True,
):
    rr.init("demo")
    rr.connect("127.0.0.1:8813")
    rr.log("/", rr.ViewCoordinates.LEFT_HAND_Y_UP, static=True)

    near_plane = 0.1
    far_plane = 100
    im_width = 350
    im_height = 350
    width = 1024
    height = 1024

    START_T = 0
    if scenario == "collide":
        FINAL_T = 15
    else:
        FINAL_T = 45

    with open(pred_file_path) as f:
        pred_file_all = json.load(f)

    all_meshes = {}
    for path, dirs, files in os.walk(mesh_file_path):
        for name in files + dirs:
            if name.endswith(".obj"):
                mesh = trimesh.load(os.path.join(path, name))
                all_meshes[name[:-4]] = mesh
    ordered_all_meshes = collections.OrderedDict(sorted(all_meshes.items()))

    scenario_path = join(hdf5_file_path, scenario + "_all_movies")
    onlyhdf5 = [
        f
        for f in listdir(scenario_path)
        if isfile(join(scenario_path, f)) and join(scenario_path, f).endswith(".hdf5")
    ]

    scaling_factor = im_height / height
    vfov = 54.43222 / 180.0 * jnp.pi
    tan_half_vfov = jnp.tan(vfov / 2.0)
    tan_half_hfov = tan_half_vfov * width / float(height)
    fx = width / 2.0 / tan_half_hfov  # focal length in pixel space
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

    _, viz_trace = b3d.chisight.dense.dense_model.make_dense_multiobject_dynamics_model(
        renderer, likelihood_func
    )

    likelihood_args = {
        "fx": renderer.fx,
        "fy": renderer.fy,
        "cx": renderer.cx,
        "cy": renderer.cy,
        "image_width": Pytree.const(renderer.width),
        "image_height": Pytree.const(renderer.height),
        "masked": Pytree.const(masked),
        "check_interp": Pytree.const(False),
        "num_mc_sample": Pytree.const(500),
        "interp_penalty": Pytree.const(1000),
    }

    viz_index = 0
    for trial_index, hdf5_file in enumerate(onlyhdf5):
        trial_name = hdf5_file[:-5]
        # if trial_name != "pilot_it2_rollingSliding_simple_ramp_tdw_1_dis_1_occ_0017":
        #     continue

        print(trial_index + 1, "\t", trial_name)
        hdf5_file_path = join(scenario_path, hdf5_file)

        pred_file = pred_file_all[trial_name]
        rgbds, seg_arr, object_ids, object_segmentation_colors, background_areas, camera_pose, _, _, = (
            load_trial(hdf5_file_path)
        )

        inference_hyperparams = b3d.chisight.gen3d.settings.inference_hyperparams
        hyperparams = settings.hyperparams
        hyperparams["camera_pose"] = camera_pose
        hyperparams["likelihood_args"] = likelihood_args

        initial_state, hyperparams = get_initial_state(
            pred_file,
            object_ids,
            object_segmentation_colors,
            ordered_all_meshes,
            seg_arr[START_T],
            rgbds[START_T],
            hyperparams,
        )

        rgbds, all_areas, background_areas = resize_rgbds_and_get_masks(
            rgbds, seg_arr, background_areas, im_height, im_width
        )
        hyperparams["background"] = jnp.asarray([foreground_background(rgbds[t], background_areas[t], jnp.inf) for t in range(rgbds.shape[0])])

        key = jax.random.PRNGKey(156)
        trace = inference.get_initial_trace(
            key,
            renderer,
            likelihood_func,
            hyperparams,
            initial_state,
            foreground_background(rgbds[START_T], all_areas[START_T], 0.0),
        )
        viz_trace(trace, t=viz_index)
        print("finished initializing trace")

        posterior_across_frames = {"pose": []}
        for T in range(FINAL_T):
            print(f"time {T}")
            key = b3d.split_key(key)
            trace, posterior_across_frames = inference.inference_step(
                key,
                trace,
                foreground_background(rgbds[T], all_areas[T], 0.0),
                inference_hyperparams,
                [
                    Pytree.const(f"object_pose_{o_id}")
                    for o_id in object_ids
                ],
                posterior_across_frames
            )
            viz_trace(trace, t=viz_index+T+1)
            print(get_new_state(trace), '\n')
        viz_index += FINAL_T+1

        write_json(pred_file,
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
    args = parser.parse_args()
    scenario = args.scenario

    # paths for reading physion metadata
    data_path = "/home/haoliangwang/data/"
    hdf5_file_path = os.path.join(
        data_path,
        "physion_hdf5",
    )
    mesh_file_path = os.path.join(
        data_path,
        "all_flex_meshes/core",
    )
    save_path = "/home/haoliangwang/data/b3d_tracking_results/test"

    pred_file_path = "/home/haoliangwang/data/pred_files/gt_info/gt.json"
    main(
        hdf5_file_path,
        scenario,
        mesh_file_path,
        pred_file_path,
        save_path,
    )
