import argparse
import collections
import itertools
import json
import os
import random
from copy import deepcopy
from os import listdir
from os.path import isfile, join

import b3d
import b3d.bayes3d as bayes3d
import b3d.chisight.dense.dense_model
import b3d.chisight.dense.likelihoods.laplace_likelihood
import b3d.chisight.gen3d.inference.inference as inference
import b3d.chisight.gen3d.settings as settings
from b3d.chisight.gen3d.dataloading import (
    get_initial_state,
    load_trial,
    resize_rgbds_and_get_masks,
)
import genjax
import jax
import jax.numpy as jnp
import numpy as np
import rerun as rr
import trimesh
from genjax import Pytree


def mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def scale_mesh(vertices, scale_factor):
    vertices_copy = deepcopy(vertices)
    vertices_copy[:, 0] *= scale_factor[0]
    vertices_copy[:, 1] *= scale_factor[1]
    vertices_copy[:, 2] *= scale_factor[2]
    return vertices_copy


def blackout_image(depth_map, area):
    # zero_depth_map = np.ones(depth_map.shape)
    zero_depth_map = np.zeros(depth_map.shape)
    zero_depth_map[area] = depth_map[area]
    return zero_depth_map

def find_missing_values(nums):
    full_range = set(range(min(nums), max(nums) + 1))
    missing_values = sorted(list(full_range - set(nums)))
    return missing_values


def main(
    hdf5_file_path,
    scenario,
    mesh_file_path,
    save_path,
    pred_file_path,
    all_scale="first_scale",
    use_gt=False,
    masked=True,
    debug=False,
):
    rr.init("demo")
    rr.connect("127.0.0.1:8812")
    rr.log("/", rr.ViewCoordinates.LEFT_HAND_Y_UP, static=True)

    near_plane = 0.1
    far_plane = 100
    im_width = 350
    im_height = 350
    width = 1024
    height = 1024

    START_T = 0
    FINAL_T = 15
    all_scale = True if all_scale == "all_scale" else False
    if use_gt:
        all_scale = False
    print("all_scale: ", all_scale)

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
    vfov = 54.43222 / 180.0 * np.pi
    tan_half_vfov = np.tan(vfov / 2.0)
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

    b3d.reload(b3d.chisight.dense.dense_model)
    b3d.reload(b3d.chisight.dense.likelihoods.laplace_likelihood)
    likelihood_func = b3d.chisight.dense.likelihoods.laplace_likelihood.likelihood_func

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
        "interp_penalty": Pytree.const(1000),
    }

    for trial_index, hdf5_file in enumerate(onlyhdf5):
        if trial_index != 0:
            continue

        trial_name = hdf5_file[:-5]
        print("\t", trial_index + 1, "\t", trial_name)
        hdf5_file_path = join(scenario_path, hdf5_file)
        
        pred_file = pred_file_all[trial_name]
        if use_gt:
            gt_info = pred_file["scene"][0]["objects"]
            for i in range(len(gt_info)):
                for feature in pred_file["scene"][0]["objects"][i].keys():
                    pred_file["scene"][0]["objects"][i][feature] = [
                        pred_file["scene"][0]["objects"][i][feature]
                    ]
        rgbds, seg_arr, object_ids, object_segmentation_colors, camera_pose, composite_mapping, reversed_composite_mapping = load_trial(hdf5_file_path)

        inference_hyperparams = b3d.chisight.gen3d.settings.inference_hyperparams
        hyperparams = settings.hyperparams
        hyperparams["camera_pose"] = camera_pose
        hyperparams["likelihood_args"] = likelihood_args
        
        initial_state, hyperparams = get_initial_state(pred_file, object_ids, object_segmentation_colors, ordered_all_meshes, seg_arr[START_T], rgbds[START_T], hyperparams)
        rgbds, all_areas = resize_rgbds_and_get_masks(rgbds, seg_arr, im_height, im_width)
        
        key = jax.random.PRNGKey(156)
        trace = inference.get_initial_trace(
            key, renderer, likelihood_func, hyperparams, initial_state, blackout_image(rgbds[START_T], all_areas[START_T])
        )
        for T in range(FINAL_T):
            key = b3d.split_key(key)
            trace, _ = inference.inference_step(
                key,
                trace,
                blackout_image(
                        rgbds[T], all_areas[T]
                    ),
                inference_hyperparams,
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="collide", type=str)
    parser.add_argument("--clip", default="first", type=str)
    parser.add_argument("--all_scale", default="first_scale", type=str)
    args = parser.parse_args()
    scenario = args.scenario
    clip = args.clip
    all_scale = args.all_scale

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

    print(f"***************{clip}***************")
    # save_path = f"/home/haoliangwang/data/b3d_tracking_results/{all_scale}/{clip}"
    # pred_file_path = f"/home/haoliangwang/data/pred_files/clip_b3d_results/pose_scale_cat_using_{clip}.json"
    # main(hdf5_file_path, scenario, mesh_file_path, save_path, pred_file_path, all_scale)
    save_path = "/home/haoliangwang/data/b3d_tracking_results/gt_all_info"
    pred_file_path = "/home/haoliangwang/data/pred_files/gt_info/gt.json"
    main(
        hdf5_file_path,
        scenario,
        mesh_file_path,
        save_path,
        pred_file_path,
        use_gt=True,
        debug=True,
    )
