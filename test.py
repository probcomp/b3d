import json
import os
from os.path import join
import trimesh
from b3d.chisight.gen3d.dataloading import (
    get_initial_state,
    load_trial,
)
import b3d.chisight.gen3d.settings as settings


scenario = 'drop'
trial_name = 'pilot_it2_drop_all_bowls_box_0002'

hdf5_file_path = "/ccn2/u/rmvenkat/data/testing_physion/regenerate_from_old_commit/test_humans_consolidated/lf_0"
mesh_file_path = "/ccn2/u/rmvenkat/data/all_flex_meshes/"
pred_file_path = "/ccn2/u/haw027/b3d_ipe/pred_files/gt_info/gt_correct.json"

with open(pred_file_path) as f:
    pred_file_all = json.load(f)
pred_file = pred_file_all[trial_name]

all_meshes = {}
for path, dirs, files in os.walk(mesh_file_path):
    for name in files + dirs:
        if name.endswith(".obj"):
            mesh = trimesh.load(os.path.join(path, name))
            all_meshes[name[:-4]] = mesh


hdf5_file_path = join(
    hdf5_file_path,
    scenario + "_all_movies",
    f"{trial_name}.hdf5",
)

(
    rgbds_original,
    seg_arr_original,
    object_ids,
    object_segmentation_colors,
    background_areas,
    camera_pose,
) = load_trial(hdf5_file_path, 15)

hyperparams = settings.hyperparams

initial_state, hyperparams = get_initial_state(
    pred_file,
    object_ids,
    object_segmentation_colors,
    all_meshes,
    seg_arr_original[0],
    rgbds_original[0],
    hyperparams,
)

from b3d.physics.physics_utils import step

stepped_model, stepped_state = step(initial_state["prev_model"], initial_state["prev_state"], hyperparams["sim_dt"])