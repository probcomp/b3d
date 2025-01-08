import os
from os import listdir
from os.path import isfile, join
import rerun as rr
import uuid

data_path = "/home/haoliangwang/data/"
hdf5_file_path = os.path.join(
    data_path,
    "physion_hdf5",
)

for scenario in ['collide', 'drop', 'roll', 'dominoes', 'support', 'link', 'contain']:
    scenario_path = join(hdf5_file_path, scenario + "_all_movies")
    onlyhdf5 = [
        f
        for f in listdir(scenario_path)
        if isfile(join(scenario_path, f)) and join(scenario_path, f).endswith(".hdf5")
    ]

    if scenario == "collide":
        FINAL_T = 15
    else:
        FINAL_T = 45

    recording_id = uuid.uuid4()
    viz_index = 0
    for trial_index, hdf5_file in enumerate(onlyhdf5):
        trial_name = hdf5_file[:-5]
        if trial_name != 'pilot_it2_collision_assorted_targets_box_0003':
            continue
        print(trial_index + 1, "\t", trial_name)
        os.system(f"python /home/haoliangwang/b3d/test_b3d_tracking_hmm_single.py --scenario {scenario} --trial_name {trial_name} --recording_id {recording_id} --viz_index {viz_index}")
        viz_index += FINAL_T+1