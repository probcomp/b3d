import os
from os import listdir
from os.path import isfile, join


scenario = 'collide'
data_path = "/home/haoliangwang/data/"
hdf5_file_path = os.path.join(
    data_path,
    "physion_hdf5",
)
scenario_path = join(hdf5_file_path, scenario + "_all_movies")
onlyhdf5 = [
    f
    for f in listdir(scenario_path)
    if isfile(join(scenario_path, f)) and join(scenario_path, f).endswith(".hdf5")
]

for trial_index, hdf5_file in enumerate(onlyhdf5):
    trial_name = hdf5_file[:-5]
    if trial_name in ["pilot_it2_collision_non-sphere_tdw_1_dis_1_occ_0039", "pilot_it2_collision_yeet_tdw_1_dis_1_occ_0025", "pilot_it2_collision_non-sphere_tdw_1_dis_1_occ_0025", "pilot_it2_collision_non-sphere_box_0003", "pilot_it2_collision_simple_box_1_dis_1_occ_0014", "pilot_it2_collision_simple_box_1_dis_1_occ_0034", "pilot_it2_collision_tiny_ball_box_0023", "pilot_it2_collision_yeet_tdw_1_dis_1_occ_0038"]:
        continue
    print(trial_index + 1, "\t", trial_name)
    os.system(f"python /home/haoliangwang/b3d/test_b3d_tracking_hmm_single.py --scenario {scenario} --trial_name {trial_name}")
