import os
import uuid
from os import listdir
from os.path import isfile, join

buggy_stims = [
    "pilot-containment-cone-plate_0017",
    "pilot-containment-cone-plate_0022",
    "pilot-containment-cone-plate_0029",
    "pilot-containment-cone-plate_0034",
    "pilot-containment-multi-bowl_0042",
    "pilot-containment-multi-bowl_0048",
    "pilot-containment-vase_torus_0031",
    "pilot_dominoes_SJ020_d3chairs_o1plants_tdwroom_0005",
    "pilot_it2_collision_non-sphere_box_0002",
    "pilot_it2_collision_non-sphere_tdw_1_dis_1_occ_0004",
    "pilot_it2_collision_non-sphere_tdw_1_dis_1_occ_0007",
    "pilot_it2_drop_simple_box_0000",
    "pilot_it2_drop_simple_box_0042",
    "pilot_it2_drop_simple_tdw_1_dis_1_occ_0003",
    "pilot_it2_rollingSliding_simple_collision_box_0008",
    "pilot_it2_rollingSliding_simple_collision_box_large_force_0009",
    "pilot_it2_rollingSliding_simple_collision_tdw_1_dis_1_occ_0002",
    "pilot_it2_rollingSliding_simple_ledge_tdw_1_dis_1_occ_0021",
    "pilot_it2_rollingSliding_simple_ledge_tdw_1_dis_1_occ_sphere_small_zone_0022",
    "pilot_it2_rollingSliding_simple_ramp_box_small_zone_0006",
    "pilot_it2_rollingSliding_simple_ramp_tdw_1_dis_1_occ_small_zone_0004",
    "pilot_it2_rollingSliding_simple_ramp_tdw_1_dis_1_occ_small_zone_0017",
    "pilot_linking_nl1-8_mg000_aCyl_bCyl_tdwroom1_long_a_0022",
    "pilot_linking_nl1-8_mg000_aCylcap_bCyl_tdwroom1_0012",
    "pilot_linking_nl1-8_mg000_aCylcap_bCyl_tdwroom_small_rings_0006",
    "pilot_linking_nl1-8_mg000_aCylcap_bCyl_tdwroom_small_rings_0010",
    "pilot_linking_nl1-8_mg000_aCylcap_bCyl_tdwroom_small_rings_0029",
    "pilot_linking_nl1-8_mg000_aCylcap_bCyl_tdwroom_small_rings_0036",
    "pilot_linking_nl6_aNone_bCone_occ1_dis1_boxroom_0028",
    "pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom_stable_0000",
    "pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom_stable_0002",
    "pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom_stable_0003",
    "pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom_stable_0010",
    "pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom_stable_0013",
    "pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom_stable_0017",
    "pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom_stable_0018",
    "pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom_stable_0032",
    "pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom_stable_0036",
    "pilot_towers_nb4_fr015_SJ000_gr01_mono0_dis1_occ1_tdwroom_unstable_0021",
    "pilot_towers_nb4_fr015_SJ000_gr01_mono0_dis1_occ1_tdwroom_unstable_0041",
    "pilot_towers_nb5_fr015_SJ030_mono0_dis0_occ0_boxroom_unstable_0006",
    "pilot_towers_nb5_fr015_SJ030_mono0_dis0_occ0_boxroom_unstable_0009",
]

data_path = "/home/haoliangwang/data/"
hdf5_file_path = os.path.join(
    data_path,
    "physion_hdf5",
)

for scenario in ["dominoes", "support", "contain", "link"]:
    scenario_path = join(hdf5_file_path, scenario + "_all_movies")
    onlyhdf5 = [
        f
        for f in listdir(scenario_path)
        if isfile(join(scenario_path, f)) and join(scenario_path, f).endswith(".hdf5")
    ]

    if scenario == "dominoes":
        START_T = 14
    else:
        START_T = 0

    if scenario == "collide":
        FINAL_T = 15
    else:
        FINAL_T = 45

    recording_id = uuid.uuid4()
    viz_index = 0
    for trial_index, hdf5_file in enumerate(onlyhdf5):
        trial_name = hdf5_file[:-5]
        print(trial_index + 1, "\t", trial_name)
        # if trial_name in buggy_stims or trial_name in ["pilot_dominoes_4midRM1_boxroom_0026", "pilot_dominoes_2mid_J020R15_d3chairs_o1plants_tdwroom_2_0005", "pilot_dominoes_2mid_J025R30_tdwroom_0025"]:
        #     continue
        if (
            trial_name != "pilot_dominoes_4mid_tdwroom_0003"
        ):  # "pilot_dominoes_0mid_d3chairs_o1plants_tdwroom_0001": #
            continue
        os.system(
            f"python /home/haoliangwang/b3d/test_b3d_tracking_hmm_single.py --scenario {scenario} --trial_name {trial_name} --recording_id {recording_id} --viz_index {viz_index}"
        )
        viz_index += FINAL_T - START_T
