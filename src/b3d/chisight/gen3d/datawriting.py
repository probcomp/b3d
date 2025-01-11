from scipy.spatial.transform import Rotation
from copy import deepcopy
import trimesh
import numpy as np
import os
import json

import b3d


NUM_SAMPLE_FROM_POSTERIOR = 20
SMOOTHING_WINDOW_SIZE = 3
FPS = 100
STATIC_POSITION_THRESHHOLD = 0.007
STATIC_ROTATION_THRESHHOLD = 0.001

def mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def find_missing_values(nums):
    full_range = set(range(min(nums), max(nums) + 1))
    missing_values = sorted(list(full_range - set(nums)))
    return missing_values


def compute_linear_velocity(
    mesh,
    scale,
    object_pose_last_frame,
    object_pose_window_frame,
    dt,
):
    def compute_center_of_mass(mesh, object_pose):
        mesh_transform = mesh.transform(object_pose)
        mesh_transform_tri = trimesh.Trimesh(
            mesh_transform.vertices, mesh_transform.faces
        )
        center_of_mass = mesh_transform_tri.center_mass
        return center_of_mass

    mesh = b3d.Mesh(
        vertices=scale_mesh(mesh.vertices, scale),
        faces=mesh.faces,
        vertex_attributes=None,
    )
    pos_now = compute_center_of_mass(mesh, object_pose_last_frame)
    pos_last = compute_center_of_mass(mesh, object_pose_window_frame)
    linear_vel = (pos_now - pos_last) / dt
    return {"x": linear_vel[0], "y": linear_vel[1], "z": linear_vel[2]}


def compute_angular_velocity(q1, q2, dt):
    """
    Compute angular velocity in radians per second from two quaternions.

    Parameters:
        q1 (array-like): Quaternion at the earlier time [w, x, y, z].
        q2 (array-like): Quaternion at the later time [w, x, y, z].
        dt (float): Time difference between the two quaternions.

    Returns:
        angular_velocity (numpy array): Angular velocity vector (radians per second).
    """
    # Convert quaternions to scipy Rotation objects
    rot1 = Rotation.from_quat(q1)
    rot2 = Rotation.from_quat(q2)

    # Compute the relative rotation
    relative_rotation = rot2 * rot1.inv()

    # Convert the relative rotation to angle-axis representation
    angle = relative_rotation.magnitude()  # Rotation angle in radians
    axis = (
        relative_rotation.as_rotvec() / angle if angle != 0 else np.zeros(3)
    )  # Rotation axis

    # Compute angular velocity
    angular_velocity = (axis * angle) / dt
    return {
        "x": angular_velocity[0].astype(float).item(),
        "y": angular_velocity[1].astype(float).item(),
        "z": angular_velocity[2].astype(float).item(),
    }


def scale_mesh(vertices, scale_factor):
    vertices_copy = deepcopy(vertices)
    vertices_copy[:, 0] *= scale_factor['x']
    vertices_copy[:, 1] *= scale_factor['y']
    vertices_copy[:, 2] *= scale_factor['z']
    return vertices_copy


# def get_object_id_from_composite_id(feature):
#         if int(feature.split("_")[-2]) == base_id:
#             o_id = composite_mapping["_".join(feature.split("_")[-2:])]
#         else:
#             o_id = feature.split("_")[-2]
#         return o_id

# def get_composite_id_from_object_id(feature):
#     if feature in list(reversed_composite_mapping.keys()):
#         o_id = reversed_composite_mapping[str(feature)]
#     else:
#         o_id = str(feature) + "_0"
#     return o_id

# def get_all_component_poses(
#     best_mc_obj_cat_sample,
#     pose_samples_from_posterior,
#     composite_scales,
#     reversed_composite_mapping,
# ):
#     best_mc_obj_cat_sample[3][base_id]
#     composite_ids = list(reversed_composite_mapping.keys())[1:]
#     for composite_id in composite_ids:
#         pose_samples_from_posterior[composite_id] = [[]]
#     for i, base_pose in enumerate(pose_samples_from_posterior[base_id]):
#         assert len(pose_samples_from_posterior[base_id]) == len(
#             composite_scales[base_id]
#         )
#         best_base_pose = base_pose[-1]
#         top = (
#             best_mc_obj_cat_sample[3][base_id][0].vertices[:, 1].max()
#             * composite_scales[base_id][i]["y"]
#         )
#         for j, composite_id in enumerate(composite_ids):
#             pose = b3d.Pose.from_translation(jnp.array([0.0, top, 0.0]))
#             pose_samples_from_posterior[composite_id][0].append(base_pose[0] @ pose)
#             pose_samples_from_posterior[composite_id].append(best_base_pose @ pose)
#             top += (
#                 best_mc_obj_cat_sample[3][base_id][j + 1].vertices[:, 1].max()
#                 * composite_scales[composite_id][i]["y"]
#             )

def sample_from_posterior(log_probs_categories, option="rank"):
        log_probs = [item[0] for item in log_probs_categories]
        categories = [item[1] for item in log_probs_categories]
        num_categories = len(log_probs)

        if option == "uniform":
            def draw_single_sample():
                index = np.random.choice(num_categories)
                return categories[index]
        elif option == "veridical":
            def draw_single_sample():
                # see this: https://stackoverflow.com/questions/58339083/how-to-sample-from-a-log-probability-distribution
                gumbels = np.random.gumbel(size=num_categories)
                index = np.argmax(log_probs + gumbels)
                return categories[index]
        elif option == "rank":
            def draw_single_sample():
                weights = np.array([1 / (n + 1) for n in range(num_categories)])
                weights_norm = weights / weights.sum()
                index = np.random.choice(num_categories, p=weights_norm)
                return categories[index]
        elif option == "mix":
            def draw_single_sample():
                t = 0.5
                t * np.array(log_probs) + (1 - t) * (1 / num_categories)
                return
        else:
            raise NotImplementedError

        samples = []
        np.random.seed(42)
        for _ in range(NUM_SAMPLE_FROM_POSTERIOR):
            sample = draw_single_sample()
            samples.append(sample)
        return samples

def get_posterior_poses_for_frame(frame, posterior_across_frames):
    pose_samples_from_posterior = {}
    for o_id, poses in posterior_across_frames["pose"][frame].items():
        best_pose = poses[1]
        pose_samples_from_posterior[o_id] = [
            [pose for pose in sample_from_posterior(poses[0])],
            best_pose,
        ]
        # if o_id == base_id:
        #     get_all_component_poses(
        #         best_mc_obj_cat_sample,
        #         pose_samples_from_posterior,
        #         json_file["scale"],
        #         {value: feature for feature, value in composite_mapping.items()},
        #     )
    return pose_samples_from_posterior


def write_json(pred_file, hyperparams, posterior_across_frames, save_path, scenario, trial_name, debug=False):
    pred = pred_file["scene"][0]["objects"]

    # prepare the json file to write
    json_file = {}
    json_file["model"] = {}
    json_file["scale"] = {}

    for i, o_id in enumerate(hyperparams["object_ids"].unwrap()):
        json_file["model"][int(o_id)] = [
            pred[i]["type"][0]
            for _ in range(NUM_SAMPLE_FROM_POSTERIOR)
        ]
        json_file["scale"][int(o_id)] = [
            {
                "x": pred[i]["scale"][0][0],
                "y": pred[i]["scale"][0][1],
                "z": pred[i]["scale"][0][2],
            }
            for _ in range(NUM_SAMPLE_FROM_POSTERIOR)
        ]

    pose_samples_from_posterior_last_frame = get_posterior_poses_for_frame(
        -1,
        posterior_across_frames,
    )
    pose_samples_from_posterior_window_frame = get_posterior_poses_for_frame(
        -(SMOOTHING_WINDOW_SIZE + 1),
        posterior_across_frames,
    )
    assert len(pose_samples_from_posterior_last_frame) == len(
        pose_samples_from_posterior_window_frame
    )

    position_dict = dict(
        [
            (
                int(o_id),
                [
                    {
                        "x": pose._position[0].astype(float).item(),
                        "y": pose._position[1].astype(float).item(),
                        "z": pose._position[2].astype(float).item(),
                    }
                    for pose in poses[0]
                ],
            )
            for o_id, poses in pose_samples_from_posterior_last_frame.items()
        ]
    )
    rotation_dict = dict(
        [
            (
                int(o_id),
                [
                    {
                        "x": pose._quaternion[0].astype(float).item(),
                        "y": pose._quaternion[1].astype(float).item(),
                        "z": pose._quaternion[2].astype(float).item(),
                        "w": pose._quaternion[3].astype(float).item(),
                    }
                    for pose in poses[0]
                ],
            )
            for o_id, poses in pose_samples_from_posterior_last_frame.items()
        ]
    )

    linear_velocity_dict = {}
    linear_velocity_dict_optim = {}
    for o_id in pose_samples_from_posterior_last_frame.keys():
        if np.allclose(pose_samples_from_posterior_last_frame[o_id][-1]._position, pose_samples_from_posterior_window_frame[o_id][-1]._position, atol=STATIC_POSITION_THRESHHOLD*SMOOTHING_WINDOW_SIZE):
            linear_velocity_dict[int(o_id)] = [{"x": 0, "y": 0, "z": 0} for _ in range(NUM_SAMPLE_FROM_POSTERIOR)]
            linear_velocity_dict_optim[int(o_id)] = [{"x": 0, "y": 0, "z": 0} for _ in range(NUM_SAMPLE_FROM_POSTERIOR)]
        else:
            linear_velocity_dict[int(o_id)] = [
                compute_linear_velocity(
                    hyperparams["meshes"][int(o_id)][0],
                    json_file["scale"][o_id][i],
                    pose_samples_from_posterior_last_frame[o_id][0][i],
                    pose_samples_from_posterior_window_frame[o_id][-1],  # using optim pose for window frame
                    SMOOTHING_WINDOW_SIZE / FPS,
                )
                for i in range(NUM_SAMPLE_FROM_POSTERIOR)]
            linear_velocity_dict_optim[int(o_id)] = [
                compute_linear_velocity(
                    hyperparams["meshes"][int(o_id)][0],
                    json_file["scale"][o_id][i],
                    pose_samples_from_posterior_last_frame[o_id][-1],
                    pose_samples_from_posterior_window_frame[o_id][-1],  # using optim pose for window frame
                    SMOOTHING_WINDOW_SIZE / FPS,
                )
                for i in range(NUM_SAMPLE_FROM_POSTERIOR)]
            

    angular_velocity_dict = {}
    angular_velocity_dict_optim = {}
    for o_id in pose_samples_from_posterior_last_frame.keys():
        if np.allclose(pose_samples_from_posterior_last_frame[o_id][-1]._quaternion, pose_samples_from_posterior_window_frame[o_id][-1]._quaternion, atol=STATIC_ROTATION_THRESHHOLD*SMOOTHING_WINDOW_SIZE):
            angular_velocity_dict[int(o_id)] = [{"x": 0, "y": 0, "z": 0} for _ in range(NUM_SAMPLE_FROM_POSTERIOR)]
            angular_velocity_dict_optim[int(o_id)] = [{"x": 0, "y": 0, "z": 0} for _ in range(NUM_SAMPLE_FROM_POSTERIOR)]
        else:
            angular_velocity_dict[int(o_id)] = [
                compute_angular_velocity(
                    pose_samples_from_posterior_window_frame[o_id][-1]._quaternion,  # using optim pose for window frame
                    pose_samples_from_posterior_last_frame[o_id][0][i]._quaternion,
                    SMOOTHING_WINDOW_SIZE / FPS,
                )
                for i in range(NUM_SAMPLE_FROM_POSTERIOR)
            ]
            angular_velocity_dict_optim[int(o_id)] = [
                compute_angular_velocity(
                    pose_samples_from_posterior_window_frame[o_id][-1]._quaternion,  # using optim pose for window frame
                    pose_samples_from_posterior_last_frame[o_id][-1]._quaternion,
                    SMOOTHING_WINDOW_SIZE / FPS,
                )
                for i in range(NUM_SAMPLE_FROM_POSTERIOR)
            ]
    json_file["position"] = position_dict
    json_file["rotation"] = rotation_dict
    json_file["velocity"] = linear_velocity_dict
    json_file["angular_velocity"] = angular_velocity_dict

    missing = find_missing_values(np.array([int(item) for item in hyperparams["object_ids"].unwrap()]))
    for feature, val in json_file.items():
        for o_id in missing:
            json_file[feature][o_id] = val[int(hyperparams["object_ids"].unwrap()[0])]

    json_file_optim = deepcopy(json_file)
    json_file_optim['velocity'] = linear_velocity_dict_optim
    json_file_optim['angular_velocity'] = angular_velocity_dict_optim

    mkdir(f"{save_path}/{scenario}/")
    with open(f"{save_path}/{scenario}/{trial_name}.json", "w") as f:
        json.dump(json_file, f)

    mkdir(f"{save_path}/{scenario}_optim/")
    with open(f"{save_path}/{scenario}_optim/{trial_name}.json", "w") as f:
        json.dump(json_file_optim, f)

    if debug:
        for frame_index, frame_info in enumerate(posterior_across_frames["pose"]):
            for o_id, o_id_info in frame_info.items():
                posterior_across_frames["pose"][frame_index][o_id][1] = (
                    o_id_info[1]._position.astype(float).tolist(),
                    o_id_info[1]._quaternion.astype(float).tolist(),
                )
                for j, rank in enumerate(o_id_info[0]):
                    posterior_across_frames["pose"][frame_index][o_id][0][j] = (
                        rank[0].astype(float).item(),
                        rank[1]._position.astype(float).tolist(),
                        rank[1]._quaternion.astype(float).tolist(),
                        )
        mkdir(f"{save_path}/{scenario}_verbose/")
        with open(f"{save_path}/{scenario}_verbose/{trial_name}.json", "w") as f:
            json.dump(posterior_across_frames, f)
    return