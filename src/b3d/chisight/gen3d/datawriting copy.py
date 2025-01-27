import json
import os
from copy import deepcopy

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree

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


def compute_center_of_mass(mesh, object_pose):
    mesh_transform = mesh.transform(object_pose)
    mesh_transform_tri = trimesh.Trimesh(
        mesh_transform.vertices, mesh_transform.faces
    )
    center_of_mass = mesh_transform_tri.center_mass
    return center_of_mass


def compute_linear_velocity(
    mesh,
    # scale,
    object_pose_last_frame,
    object_pose_window_frame,
    dt,
):
    # mesh = b3d.Mesh(
    #     vertices=mesh.vertices,
    #     faces=mesh.faces,
    #     vertex_attributes=None,
    # )
    pos_now = compute_center_of_mass(mesh, object_pose_last_frame)
    pos_last = compute_center_of_mass(mesh, object_pose_window_frame)
    # print("center now: ", pos_now)
    # print("center last: ", pos_last)
    # print("dt: ", dt)
    linear_vel = (pos_now - pos_last) / dt
    # print("linear_vel: ", linear_vel)
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
    vertices_copy[:, 0] *= scale_factor["x"]
    vertices_copy[:, 1] *= scale_factor["y"]
    vertices_copy[:, 2] *= scale_factor["z"]
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


# def get_posterior_poses_for_frame(frame, posterior_across_frames):
#     pose_samples_from_posterior = {}
#     for o_id, poses in posterior_across_frames["pose"][frame].items():
#         best_pose = poses[1]
#         pose_samples_from_posterior[o_id] = [
#             [pose for pose in sample_from_posterior(poses[0])],
#             best_pose,
#         ]
#         # if o_id == base_id:
#         #     get_all_component_poses(
#         #         best_mc_obj_cat_sample,
#         #         pose_samples_from_posterior,
#         #         json_file["scale"],
#         #         {value: feature for feature, value in composite_mapping.items()},
#         #     )
#     return pose_samples_from_posterior


def get_last_appearance(posterior_across_frames, o_id, start, stop, step):
    for index in range(start, stop, step):
        if o_id in list(posterior_across_frames["pose"][index].keys()):
            return index
    return None


def get_posterior_poses_for_frame_for_object(posterior_across_frames, o_id, frame, c2f_level=0):
    for idx, poses in posterior_across_frames["pose"][frame].items():
        if o_id == idx:
            # print("yes!!")
            # poses = posterior_across_frames["pose"][frame][o_id]
            best_pose = poses[-1]
            pose_samples_from_posterior = [
                    [pose for pose in sample_from_posterior(poses[0][c2f_level])],
                    best_pose,
                ]
            return pose_samples_from_posterior


def write_json(
    pred_file,
    hyperparams,
    posterior_across_frames,
    save_path,
    scenario,
    trial_name,
    debug=False,
):
    pred = pred_file["scene"][0]["objects"]

    # prepare the json file to write
    json_file = {}
    json_file["model"] = {}
    json_file["scale"] = {}

    for i, o_id in enumerate(hyperparams["object_ids"].unwrap()):
        json_file["model"][int(o_id)] = [
            pred[str(o_id)]["type"][0] for _ in range(NUM_SAMPLE_FROM_POSTERIOR)
        ]
        json_file["scale"][int(o_id)] = [
            {
                "x": pred[str(o_id)]["scale"][0][0],
                "y": pred[str(o_id)]["scale"][0][1],
                "z": pred[str(o_id)]["scale"][0][2],
            }
            for _ in range(NUM_SAMPLE_FROM_POSTERIOR)
        ]

    # pose_samples_from_posterior_last_frame = get_posterior_poses_for_frame(
    #     -1,
    #     posterior_across_frames,
    # )
    # pose_samples_from_posterior_window_frame = get_posterior_poses_for_frame(
    #     -(SMOOTHING_WINDOW_SIZE + 1),
    #     posterior_across_frames,
    # )
    # assert len(pose_samples_from_posterior_last_frame) == len(
    #     pose_samples_from_posterior_window_frame
    # )

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
                    for pose in get_posterior_poses_for_frame_for_object(posterior_across_frames, o_id, get_last_appearance(posterior_across_frames, o_id, len(posterior_across_frames["pose"]) - 1, -1, -1))[0]
                ],
            )
            for o_id in hyperparams["object_ids"].unwrap()
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
                    for pose in get_posterior_poses_for_frame_for_object(posterior_across_frames, o_id, get_last_appearance(posterior_across_frames, o_id, len(posterior_across_frames["pose"]) - 1, -1, -1))[0]
                ],
            )
            for o_id in hyperparams["object_ids"].unwrap()
        ]
    )

    linear_velocity_dict = {}
    linear_velocity_dict_optim = {}
    for o_id in hyperparams["object_ids"].unwrap():
        # print("object: ", o_id)
        if o_id not in list(posterior_across_frames["pose"][-1].keys()):
            linear_velocity_dict[int(o_id)] = [{"x": 0.0, "y": 0.0, "z": 0.0} for _ in range(NUM_SAMPLE_FROM_POSTERIOR)]
            linear_velocity_dict_optim[int(o_id)] = [{"x": 0.0, "y": 0.0, "z": 0.0} for _ in range(NUM_SAMPLE_FROM_POSTERIOR)]
        else:
            # start_smooth_frame = get_last_appearance(posterior_across_frames, o_id, len(posterior_across_frames["pose"])-(SMOOTHING_WINDOW_SIZE + 1), len(posterior_across_frames["pose"]), 1)
            # print("start_smooth_frame: ", get_last_appearance(posterior_across_frames, o_id, len(posterior_across_frames["pose"])-2, -1, -1))
            anchor_pt = get_posterior_poses_for_frame_for_object(posterior_across_frames, o_id, get_last_appearance(posterior_across_frames, o_id, len(posterior_across_frames["pose"])-2, -1, -1))
            sample_pt = get_posterior_poses_for_frame_for_object(posterior_across_frames, o_id, -1, c2f_level=1)
            linear_velocity_dict[int(o_id)] = [
                compute_linear_velocity(
                    hyperparams["meshes"][int(o_id)],
                    # json_file["scale"][int(o_id)][i],
                    sample_pt[0][i],
                    anchor_pt[-1],  # using optim pose for window frame
                    1 / FPS,
                )
                for i in range(NUM_SAMPLE_FROM_POSTERIOR)
            ]
            linear_velocity_dict_optim[int(o_id)] = [
                compute_linear_velocity(
                    hyperparams["meshes"][int(o_id)],
                    # json_file["scale"][int(o_id)][i],
                    sample_pt[-1],
                    anchor_pt[-1],  # using optim pose for window frame
                    1 / FPS,
                )
                for i in range(NUM_SAMPLE_FROM_POSTERIOR)
            ]
            # if start_smooth_frame == len(posterior_across_frames["pose"])-1:
            #     anchor_pt = get_posterior_poses_for_frame_for_object(posterior_across_frames, o_id, get_last_appearance(posterior_across_frames, o_id, len(posterior_across_frames["pose"])-(SMOOTHING_WINDOW_SIZE + 2), -1, -1))[-1]
            #     sample_pt = get_posterior_poses_for_frame_for_object(posterior_across_frames, o_id, -1)[0]
            #     linear_velocity_dict[int(o_id)] = [
            #         compute_linear_velocity(
            #             hyperparams["meshes"][int(o_id)],
            #             # json_file["scale"][int(o_id)][i],
            #             sample_pt[i],
            #             anchor_pt,  # using optim pose for window frame
            #             1 / FPS,
            #         )
            #         for i in range(NUM_SAMPLE_FROM_POSTERIOR)
            #     ]
            # else:
            #     anchor_pt = get_posterior_poses_for_frame_for_object(posterior_across_frames, o_id, start_smooth_frame)[-1]
            #     sample_pt = get_posterior_poses_for_frame_for_object(posterior_across_frames, o_id, -1)[0]
            #     linear_velocity_dict[int(o_id)] = [
            #         compute_linear_velocity(
            #             hyperparams["meshes"][int(o_id)],
            #             # json_file["scale"][int(o_id)][i],
            #             sample_pt[i],
            #             anchor_pt,  # using optim pose for window frame
            #             (len(posterior_across_frames["pose"])-start_smooth_frame-1) / FPS,
            #         )
            #         for i in range(NUM_SAMPLE_FROM_POSTERIOR)
            #     ]

            # print("anchor_pt: ", anchor_pt)
            # print("sample_pt: ", sample_pt)
            
    angular_velocity_dict = {}
    angular_velocity_dict_optim = {}
    for o_id in hyperparams["object_ids"].unwrap():
        if o_id not in list(posterior_across_frames["pose"][-1].keys()):
            angular_velocity_dict[int(o_id)] = [{"x": 0.0, "y": 0.0, "z": 0.0} for _ in range(NUM_SAMPLE_FROM_POSTERIOR)]
            angular_velocity_dict_optim[int(o_id)] = [{"x": 0.0, "y": 0.0, "z": 0.0} for _ in range(NUM_SAMPLE_FROM_POSTERIOR)]
        else:
            anchor_pt = get_posterior_poses_for_frame_for_object(posterior_across_frames, o_id, get_last_appearance(posterior_across_frames, o_id, len(posterior_across_frames["pose"])-2, -1, -1))
            sample_pt = get_posterior_poses_for_frame_for_object(posterior_across_frames, o_id, -1, c2f_level=1)
            angular_velocity_dict[int(o_id)] = [
                compute_angular_velocity(
                    anchor_pt[-1]._quaternion,
                    sample_pt[0][i]._quaternion,
                    1 / FPS,
                )
                for i in range(NUM_SAMPLE_FROM_POSTERIOR)
            ]
            angular_velocity_dict_optim[int(o_id)] = [
                compute_angular_velocity(
                    anchor_pt[-1]._quaternion,
                    sample_pt[-1]._quaternion,
                    1 / FPS,
                )
                for i in range(NUM_SAMPLE_FROM_POSTERIOR)
            ]
            # start_smooth_frame = get_last_appearance(posterior_across_frames, o_id, len(posterior_across_frames["pose"])-(SMOOTHING_WINDOW_SIZE + 1), len(posterior_across_frames["pose"]), 1)
            # if start_smooth_frame == len(posterior_across_frames["pose"])-1:
            #     anchor_pt = get_posterior_poses_for_frame_for_object(posterior_across_frames, o_id, get_last_appearance(posterior_across_frames, o_id, len(posterior_across_frames["pose"])-(SMOOTHING_WINDOW_SIZE + 2), -1, -1))[-1]
            #     sample_pt = get_posterior_poses_for_frame_for_object(posterior_across_frames, o_id, -1)[0]
            #     angular_velocity_dict[int(o_id)] = [
            #         compute_angular_velocity(
            #             anchor_pt._quaternion,
            #             sample_pt[i]._quaternion,
            #             1 / FPS,
            #         )
            #         for i in range(NUM_SAMPLE_FROM_POSTERIOR)
            #     ]
            # else:
            #     anchor_pt = get_posterior_poses_for_frame_for_object(posterior_across_frames, o_id, start_smooth_frame)[-1]
            #     sample_pt = get_posterior_poses_for_frame_for_object(posterior_across_frames, o_id, -1)[0]
            #     angular_velocity_dict[int(o_id)] = [
            #         compute_angular_velocity(
            #             anchor_pt._quaternion,
            #             sample_pt[i]._quaternion,
            #             (len(posterior_across_frames["pose"])-start_smooth_frame-1) / FPS,
            #         )
            #         for i in range(NUM_SAMPLE_FROM_POSTERIOR)
            #     ]

    json_file["position"] = position_dict
    json_file["rotation"] = rotation_dict
    json_file["velocity"] = linear_velocity_dict
    json_file["angular_velocity"] = angular_velocity_dict

    missing = find_missing_values(
        np.array([int(item) for item in hyperparams["object_ids"].unwrap()])
    )
    for feature, val in json_file.items():
        for o_id in missing:
            json_file[feature][o_id] = val[int(hyperparams["object_ids"].unwrap()[0])]

    mkdir(f"{save_path}/{scenario}/")
    with open(f"{save_path}/{scenario}/{trial_name}.json", "w") as f:
        json.dump(json_file, f)

    json_file_optim = deepcopy(json_file)
    json_file_optim["velocity"] = linear_velocity_dict_optim
    json_file_optim["angular_velocity"] = angular_velocity_dict_optim
    mkdir(f"{save_path}/{scenario}_optim/")
    with open(f"{save_path}/{scenario}_optim/{trial_name}.json", "w") as f:
        json.dump(json_file_optim, f)


    if debug:
        for frame_index, frame_info in enumerate(posterior_across_frames["pose"]):
            for o_id, o_id_info in frame_info.items():
                posterior_across_frames["pose"][frame_index][o_id][-1] = (
                    o_id_info[-1]._position.astype(float).tolist() + o_id_info[-1]._quaternion.astype(float).tolist() + compute_center_of_mass(hyperparams["meshes"][int(o_id)], o_id_info[-1]).astype(float).tolist(),
                )
                for i, c2f_level in enumerate(o_id_info[0]):
                    for j, rank in enumerate(c2f_level):
                        posterior_across_frames["pose"][frame_index][o_id][0][i][j] = (
                            rank[0].astype(float).item(),
                            rank[1]._position.astype(float).tolist() + rank[1]._quaternion.astype(float).tolist() + compute_center_of_mass(hyperparams["meshes"][int(o_id)], rank[1]).astype(float).tolist(),
                        )
        mkdir(f"{save_path}/{scenario}_verbose/")
        with open(f"{save_path}/{scenario}_verbose/{trial_name}.json", "w") as f:
            json.dump(posterior_across_frames, f)
    return


# measure errors
def apply_transform(pose: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    return (pose[:3, :3] @ vertices.T + pose[:3, 3][:, None]).T


def add_err(pred_pose: np.ndarray, gt_pose: np.ndarray, vertices: np.ndarray) -> float:
    """Compute the Average Distance (ADD) error between the predicted pose and the
    ground truth pose, given the vertices of the object.

    References:
    - https://github.com/thodan/bop_toolkit/blob/59c5f486fe3a7886329d9fc908935e40d3bc0248/bop_toolkit_lib/pose_error.py#L210-L224
    - https://github.com/NVlabs/FoundationPose/blob/cd3ca4bc080529c53d5e5235212ca476d82bccf7/Utils.py#L232-L240
    - https://github.com/chensong1995/HybridPose/blob/106c86cddaa52765eb82f17bd00fdc72b98a02ca/lib/utils.py#L36-L49

    Args:
        pred_pose (np.ndarray): A 4x4 transformation matrix representing the predicted pose.
        gt_pose (np.ndarray): A 4x4 transformation matrix representing the ground truth pose.
        vertices (np.ndarray): The vertices of shape (num_vertices, 3) in the object frame,
            representing the 3D model of the object. Note that we should be using the vertices
            from the ground truth mesh file instead of the reconstructed point cloud.
    """
    pred_locs = apply_transform(pred_pose, vertices)
    gt_locs = apply_transform(gt_pose, vertices)
    return np.linalg.norm(pred_locs - gt_locs, axis=-1).mean()


def adds_err(pred_pose: np.ndarray, gt_pose: np.ndarray, vertices: np.ndarray) -> float:
    """Compute the Average Closest Point Distance (ADD-S) error between the predicted pose and the
    ground truth pose, given the vertices of the object. ADD-S is an ambiguity-invariant pose
    error metric which takes care of both symmetric and non-symmetric objects

    References:
    - https://github.com/thodan/bop_toolkit/blob/59c5f486fe3a7886329d9fc908935e40d3bc0248/bop_toolkit_lib/pose_error.py#L227-L247
    - https://github.com/NVlabs/FoundationPose/blob/cd3ca4bc080529c53d5e5235212ca476d82bccf7/Utils.py#L242-L253
    - https://github.com/chensong1995/HybridPose/blob/106c86cddaa52765eb82f17bd00fdc72b98a02ca/lib/utils.py#L51-L68

    Args:
        pred_pose (np.ndarray): A 4x4 transformation matrix representing the predicted pose.
        gt_pose (np.ndarray): A 4x4 transformation matrix representing the ground truth pose.
        vertices (np.ndarray): The vertices of shape (num_vertices, 3) in the object frame,
            representing the 3D model of the object. Note that we should be using the vertices
            from the ground truth mesh file instead of the reconstructed point cloud.
    """
    pred_locs = apply_transform(pred_pose, vertices)
    gt_locs = apply_transform(gt_pose, vertices)

    # Calculate distances to the nearest neighbors from vertices in the
    # ground-truth pose to vertices in the estimated pose.
    nn_index = cKDTree(pred_locs)
    nn_dists, _ = nn_index.query(gt_locs, k=1)

    return nn_dists.mean()