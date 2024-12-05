import argparse
import collections
import io
import itertools
import json
import os
import random
from copy import deepcopy
from functools import reduce
from os import listdir
from os.path import isfile, join

import b3d
import b3d.bayes3d as bayes3d
import b3d.chisight.dense.dense_model
import b3d.chisight.dense.likelihoods.laplace_likelihood
import genjax
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import rerun as rr
import trimesh
from genjax import Pytree
from PIL import Image
from scipy.spatial.transform import Rotation


def mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def scale_mesh(vertices, scale_factor):
    vertices_copy = deepcopy(vertices)
    vertices_copy[:, 0] *= scale_factor[0]
    vertices_copy[:, 1] *= scale_factor[1]
    vertices_copy[:, 2] *= scale_factor[2]
    return vertices_copy


def quaternion_to_euler_angles(quaternion):
    rot = Rotation.from_quat(quaternion)
    rot_euler = rot.as_euler("xyz", degrees=True)
    return rot_euler


def blackout_image(depth_map, area):
    # zero_depth_map = np.ones(depth_map.shape)
    zero_depth_map = np.zeros(depth_map.shape)
    zero_depth_map[area] = depth_map[area]
    return zero_depth_map


def get_mask_area(seg_img, colors):
    arrs = []
    for color in colors:
        arr = seg_img == color
        arr = arr.min(-1).astype("float32")
        arr = arr.reshape((arr.shape[-1], arr.shape[-1])).astype(bool)
        arrs.append(arr)
    return reduce(np.logical_or, arrs)


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
    use_gt=False,
    masked=True,
):
    def initial_samples(num_sample):
        trunc = [0, 1]
        if num_sample > np.power(len(trunc), len(object_ids)):
            num_sample = np.power(len(trunc), len(object_ids))
        all_comb = [trunc for _ in range(len(object_ids))]
        product = [element for element in itertools.product(*all_comb)]
        product.remove(tuple([0 for _ in range(len(object_ids))]))
        samples = random.sample(product, num_sample - 1)
        samples.append(tuple([0 for _ in range(len(object_ids))]))
        return samples

    def sample_from_posterior(num_sample, log_probs_categories, option="rank"):
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
        for _ in range(num_sample):
            sample = draw_single_sample()
            samples.append(sample)
        return samples

    def re_weight(pose_scale_mesh_list, offset):
        def _re_weight(pose_scale_mesh, offset):
            pose_list = []
            scale_list = []
            obj_names = {}
            for o_id, val in pose_scale_mesh.items():
                pose_list.append((f"object_pose_{o_id}", val[0]))
                for i, j in enumerate(val[1]):
                    scale_list.append((f"object_scale_{o_id}_{i}", j))
                    obj_names[f"object_name_{o_id}_{i}"] = val[-1][i]

            pre_dict = (
                [
                    ("camera_pose", camera_pose),
                    ("depth_noise_variance", 0.01),
                    ("color_noise_variance", 1),
                    ("outlier_probability", 0.1),
                ]
                + pose_list
                + scale_list
            )

            if masked:
                pre_dict += [
                    ("rgbd", blackout_image(rgbds[START_T], all_areas[START_T]))
                ]
            else:
                pre_dict += [("rgbd", rgbds[START_T])]
            choice_map = dict(pre_dict)

            trace, _ = importance_jit(
                jax.random.PRNGKey(0),
                genjax.ChoiceMap.d(choice_map),
                (
                    Pytree.const([o_id for o_id in pose_scale_mesh.keys()]),
                    [val[2] for val in pose_scale_mesh.values()],
                    likelihood_args,
                ),
            )

            # scales_scores = {}
            best_pose_scale = {}
            key = jax.random.PRNGKey(0)
            for iter in range(num_inference_step):
                for o_id in pose_scale_mesh.keys():
                    trace, key, _, _ = bayes3d.enumerate_and_select_best_move_pose(
                        trace, Pytree.const((f"object_pose_{o_id}",)), key, all_deltas
                    )
                    if iter == num_inference_step - 1:
                        best_pose_scale[f"object_pose_{o_id}"] = trace.get_choices()[
                            f"object_pose_{o_id}"
                        ]
                    components = [
                        item[0]
                        for item in scale_list
                        if item[0].startswith(f"object_scale_{o_id}")
                    ]
                    for component in components:
                        trace, key, posterior_scales, scores = (
                            bayes3d.enumerate_and_select_best_move_scale(
                                trace, Pytree.const((component,)), key, scale_deltas
                            )
                        )
                        if iter == num_inference_step - 1:
                            best_pose_scale[component] = trace.get_choices()[component]
                            # scales_scores[component] = [
                            #     (score.astype(float).item(), posterior_scale)
                            #     for (score, posterior_scale) in zip(
                            #         scores, posterior_scales
                            #     )
                            # ]
                viz_trace(trace, iter + offset, cloud=True)
            return trace.get_score().item(), best_pose_scale, obj_names

        re_weighted_samples = []
        for pose_scale_mesh in pose_scale_mesh_list:
            best_score, best_pose_scale, obj_names = _re_weight(pose_scale_mesh, offset)
            re_weighted_samples.append(
                [
                    best_score,
                    best_pose_scale,
                    Pytree.const([o_id for o_id in pose_scale_mesh.keys()]),
                    dict([(o_id, val[2]) for o_id, val in pose_scale_mesh.items()]),
                    obj_names,
                ]
            )
        return re_weighted_samples

    def get_object_id_from_composite_id(feature):
        if int(feature.split("_")[-2]) == base_id:
            o_id = composite_mapping["_".join(feature.split("_")[-2:])]
        else:
            o_id = feature.split("_")[-2]
        return o_id

    def get_composite_id_from_object_id(feature):
        if feature in list(reversed_composite_mapping.keys()):
            o_id = reversed_composite_mapping[str(feature)]
        else:
            o_id = str(feature) + "_0"
        return o_id

    def get_all_component_poses(
        best_mc_obj_cat_sample,
        pose_samples_from_posterior,
        composite_scales,
        reversed_composite_mapping,
    ):
        best_mc_obj_cat_sample[3][base_id]
        composite_ids = list(reversed_composite_mapping.keys())[1:]
        for composite_id in composite_ids:
            pose_samples_from_posterior[composite_id] = [[]]
        for i, base_pose in enumerate(pose_samples_from_posterior[base_id]):
            assert len(pose_samples_from_posterior[base_id]) == len(
                composite_scales[base_id]
            )
            best_base_pose = base_pose[-1]
            top = (
                best_mc_obj_cat_sample[3][base_id][0].vertices[:, 1].max()
                * composite_scales[base_id][i]["y"]
            )
            for j, composite_id in enumerate(composite_ids):
                pose = b3d.Pose.from_translation(jnp.array([0.0, top, 0.0]))
                pose_samples_from_posterior[composite_id][0].append(base_pose[0] @ pose)
                pose_samples_from_posterior[composite_id].append(best_base_pose @ pose)
                top += (
                    best_mc_obj_cat_sample[3][base_id][j + 1].vertices[:, 1].max()
                    * composite_scales[composite_id][i]["y"]
                )

    def get_posterior_poses_for_frame(frame, num_sample, posterior_across_frames):
        pose_samples_from_posterior = {}
        for o_id, poses in posterior_across_frames["pose"][frame].items():
            best_pose = poses[1]
            pose_samples_from_posterior[o_id] = [
                [pose for pose in sample_from_posterior(num_sample, poses[0])],
                best_pose,
            ]
            if o_id == base_id:
                get_all_component_poses(
                    best_mc_obj_cat_sample,
                    pose_samples_from_posterior,
                    json_file["scale"],
                    {value: feature for feature, value in composite_mapping.items()},
                )
        return pose_samples_from_posterior

    def compute_velocity(
        mesh,
        scale,
        object_pose_last_frame,
        object_pose_window_frame,
        dt,
    ):
        def compute_center_of_mass(object_pose):
            q = object_pose._quaternion
            bounding_box_transform = bounding_box_b3d.transform(object_pose)
            bounding_box_transform = trimesh.Trimesh(
                bounding_box_transform.vertices, bounding_box_transform.faces
            )
            center_of_mass = bounding_box_transform.center_mass
            return center_of_mass, q

        mesh_tri = trimesh.Trimesh(
            vertices=scale_mesh(mesh.vertices, scale), faces=mesh.faces
        )
        oriented_bbox = mesh_tri.bounding_box
        bounding_box_b3d = b3d.Mesh.from_trimesh(oriented_bbox)

        pos_now, q_now = compute_center_of_mass(object_pose_last_frame)
        pos_last, q_last = compute_center_of_mass(object_pose_window_frame)
        linear_vel = (pos_now - pos_last) / dt

        angular_velocity = (2 / dt) * np.array(
            [
                q_last[0] * q_now[1]
                - q_last[1] * q_now[0]
                - q_last[2] * q_now[3]
                + q_last[3] * q_now[2],
                q_last[0] * q_now[2]
                + q_last[1] * q_now[3]
                - q_last[2] * q_now[0]
                - q_last[3] * q_now[1],
                q_last[0] * q_now[3]
                - q_last[1] * q_now[2]
                + q_last[2] * q_now[1]
                - q_last[3] * q_now[0],
            ]
        )
        ang_vel = np.dot(
            angular_velocity, np.array([[0, 0, 1], [0, -1, 0], [-1, 0, 0]])
        )

        return {"x": linear_vel[0], "y": linear_vel[1], "z": linear_vel[2]}, {
            "x": ang_vel[0],
            "y": ang_vel[1],
            "z": ang_vel[2],
        }

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
    num_initial_sample = 1
    fps = 100
    smoothing_window_size = 5
    num_pose_grid = 11
    position_search_thr = 0.1
    # needs to be odd
    num_scale_grid = 11
    scale_search_thr = 0.2
    num_sample_from_posterior = 20
    num_inference_step = 5

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

    # Gridding on translation only.
    translation_deltas = b3d.Pose.concatenate_poses(
        [
            jax.vmap(lambda p: b3d.Pose.from_translation(p))(
                jnp.stack(
                    jnp.meshgrid(
                        jnp.linspace(
                            -position_search_thr, position_search_thr, num_pose_grid
                        ),
                        jnp.linspace(
                            -position_search_thr, position_search_thr, num_pose_grid
                        ),
                        jnp.linspace(
                            -position_search_thr, position_search_thr, num_pose_grid
                        ),
                    ),
                    axis=-1,
                ).reshape(-1, 3)
            ),
            b3d.Pose.identity()[None, ...],
        ]
    )
    # Sample orientations from a VMF to define a "grid" over orientations.
    rotation_deltas = b3d.Pose.concatenate_poses(
        [
            jax.vmap(b3d.Pose.sample_gaussian_vmf_pose, in_axes=(0, None, None, None))(
                jax.random.split(
                    jax.random.PRNGKey(0), num_pose_grid * num_pose_grid * num_pose_grid
                ),
                b3d.Pose.identity(),
                0.0001,
                100.0,
            ),
            b3d.Pose.identity()[None, ...],
        ]
    )
    all_deltas = b3d.Pose.stack_poses([translation_deltas, rotation_deltas])

    scale_deltas = jnp.stack(
        jnp.meshgrid(
            jnp.linspace(1 - scale_search_thr, 1 + scale_search_thr, num_scale_grid),
            jnp.linspace(1 - scale_search_thr, 1 + scale_search_thr, num_scale_grid),
            jnp.linspace(1 - scale_search_thr, 1 + scale_search_thr, num_scale_grid),
        ),
        axis=-1,
    ).reshape(-1, 3)

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
    model, viz_trace, _ = b3d.chisight.dense.dense_model.make_dense_multiobject_model(
        renderer, likelihood_func
    )
    importance_jit = jax.jit(model.importance)

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

    vis_index = 0
    for hdf5_file in onlyhdf5:
        trial_name = hdf5_file[:-5]
        print("\t", trial_name)
        hdf5_file_path = join(scenario_path, hdf5_file)

        depth_arr = []
        image_arr = []
        seg_arr = []
        (
            base_id,
            attachment_id,
            use_attachment,
            use_base,
            use_cap,
            cap_id,
        ) = None, None, None, None, None, None
        composite_mapping = {}
        with h5py.File(hdf5_file_path, "r") as f:
            # extract depth info
            for frame in f["frames"].keys():
                depth = jnp.array(f["frames"][frame]["images"]["_depth_cam0"])
                depth_arr.append(depth)
                image = jnp.array(
                    Image.open(io.BytesIO(f["frames"][frame]["images"]["_img_cam0"][:]))
                )
                image_arr.append(image)
                im_seg = np.array(
                    Image.open(io.BytesIO(f["frames"][frame]["images"]["_id_cam0"][:]))
                )
                seg_arr.append(im_seg)
            depth_arr = jnp.asarray(depth_arr)
            image_arr = jnp.asarray(image_arr) / 255
            seg_arr = jnp.asarray(seg_arr)

            # extract camera info
            camera_matrix = np.array(
                f["frames"]["0000"]["camera_matrices"]["camera_matrix_cam0"]
            ).reshape((4, 4))

            # extract object info
            object_ids = np.array(f["static"]["object_ids"])
            model_names = np.array(f["static"]["model_names"])
            object_segmentation_colors = np.array(
                f["static"]["object_segmentation_colors"]
            )
            assert (
                len(object_ids) == len(model_names) == len(object_segmentation_colors)
            )

            distractors = (
                np.array(f["static"]["distractors"])
                if np.array(f["static"]["distractors"]).size != 0
                else []
            )
            occluders = (
                np.array(f["static"]["occluders"])
                if np.array(f["static"]["occluders"]).size != 0
                else []
            )
            distractors_occluders = np.concatenate([distractors, occluders])
            if len(distractors_occluders):
                object_ids = object_ids[: -len(distractors_occluders)]
                model_names = model_names[: -len(distractors_occluders)]
                object_segmentation_colors = object_segmentation_colors[
                    : -len(distractors_occluders)
                ]
            # distractor_ids = np.concatenate([np.where(model_names==distractor)[0] for distractor in distractors], axis=0).tolist() if distractors else []
            # distractor_ids = np.concatenate([np.where(model_names==distractor)[0] for distractor in distractors], axis=0).tolist() if distractors else []
            # occluder_ids = np.concatenate([np.where(model_names==occluder)[0] for occluder in occluders], axis=0).tolist() if occluders else []
            # excluded_model_ids = distractor_ids+occluder_ids
            # included_model_ids = [idx for idx in range(len(object_ids)) if idx not in excluded_model_ids]
            # object_ids = included_model_ids

            if "use_base" in np.array(f["static"]):
                use_base = np.array(f["static"]["use_base"])
                if use_base:
                    base_id = np.array(f["static"]["base_id"])
                    assert base_id.size == 1
                    base_id = base_id.item()
                    composite_mapping[f"{base_id}_0"] = base_id
            if "use_attachment" in np.array(f["static"]):
                use_attachment = np.array(f["static"]["use_attachment"])
                if use_attachment:
                    attachment_id = np.array(f["static"]["attachment_id"])
                    assert attachment_id.size == 1
                    attachment_id = attachment_id.item()
                    composite_mapping[f"{base_id}_1"] = attachment_id
                    if "use_cap" in np.array(f["static"]):
                        use_cap = np.array(f["static"]["use_cap"])
                        if use_cap:
                            cap_id = attachment_id + 1
                            composite_mapping[f"{base_id}_1"] = cap_id

        reversed_composite_mapping = dict(
            [(value, feature) for feature, value in composite_mapping.items()]
        )
        rgbds = jnp.concatenate(
            [image_arr, jnp.reshape(depth_arr, depth_arr.shape + (1,))], axis=-1
        )

        R = camera_matrix[:3, :3]
        T = camera_matrix[0:3, 3]
        a = np.array([-R[0, :], -R[1, :], -R[2, :]])
        b = np.array(T)
        camera_position_from_matrix = np.linalg.solve(a, b)
        camera_rotation_from_matrix = -np.transpose(R)
        camera_pose = b3d.Pose(
            camera_position_from_matrix,
            b3d.Rot.from_matrix(camera_rotation_from_matrix).as_quat(),
        )

        pred_file = pred_file_all[trial_name]
        if use_gt:
            gt_info = pred_file["scene"][0]["objects"]
            for i in range(len(gt_info)):
                for feature in pred_file["scene"][0]["objects"][i].keys():
                    pred_file["scene"][0]["objects"][i][feature] = [
                        pred_file["scene"][0]["objects"][i][feature]
                    ]

        # using the 0th frame
        pred = pred_file["scene"][0]["objects"]
        pose_scale_mesh_list = []
        for sample in initial_samples(num_initial_sample):
            assert len(sample) == len(object_ids)
            pose_scale_mesh = {}
            for i, (o_id, color, idx) in enumerate(
                zip(object_ids, object_segmentation_colors, sample)
            ):
                area = get_mask_area(seg_arr[START_T], [color])
                object_colors = rgbds[START_T][..., 0:3][area]
                mean_object_colors = np.mean(object_colors, axis=0)
                assert not np.isnan(mean_object_colors).any()
                pose_scale_mesh[o_id] = (
                    b3d.Pose(
                        jnp.array(pred[i]["location"][idx]),
                        jnp.array(pred[i]["rotation"][idx]),
                    ),
                    [jnp.array(pred[i]["scale"][idx])],
                    [
                        b3d.Mesh(
                            ordered_all_meshes[pred[i]["type"][idx]].vertices,
                            ordered_all_meshes[pred[i]["type"][idx]].faces,
                            jnp.ones(
                                ordered_all_meshes[pred[i]["type"][idx]].vertices.shape
                            )
                            * mean_object_colors,
                        )
                    ],
                    [pred[i]["type"][idx]],
                )
            pose_scale_mesh_list.append(pose_scale_mesh)

        rgbds = jax.image.resize(
            rgbds,
            (rgbds.shape[0], im_height, im_width, *rgbds.shape[3:]),
            method="linear",
        )
        im_segs = jax.image.resize(
            seg_arr,
            (seg_arr.shape[0], im_height, im_width, *seg_arr.shape[3:]),
            method="linear",
        )
        all_areas = []
        for im_seg in im_segs:
            all_area = np.any(im_seg != jnp.array([0, 0, 0]), axis=-1)
            all_areas.append(all_area)

        mc_obj_cat_samples = re_weight(pose_scale_mesh_list, vis_index)
        vis_index += num_inference_step
        best_mc_obj_cat_sample = max(mc_obj_cat_samples, key=lambda x: x[0])
        pre_dict = [
            ("camera_pose", camera_pose),
            ("depth_noise_variance", 0.01),
            ("color_noise_variance", 1),
            ("outlier_probability", 0.1),
        ] + [(feature, val) for feature, val in best_mc_obj_cat_sample[1].items()]
        if masked:
            pre_dict += [("rgbd", blackout_image(rgbds[START_T], all_areas[START_T]))]
        else:
            pre_dict += [("rgbd", rgbds[START_T])]
        choice_map = dict(pre_dict)

        trace_post_obj_cat_inference, _ = importance_jit(
            jax.random.PRNGKey(0),
            genjax.ChoiceMap.d(choice_map),
            (
                best_mc_obj_cat_sample[2],
                list(best_mc_obj_cat_sample[3].values()),
                likelihood_args,
            ),
        )

        trace = trace_post_obj_cat_inference
        key = jax.random.PRNGKey(0)
        posterior_across_frames = {"pose": []}
        for T_observed_image in range(FINAL_T):
            posterior_across_frames["pose"].append({})
            # Constrain on new RGB and Depth data.
            if masked:
                trace = b3d.update_choices(
                    trace,
                    Pytree.const(("rgbd",)),
                    blackout_image(
                        rgbds[T_observed_image], all_areas[T_observed_image]
                    ),
                )
            else:
                trace = b3d.update_choices(
                    trace,
                    Pytree.const(("rgbd",)),
                    rgbds[T_observed_image],
                )
            for o_id in best_mc_obj_cat_sample[2].const:
                trace, key, posterior_poses, scores = (
                    bayes3d.enumerate_and_select_best_move_pose(
                        trace, Pytree.const((f"object_pose_{o_id}",)), key, all_deltas
                    )
                )
                posterior_across_frames["pose"][-1][int(o_id)] = [
                    [
                        (score, posterior_pose)
                        for (posterior_pose, score) in zip(posterior_poses, scores)
                    ],
                    trace.get_choices()[f"object_pose_{o_id}"],
                ]
            viz_trace(trace, vis_index, cloud=True)
            vis_index += 1
        vis_index += 1

        # prepare the json file to write
        json_file = {}
        json_file["model"] = {}
        json_file["scale"] = {}

        for feature, val in best_mc_obj_cat_sample[1].items():
            if feature.startswith("object_pose"):
                continue
            id_long = feature
            o_id = get_object_id_from_composite_id(id_long)
            json_file["model"][int(o_id)] = [
                best_mc_obj_cat_sample[-1][id_long.replace("scale", "name")]
                for _ in range(num_sample_from_posterior)
            ]
            json_file["scale"][int(o_id)] = [
                {
                    "x": val[0].astype(float).item(),
                    "y": val[1].astype(float).item(),
                    "z": val[2].astype(float).item(),
                }
                for _ in range(num_sample_from_posterior)
            ]

        # optim_scales = {}
        # for item in best_mc_obj_cat_sample[1]:
        #     key = item[0]
        #     scale = item[1]
        #     if key.startswith("object_scale_"):
        #         o_id = get_object_id_from_composite_id(key)
        #         optim_scales[int(o_id)] = scale

        pose_samples_from_posterior_last_frame = get_posterior_poses_for_frame(
            -1,
            num_sample_from_posterior,
            posterior_across_frames,
        )
        pose_samples_from_posterior_window_frame = get_posterior_poses_for_frame(
            -(smoothing_window_size + 1),
            num_sample_from_posterior,
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
                            "y": pose._position[1].astype(float).item()
                            if pose._position[1].astype(float).item() >= 0
                            else 0,
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
                            "x": quaternion_to_euler_angles(pose._quaternion)[0].astype(
                                float
                            ),
                            "y": quaternion_to_euler_angles(pose._quaternion)[1].astype(
                                float
                            ),
                            "z": quaternion_to_euler_angles(pose._quaternion)[2].astype(
                                float
                            ),
                        }
                        for pose in poses[0]
                    ],
                )
                for o_id, poses in pose_samples_from_posterior_last_frame.items()
            ]
        )

        velocity_dict = dict(
            [
                (
                    int(o_id),
                    [
                        compute_velocity(
                            best_mc_obj_cat_sample[3][
                                int(get_composite_id_from_object_id(o_id).split("_")[0])
                            ][int(get_composite_id_from_object_id(o_id).split("_")[1])],
                            best_mc_obj_cat_sample[1][
                                f"object_scale_{get_composite_id_from_object_id(o_id)}"
                            ],
                            pose_samples_from_posterior_last_frame[o_id][0][i],
                            pose_samples_from_posterior_window_frame[o_id][
                                -1
                            ],  # using optim pose for window frame
                            smoothing_window_size / fps,
                        )
                        for i in range(num_sample_from_posterior)
                    ],
                )
                for o_id in pose_samples_from_posterior_last_frame.keys()
            ]
        )
        linear_velocity_dict = dict(
            [
                (o_id, [value[i][0] for i in range(num_sample_from_posterior)])
                for o_id, value in velocity_dict.items()
            ]
        )
        angular_velocity_dict = dict(
            [
                (o_id, [value[i][1] for i in range(num_sample_from_posterior)])
                for o_id, value in velocity_dict.items()
            ]
        )

        json_file["position"] = position_dict
        json_file["rotation"] = rotation_dict
        json_file["velocity"] = linear_velocity_dict
        json_file["angular_velocity"] = angular_velocity_dict

        missing = find_missing_values(object_ids)
        for feature, val in json_file.items():
            for o_id in missing:
                json_file[feature][o_id] = val[object_ids[0]]

        mkdir(f"{save_path}/{scenario}/")
        with open(
            f"{save_path}/{scenario}/{trial_name}.json",
            "w",
        ) as f:
            json.dump(json_file, f)


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
    save_path = "/home/haoliangwang/data/b3d_tracking_results"
    pred_file_path = "/home/haoliangwang/data/pred_files/clip_b3d_results/pose_scale_cat_using_first.json"
    main(
        hdf5_file_path, scenario, mesh_file_path, save_path, pred_file_path, masked=True
    )
