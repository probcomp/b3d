import collections
import io
import itertools
import json
import os
import random
from functools import reduce

import b3d
import b3d.bayes3d as bayes3d
import b3d.chisight.dense.dense_model
import b3d.chisight.dense.likelihoods.laplace_likelihood
import genjax
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import trimesh
from genjax import Pytree
from PIL import Image


def mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


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


def main(
    hdf5_file_path,
    mesh_file_path,
    save_path,
    pred_file_path,
    trial_name="pilot_it2_collision_non-sphere_box_0023",
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

    def re_weight(pose_scale_mesh_list):
        def _re_weight(pose_scale_mesh):
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
            for iter_first in range(num_inference_step):
                for o_id in pose_scale_mesh.keys():
                    trace, key, _, _ = bayes3d.enumerate_and_select_best_move_pose(
                        trace, Pytree.const((f"object_pose_{o_id}",)), key, all_deltas
                    )
                    if iter_first == num_inference_step - 1:
                        best_pose_scale[f"object_pose_{o_id}"] = trace.get_choices()[
                            f"object_pose_{o_id}"
                        ]
                    components = [
                        item[0]
                        for item in scale_list
                        if item[0].startswith(f"object_scale_{o_id}")
                    ]
                    for component in components:
                        if use_gt:
                            best_pose_scale[component] = trace.get_choices()[component]
                            continue
                        trace, key, posterior_scales, scores = (
                            bayes3d.enumerate_and_select_best_move_scale(
                                trace, Pytree.const((component,)), key, scale_deltas
                            )
                        )
                        if iter_first == num_inference_step - 1:
                            best_pose_scale[component] = trace.get_choices()[component]
                            # scales_scores[component] = [
                            #     (score.astype(float).item(), posterior_scale)
                            #     for (score, posterior_scale) in zip(
                            #         scores, posterior_scales
                            #     )
                            # ]
            return trace.get_score().item(), best_pose_scale, obj_names

        re_weighted_samples = []
        for pose_scale_mesh in pose_scale_mesh_list:
            best_score, best_pose_scale, obj_names = _re_weight(pose_scale_mesh)
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

    near_plane = 0.1
    far_plane = 100
    im_width = 350
    im_height = 350
    width = 1024
    height = 1024

    START_T = 0
    FINAL_T = 15
    num_initial_sample = 1
    num_pose_grid = 11
    position_search_thr = 0.1
    # needs to be odd
    num_scale_grid = 11
    scale_search_thr = 0.2
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
                0.01,
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
    model, _, _ = b3d.chisight.dense.dense_model.make_dense_multiobject_model(
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

    hdf5_file_path = (
        f"/home/haoliangwang/data/physion_hdf5/collide_all_movies/{trial_name}.hdf5"
    )

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
        object_segmentation_colors = np.array(f["static"]["object_segmentation_colors"])
        assert len(object_ids) == len(model_names) == len(object_segmentation_colors)

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

    mc_obj_cat_samples = re_weight(pose_scale_mesh_list)
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
    for T_observed_image in range(FINAL_T):
        print(f"frame {T_observed_image}")
        # Constrain on new RGB and Depth data.
        if masked:
            trace = b3d.update_choices(
                trace,
                Pytree.const(("rgbd",)),
                blackout_image(rgbds[T_observed_image], all_areas[T_observed_image]),
            )
        else:
            trace = b3d.update_choices(
                trace,
                Pytree.const(("rgbd",)),
                rgbds[T_observed_image],
            )

        for o_id in best_mc_obj_cat_sample[2].const:
            trace, key, _, _ = bayes3d.enumerate_and_select_best_move_pose(
                trace,
                Pytree.const((f"object_pose_{o_id}",)),
                key,
                all_deltas,
            )
        ret_vals = trace.get_retval()["likelihood_args"]
        rgb = np.asarray(ret_vals["latent_rgbd"][..., :3]) * 255
        im = Image.fromarray(rgb.astype(np.uint8))
        im.save(f"{save_path}/{trial_name}_{T_observed_image}.png")
    print(trace.get_choices()["object_pose_1"])


if __name__ == "__main__":
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

    save_path = "/home/haoliangwang/data/b3d_tracking_results/b3d_track_imgs"
    mkdir(save_path)
    pred_file_path = "/home/haoliangwang/data/pred_files/gt_info/gt.json"
    main(
        hdf5_file_path,
        mesh_file_path,
        save_path,
        pred_file_path,
        use_gt=True,
    )
