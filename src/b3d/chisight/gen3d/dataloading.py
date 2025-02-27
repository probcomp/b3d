import io
from functools import reduce
from copy import deepcopy

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from genjax import Pytree
from PIL import Image
import warp as wp
from warp.sim.model import Model
import warp.sim.render

import b3d


def scale_mesh(vertices, scale_factor):
    vertices_copy = deepcopy(vertices)
    vertices_copy[:, 0] *= scale_factor[0]
    vertices_copy[:, 1] *= scale_factor[1]
    vertices_copy[:, 2] *= scale_factor[2]
    return vertices_copy


def load_trial(hdf5_file_path, FINAL_T):
    depth_arr = []
    image_arr = []
    seg_arr = []
    # (
    #     base_id,
    #     attachment_id,
    #     use_attachment,
    #     use_base,
    #     use_cap,
    #     cap_id,
    # ) = None, None, None, None, None, None
    # composite_mapping = {}
    with h5py.File(hdf5_file_path, "r") as f:
        # extract depth info
        for frame in f["frames"].keys():
            if int(frame) >= FINAL_T:
                break
            depth = jnp.array(f["frames"][frame]["images"]["_depth_cam0"])
            depth_arr.append(depth)
            image = jnp.array(
                Image.open(io.BytesIO(f["frames"][frame]["images"]["_img_cam0"][:]))
            )
            image_arr.append(image)
            im_seg = jnp.array(
                Image.open(io.BytesIO(f["frames"][frame]["images"]["_id_cam0"][:]))
            )
            seg_arr.append(im_seg)
        depth_arr = jnp.asarray(depth_arr)
        image_arr = jnp.asarray(image_arr) / 255
        seg_arr = jnp.asarray(seg_arr)

        # extract camera info
        camera_matrix = jnp.array(
            f["frames"]["0000"]["camera_matrices"]["camera_matrix_cam0"]
        ).reshape((4, 4))

        # extract object info
        object_ids = np.array(f["static"]["object_ids"])
        object_segmentation_colors = jnp.array(
            f["static"]["object_segmentation_colors"]
        )
        assert len(object_ids) == len(object_segmentation_colors)

        background_areas = np.zeros(
            (image_arr.shape[0], image_arr.shape[1], image_arr.shape[2])
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
            background_areas = jnp.asarray(
                [
                    get_mask_area(
                        seg, object_segmentation_colors[-len(distractors_occluders) :]
                    )
                    for seg in seg_arr
                ]
            )
            object_ids = object_ids[: -len(distractors_occluders)]
            object_segmentation_colors = object_segmentation_colors[
                : -len(distractors_occluders)
            ]

    #     if "use_base" in np.array(f["static"]):
    #         use_base = np.array(f["static"]["use_base"])
    #         if use_base:
    #             base_id = np.array(f["static"]["base_id"])
    #             assert base_id.size == 1
    #             base_id = base_id.item()
    #             composite_mapping[f"{base_id}_0"] = base_id
    #     if "use_attachment" in np.array(f["static"]):
    #         use_attachment = np.array(f["static"]["use_attachment"])
    #         if use_attachment:
    #             attachment_id = np.array(f["static"]["attachment_id"])
    #             assert attachment_id.size == 1
    #             attachment_id = attachment_id.item()
    #             composite_mapping[f"{base_id}_1"] = attachment_id
    #             if "use_cap" in np.array(f["static"]):
    #                 use_cap = np.array(f["static"]["use_cap"])
    #                 if use_cap:
    #                     cap_id = attachment_id + 1
    #                     composite_mapping[f"{base_id}_1"] = cap_id

    # reversed_composite_mapping = dict(
    #     [(value, feature) for feature, value in composite_mapping.items()]
    # )
    rgbds = jnp.concatenate(
        [image_arr, jnp.reshape(depth_arr, depth_arr.shape + (1,))], axis=-1
    )

    R = camera_matrix[:3, :3]
    T = camera_matrix[0:3, 3]
    a = jnp.array([-R[0, :], -R[1, :], -R[2, :]])
    b = jnp.array(T)
    camera_position_from_matrix = jnp.linalg.solve(a, b)
    camera_rotation_from_matrix = -jnp.transpose(R)
    camera_pose = b3d.Pose(
        camera_position_from_matrix,
        b3d.Rot.from_matrix(camera_rotation_from_matrix).as_quat(),
    )
    return (
        rgbds,
        seg_arr,
        object_ids,
        object_segmentation_colors,
        background_areas,
        camera_pose,
        # composite_mapping,
        # reversed_composite_mapping,
    )


def get_mask_area(seg_img, colors):
    arrs = []
    for color in colors:
        arr = seg_img == color
        arr = arr.min(-1).astype("float32")
        arr = arr.reshape((arr.shape[-1], arr.shape[-1])).astype(bool)
        arrs.append(arr)
    return reduce(jnp.logical_or, arrs)


def resize_rgbds_and_get_masks(rgbds, seg_arr, background_areas, im_height, im_width):
    rgbds = jax.image.resize(
        rgbds,
        (rgbds.shape[0], im_height, im_width, *rgbds.shape[3:]),
        method="linear",
    )
    background_areas = jax.image.resize(
        background_areas,
        (background_areas.shape[0], im_height, im_width),
        method="linear",
    ).astype(bool)
    im_segs = jax.image.resize(
        seg_arr,
        (seg_arr.shape[0], im_height, im_width, *seg_arr.shape[3:]),
        method="linear",
    )
    all_areas = []
    for im_seg in im_segs:
        all_area = jnp.any(im_seg != jnp.array([0, 0, 0]), axis=-1)
        all_areas.append(all_area)
    return rgbds, all_areas, background_areas


def wp_to_jax(model_wp, state_wp, hyperparams):
    with jax.experimental.enable_x64():
        shape_geo_source = wp.to_jax(model_wp.shape_geo.source).astype(jnp.uint64)
    
    model_jax = b3d.Model(wp.to_jax(model_wp.rigid_contact_count), wp.to_jax(model_wp.rigid_contact_broad_shape0), wp.to_jax(model_wp.rigid_contact_broad_shape1), wp.to_jax(model_wp.shape_contact_pairs), wp.to_jax(model_wp.shape_transform), wp.to_jax(model_wp.shape_body), wp.to_jax(model_wp.body_mass), wp.to_jax(model_wp.shape_geo.type), wp.to_jax(model_wp.shape_geo.scale), shape_geo_source, wp.to_jax(model_wp.shape_geo.thickness), wp.to_jax(model_wp.shape_collision_radius), wp.to_jax(model_wp.rigid_contact_point_id), wp.to_jax(model_wp.shape_ground_contact_pairs), wp.to_jax(model_wp.rigid_contact_tids), wp.to_jax(model_wp.rigid_contact_shape0), wp.to_jax(model_wp.rigid_contact_shape1), wp.to_jax(model_wp.rigid_contact_point0), wp.to_jax(model_wp.rigid_contact_point1), wp.to_jax(model_wp.rigid_contact_offset0), wp.to_jax(model_wp.rigid_contact_offset1), wp.to_jax(model_wp.rigid_contact_normal), wp.to_jax(model_wp.rigid_contact_thickness), wp.to_jax(model_wp.body_com), wp.to_jax(model_wp.body_inertia), wp.to_jax(model_wp.body_inv_mass), wp.to_jax(model_wp.body_inv_inertia), wp.to_jax(model_wp.shape_materials.ke), wp.to_jax(model_wp.shape_materials.kd), wp.to_jax(model_wp.shape_materials.kf), wp.to_jax(model_wp.shape_materials.ka), wp.to_jax(model_wp.shape_materials.mu))
    state_jax = b3d.State(wp.to_jax(state_wp.body_q), wp.to_jax(state_wp.body_qd), wp.to_jax(state_wp.body_f))
    hyperparams["physics_args"]["rigid_contact_max"] = Pytree.const(model_wp.rigid_contact_max)
    hyperparams["physics_args"]["shape_contact_pair_count"] = Pytree.const(model_wp.shape_contact_pair_count)
    hyperparams["physics_args"]["shape_ground_contact_pair_count"] = Pytree.const(model_wp.shape_ground_contact_pair_count)
    hyperparams["physics_args"]["rigid_contact_margin"] = Pytree.const(model_wp.rigid_contact_margin)
    hyperparams["physics_args"]["body_count"] = Pytree.const(len(model_wp.body_mass))
    return model_jax, state_jax, hyperparams


def get_initial_state(
    pred_file, object_ids, object_segmentation_colors, meshes, seg, rgbd, hyperparams
):
    fps = hyperparams["physics_args"]["fps"].unwrap()
    sim_substeps = hyperparams["physics_args"]["sim_substeps"].unwrap()
    frame_dt = 1.0 / fps
    sim_dt = frame_dt/sim_substeps
    hyperparams["physics_args"]["sim_dt"] = Pytree.const(sim_dt)

    builder = wp.sim.ModelBuilder()
    pred = pred_file["scene"][0]["objects"]

    initial_state = {}
    hyperparams["meshes"] = {}
    for o_id, color in zip(object_ids, object_segmentation_colors):
        area = get_mask_area(seg, [color])
        object_colors = rgbd[..., 0:3][area]
        mean_object_colors = jnp.mean(object_colors, axis=0)
        assert not jnp.isnan(mean_object_colors).any()

        initial_state[f"object_pose_{o_id}"] = b3d.Pose(
            jnp.array(pred[str(o_id)]["location"][0]),
            jnp.array(pred[str(o_id)]["rotation"][0]),
        )
        hyperparams["meshes"][int(o_id)] = b3d.Mesh(
                scale_mesh(meshes[pred[str(o_id)]["type"][0]].vertices, jnp.array(pred[str(o_id)]["scale"][0])),
                meshes[pred[str(o_id)]["type"][0]].faces,
                jnp.ones(meshes[pred[str(o_id)]["type"][0]].vertices.shape)
                * mean_object_colors,
            )
        
        if o_id == 1:
            continue
        b = builder.add_body(
                origin=wp.transform(
                    np.array(pred[str(o_id)]["location"][0]), np.array(pred[str(o_id)]["rotation"][0]),
                )
        )
        builder.add_shape_mesh(
            body=b,
            mesh=wp.sim.Mesh(meshes[pred[str(o_id)]["type"][0]].vertices, meshes[pred[str(o_id)]["type"][0]].faces),
            pos=wp.vec3(0.0, 0.0, 0.0),
            scale=np.array(pred[str(o_id)]["scale"][0]),
            restitution=hyperparams["physics_args"]["restitution"].unwrap(),
            mu=hyperparams["physics_args"]["mu"].unwrap(),
            # ke=self.ke,
            # kd=self.kd,
            # kf=self.kf,
            density=1e3,
            has_ground_collision=True,
            has_shape_collision=True
        )

    hyperparams["object_ids"] = Pytree.const([o_id for o_id in object_ids])

    builder.set_ground_plane(mu=hyperparams["physics_args"]["mu"].unwrap())
    model = builder.finalize()
    model.ground = True

    state_0 = model.state()
    wp.sim.eval_fk(model, model.joint_q, model.joint_qd, None, state_0)

    model_jax, state_0_jax, hyperparams = wp_to_jax(model, state_0, hyperparams)
    initial_state["prev_model"] = model_jax
    initial_state["prev_state"] = state_0_jax

    renderer = wp.sim.render.SimRenderer(model, '/home/haw027/code/b3d/test.usd', scaling=0.5)

    return initial_state, hyperparams, renderer, state_0


def calculate_relevant_objects(
    rgbds_t2, rgbds_t1, seg2, seg1, object_ids, object_segmentation_colors
):
    # diff = rgbds_t2[...,3] - rgbds_t1[...,3]
    diff = rgbds_t2 - rgbds_t1
    relevant_objects = []
    for o_id, color in zip(object_ids, object_segmentation_colors):
        mask1 = get_mask_area(seg1, [color])
        mask2 = get_mask_area(seg2, [color])
        mask = np.logical_and(mask1, mask2)
        area = np.abs(diff[mask])
        diff_area = np.sum(area)
        if diff[mask].shape[0] == 0:
            relevant_objects.append(o_id)
        elif diff_area/diff[mask].shape[0] > 0.01:
            relevant_objects.append(o_id)
    return relevant_objects
