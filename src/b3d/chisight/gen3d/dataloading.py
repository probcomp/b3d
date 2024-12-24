import io
from functools import reduce

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from genjax import Pytree
from PIL import Image

import b3d


def load_trial(hdf5_file_path):
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
        object_ids = jnp.array(f["static"]["object_ids"])
        object_segmentation_colors = jnp.array(
            f["static"]["object_segmentation_colors"]
        )
        assert len(object_ids) == len(object_segmentation_colors)

        background_areas = np.zeros((image_arr.shape[1], image_arr.shape[2]))

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
            background_areas = get_mask_area(seg_arr[0], object_segmentation_colors[
                -len(distractors_occluders):
            ])
            object_ids = object_ids[: -len(distractors_occluders)]
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

    reversed_composite_mapping = dict(
        [(value, feature) for feature, value in composite_mapping.items()]
    )
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
        composite_mapping,
        reversed_composite_mapping,
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
        (im_height, im_width),
        method="linear",
    ).astype(bool)
    im_segs = jax.image.resize(
        seg_arr,
        (seg_arr.shape[0], im_height, im_width, *seg_arr.shape[3:]),
        method="linear",
    )
    all_valid_areas = []
    for im_seg in im_segs:
        all_valid_area = jnp.any(im_seg != jnp.array([0, 0, 0]), axis=-1)
        all_valid_areas.append(all_valid_area)

    return rgbds, all_valid_areas, background_areas


def get_initial_state(
    pred_file, object_ids, object_segmentation_colors, meshes, seg, rgbd, hyperparams
):
    pred = pred_file["scene"][0]["objects"]

    initial_state = {}
    hyperparams["meshes"] = []
    for i, (o_id, color) in enumerate(zip(object_ids, object_segmentation_colors)):
        area = get_mask_area(seg, [color])
        object_colors = rgbd[..., 0:3][area]
        mean_object_colors = jnp.mean(object_colors, axis=0)
        assert not jnp.isnan(mean_object_colors).any()

        initial_state[f"object_pose_{o_id}"] = b3d.Pose(
            jnp.array(pred[i]["location"][0]),
            jnp.array(pred[i]["rotation"][0]),
        )
        initial_state[f"object_scale_{o_id}_0"] = jnp.array(pred[i]["scale"][0])
        # if o_id == 3:
        #     initial_state[f"object_vel_{o_id}"] = jnp.array([0.063420634, 2.0810217e-06, 3.2106633e-05])
        # else:
        #     initial_state[f"object_vel_{o_id}"] = jnp.zeros(3)
        initial_state[f"object_vel_{o_id}"] = jnp.zeros(3)
        # initial_state[f"object_ang_vel_{o_id}"] = jnp.zeros(3)
        hyperparams["meshes"].append(
            [
                b3d.Mesh(
                    meshes[pred[i]["type"][0]].vertices,
                    meshes[pred[i]["type"][0]].faces,
                    jnp.ones(meshes[pred[i]["type"][0]].vertices.shape)
                    * mean_object_colors,
                )
            ]
        )

    hyperparams["object_ids"] = Pytree.const([o_id for o_id in object_ids])
    # hyperparams["object_ids"] = [o_id for o_id in object_ids]
    return initial_state, hyperparams
