import io
import os
from functools import reduce

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from genjax import Pytree
from PIL import Image

import b3d
from b3d import Mesh


def load_scene(scene_id, FRAME_RATE=50, subdir="test", T0=0):
    num_scenes = b3d.io.data_loader.get_ycbv_num_images(scene_id, subdir=subdir)

    image_ids = range(1, num_scenes + 1, FRAME_RATE)
    all_data = b3d.io.get_ycbv_data(scene_id, image_ids, subdir=subdir)

    ycb_dir = os.path.join(b3d.utils.get_assets_path(), "bop/ycbv")
    meshes = [
        Mesh.from_obj_file(
            os.path.join(ycb_dir, f'models/obj_{f"{id + 1}".rjust(6, "0")}.ply')
        ).scale(0.001)
        for id in all_data[T0]["object_types"]
    ]

    image_height, image_width = all_data[T0]["rgbd"].shape[:2]
    fx, fy, cx, cy = all_data[T0]["camera_intrinsics"]
    scaling_factor = 1.0
    renderer = b3d.renderer.renderer_original.RendererOriginal(
        image_width * scaling_factor,
        image_height * scaling_factor,
        fx * scaling_factor,
        fy * scaling_factor,
        cx * scaling_factor,
        cy * scaling_factor,
        0.01,
        4.0,
    )

    intrinsics = {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "image_height": Pytree.const(image_height),
        "image_width": Pytree.const(image_width),
        "near": 0.01,
        "far": 3.0,
    }

    initial_object_poses = all_data[T0]["object_poses"]

    return all_data, meshes, renderer, intrinsics, initial_object_poses


def load_object_given_scene(all_data, meshes, renderer, OBJECT_INDEX, T0=0):
    T = T0
    fx, fy, cx, cy = all_data[T]["camera_intrinsics"]

    template_pose = (
        all_data[T]["camera_pose"].inv() @ all_data[T]["object_poses"][OBJECT_INDEX]
    )
    rendered_rgbd = renderer.render_rgbd_from_mesh(
        meshes[OBJECT_INDEX].transform(template_pose)
    )
    xyz_rendered = b3d.xyz_from_depth(rendered_rgbd[..., 3], fx, fy, cx, cy)

    xyz_observed = b3d.xyz_from_depth(all_data[T]["rgbd"][..., 3], fx, fy, cx, cy)
    mask = (
        all_data[T]["masks"][OBJECT_INDEX]
        * (xyz_observed[..., 2] > 0)
        * (jnp.linalg.norm(xyz_rendered - xyz_observed, axis=-1) < 0.01)
    )
    model_vertices = template_pose.inv().apply(xyz_rendered[mask])
    model_colors = all_data[T]["rgbd"][..., :3][mask]

    subset = jax.random.permutation(jax.random.PRNGKey(0), len(model_vertices))[
        : min(10000, len(model_vertices))
    ]
    model_vertices = model_vertices[subset]
    model_colors = model_colors[subset]

    return (template_pose, model_vertices, model_colors)


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


def resize_rgbds_and_get_masks(rgbds, seg_arr, im_height, im_width):
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
        all_area = jnp.any(im_seg != jnp.array([0, 0, 0]), axis=-1)
        all_areas.append(all_area)
    return rgbds, all_areas


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
        initial_state[f"object_vel_{o_id}"] = jnp.zeros(3)
        initial_state[f"object_ang_vel_{o_id}"] = jnp.zeros(3)
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
    return initial_state, hyperparams
