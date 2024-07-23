import glob
import json
import os
import subprocess
from pathlib import Path

import cv2
import imageio
import jax
import jax.numpy as jnp
import liblzfse  # https://pypi.org/project/pyliblzfse/
import numpy as np
from natsort import natsorted
from PIL import Image
from tqdm import tqdm

from b3d.pose import Pose

YCB_MODEL_NAMES = [
    "002_master_chef_can",
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "007_tuna_fish_can",
    "008_pudding_box",
    "009_gelatin_box",
    "010_potted_meat_can",
    "011_banana",
    "019_pitcher_base",
    "021_bleach_cleanser",
    "024_bowl",
    "025_mug",
    "035_power_drill",
    "036_wood_block",
    "037_scissors",
    "040_large_marker",
    "051_large_clamp",
    "052_extra_large_clamp",
    "061_foam_brick",
]


def remove_zero_pad(img_id):
    for i, ch in enumerate(img_id):
        if ch != "0":
            return img_id[i:]


def get_ycbv_num_test_images(ycb_dir, scene_id):
    scene_id = str(scene_id).rjust(6, "0")

    data_dir = os.path.join(ycb_dir, "test")
    scene_data_dir = os.path.join(
        data_dir, scene_id
    )  # depth, mask, mask_visib, rgb; scene_camera.json, scene_gt_info.json, scene_gt.json

    scene_rgb_images_dir = os.path.join(scene_data_dir, "rgb")
    sorted(glob.glob(scene_rgb_images_dir + "/*.png"))[-1]
    return int(
        os.path.basename(sorted(glob.glob(scene_rgb_images_dir + "/*.png"))[-1]).split(
            "."
        )[0]
    )


def get_ycbv_test_images(ycb_dir, scene_id, images_indices, fields=[]):
    scene_id = str(scene_id).rjust(6, "0")

    data_dir = os.path.join(ycb_dir, "test")
    scene_data_dir = os.path.join(
        data_dir, scene_id
    )  # depth, mask, mask_visib, rgb; scene_camera.json, scene_gt_info.json, scene_gt.json

    scene_rgb_images_dir = os.path.join(scene_data_dir, "rgb")
    scene_depth_images_dir = os.path.join(scene_data_dir, "depth")

    with open(os.path.join(scene_data_dir, "scene_camera.json")) as scene_cam_data_json:
        scene_cam_data = json.load(scene_cam_data_json)

    with open(os.path.join(scene_data_dir, "scene_gt.json")) as scene_imgs_gt_data_json:
        scene_imgs_gt_data = json.load(scene_imgs_gt_data_json)

    all_data = []
    for image_index in tqdm(images_indices):
        img_id = str(image_index).rjust(6, "0")

        data = {
            "img_id": img_id,
        }

        # get camera intrinsics and pose for image
        image_cam_data = scene_cam_data[remove_zero_pad(img_id)]
        cam_K = jnp.array(image_cam_data["cam_K"]).reshape(3, 3)
        cam_R_w2c = jnp.array(image_cam_data["cam_R_w2c"]).reshape(3, 3)
        cam_t_w2c = jnp.array(image_cam_data["cam_t_w2c"]).reshape(3, 1)
        cam_pose_w2c = jnp.vstack(
            [jnp.hstack([cam_R_w2c, cam_t_w2c]), jnp.array([0, 0, 0, 1])]
        )
        cam_pose = jnp.linalg.inv(cam_pose_w2c)
        cam_pose = b3d.Pose.from_matrix(
            cam_pose.at[:3, 3].set(cam_pose[:3, 3] * 1.0 / 1000.0)
        )
        data["camera_pose"] = cam_pose

        cam_K = np.array(cam_K)
        fx, fy, cx, cy = (
            cam_K[0, 0],
            cam_K[1, 1],
            cam_K[0, 2],
            cam_K[1, 2],
        )
        data["camera_intrinsics"] = jnp.array([fx, fy, cx, cy])

        cam_depth_scale = image_cam_data["depth_scale"]

        # get rgb image
        rgb = jnp.array(Image.open(os.path.join(scene_rgb_images_dir, f"{img_id}.png")))

        # get depth image
        depth = jnp.array(
            Image.open(os.path.join(scene_depth_images_dir, f"{img_id}.png"))
        )
        rgbd = jnp.concatenate(
            [rgb / 255.0, (depth * cam_depth_scale / 1000.0)[..., None]], axis=-1
        )
        data["rgbd"] = rgbd

        # get GT object model ID+poses
        objects_gt_data = scene_imgs_gt_data[remove_zero_pad(img_id)]
        gt_poses = []
        gt_ids = []
        for d in objects_gt_data:
            model_R = jnp.array(d["cam_R_m2c"]).reshape(3, 3)
            model_t = jnp.array(d["cam_t_m2c"]).reshape(3, 1)
            model_pose = jnp.vstack(
                [jnp.hstack([model_R, model_t]), jnp.array([0, 0, 0, 1])]
            )
            model_pose = model_pose.at[:3, 3].set(model_pose[:3, 3] * 1.0 / 1000.0)
            gt_poses.append(model_pose)

            obj_id = d["obj_id"] - 1

            gt_ids.append(obj_id)

        data["object_types"] = jnp.array(gt_ids)
        data["object_poses"] = cam_pose @ b3d.Pose.stack_poses(
            [b3d.Pose.from_matrix(p) for p in jnp.array(gt_poses)]
        )

        all_data.append(data)
    return all_data


def get_ycb_mesh(ycb_dir, id):
    return b3d.Mesh.from_obj_file(
        os.path.join(ycb_dir, f'models/obj_{f"{id + 1}".rjust(6, "0")}.ply')
    ).scale(0.001)


def get_ycbineoat_images(ycbineaot_dir, video_name, images_indices):
    video_dir = os.path.join(ycbineaot_dir, f"{video_name}")

    color_files = sorted(glob.glob(f"{video_dir}/rgb/*.png"))
    K = np.loadtxt(f"{video_dir}/cam_K.txt").reshape(3, 3)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    gt_pose_files = sorted(glob.glob(f"{video_dir}/annotated_poses/*"))

    all_data = []
    for image_index in images_indices:
        videoname_to_object = {
            "bleach0": "021_bleach_cleanser",
            "bleach_hard_00_03_chaitanya": "021_bleach_cleanser",
            "cracker_box_reorient": "003_cracker_box",
            "cracker_box_yalehand0": "003_cracker_box",
            "mustard0": "006_mustard_bottle",
            "mustard_easy_00_02": "006_mustard_bottle",
            "sugar_box1": "004_sugar_box",
            "sugar_box_yalehand0": "004_sugar_box",
            "tomato_soup_can_yalehand0": "005_tomato_soup_can",
        }
        object_id = b3d.io.YCB_MODEL_NAMES.index(
            videoname_to_object[os.path.basename(video_dir)]
        )
        rgb = imageio.imread(color_files[image_index])[..., :3]
        depth = cv2.imread(color_files[image_index].replace("rgb", "depth"), -1) / 1e3
        depth[(depth < 0.1)] = 0
        rgbd = jnp.concatenate([rgb / 255.0, depth[..., None]], axis=-1)
        gt_pose = Pose.from_matrix(np.loadtxt(gt_pose_files[image_index]).reshape(4, 4))
        data = {}
        data["rgbd"] = rgbd
        data["camera_intrinsics"] = jnp.array([fx, fy, cx, cy])
        data["object_poses"] = gt_pose[None, ...]
        data["object_types"] = jnp.array([object_id])
        data["camera_pose"] = Pose.identity()

        all_data.append(data)
    return all_data


### R3D


def load_depth(filepath):
    with open(filepath, "rb") as depth_fh:
        raw_bytes = depth_fh.read()
        decompressed_bytes = liblzfse.decompress(raw_bytes)
        depth_img = jnp.frombuffer(decompressed_bytes, dtype=jnp.float32)

    # depth_img = depth_img.reshape((640, 480))  # For a FaceID camera 3D Video
    depth_img = depth_img.reshape((256, 192))  # For a LiDAR 3D Video

    return depth_img


def load_color(filepath):
    img = cv2.imread(filepath)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_intrinsics(metadata: dict):
    """Converts Record3D metadata dict into intrinsic info needed by nerfstudio
    Args:
        metadata: Dict containing Record3D metadata
        downscale_factor: factor to scale RGB image by (usually scale factor is
            set to 7.5 for record3d 1.8 or higher -- this is the factor that downscales
            RGB images to lidar)
    Returns:
        dict with camera intrinsics keys needed by nerfstudio
    """

    # Camera intrinsics
    K = jnp.array(metadata["K"]).reshape((3, 3)).T
    K = K
    fx = K[0, 0]
    fy = K[1, 1]

    H = metadata["h"]
    W = metadata["w"]

    # # TODO(akristoffersen): The metadata dict comes with principle points,
    # # but caused errors in image coord indexing. Should update once that is fixed.
    # cx, cy = W / 2, H / 2
    cx, cy = K[0, 2], K[1, 2]

    scaling_factor = metadata["dw"] / metadata["w"]
    return (
        jnp.array([W, H, fx, fy, cx, cy, 0.01, 100.0]),
        jnp.array(
            [
                W * scaling_factor,
                H * scaling_factor,
                fx * scaling_factor,
                fy * scaling_factor,
                cx * scaling_factor,
                cy * scaling_factor,
                0.01,
                100.0,
            ]
        ),
    )


def load_r3d(r3d_path):
    r3d_path = Path(r3d_path)
    subprocess.run([f"cp {r3d_path} /tmp/{r3d_path.name}.zip"], shell=True)
    subprocess.run(
        [f"unzip -qq -o /tmp/{r3d_path.name}.zip -d /tmp/{r3d_path.name}"], shell=True
    )
    datapath = f"/tmp/{r3d_path.name}"

    f = open(os.path.join(datapath, "metadata"), "r")
    metadata = json.load(f)

    intrinsics_rgb, intrinsics_depth = get_intrinsics(metadata)

    color_paths = natsorted(glob.glob(os.path.join(datapath, "rgbd", "*.jpg")))
    depth_paths = natsorted(glob.glob(os.path.join(datapath, "rgbd", "*.depth")))
    natsorted(glob.glob(os.path.join(datapath, "rgbd", "*.conf")))

    rgb = jnp.array([load_color(color_paths[i]) for i in range(len(color_paths))])
    depths = jnp.array([load_depth(depth_paths[i]) for i in range(len(color_paths))])
    depths = depths.at[jnp.isnan(depths)].set(0.0)
    jax.vmap(jax.image.resize, in_axes=(0, None, None))(
        depths,
        (rgb.shape[1], rgb.shape[2]),
        "linear",
    )

    pose_data = jnp.array(metadata["poses"])  # (N, 7)
    # NB: Record3D / scipy use "scalar-last" format quaternions (x y z w)
    # https://fzheng.me/2017/11/12/quaternion_conventions_en/
    positions = pose_data[..., 4:] * jnp.array([1, -1, -1])  # (N, 3)
    quaternions = pose_data[..., :4] * jnp.array([-1, 1, 1, -1])  # (N, 4)
    _, _, _fx, _fy, _cx, _cy, _, _ = intrinsics_depth

    return {
        "rgb": rgb,
        "depth": depths,
        "camera_pose": Pose(positions, quaternions),
        "camera_intrinsics_rgb": intrinsics_rgb,
        "camera_intrinsics_depth": intrinsics_depth,
    }
