import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import cv2
import jax
import b3d
import argparse
from natsort import natsorted
import subprocess
import glob
import os
import json
import liblzfse  # https://pypi.org/project/pyliblzfse/
from pathlib import Path

parser = argparse.ArgumentParser("r3d_to_video_input")
parser.add_argument("input", help=".r3d File", type=str)
args = parser.parse_args()

def load_depth(filepath):
    with open(filepath, 'rb') as depth_fh:
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
        jnp.array([
            W * scaling_factor,
            H * scaling_factor,
            fx * scaling_factor,
            fy * scaling_factor,
            cx * scaling_factor,
            cy * scaling_factor,
            0.01, 100.0
        ])
    )

def load_r3d_video_input(r3d_path):
    r3d_path = Path(r3d_path)
    subprocess.run([f"cp {r3d_path} /tmp/{r3d_path.name}.zip"], shell=True)
    subprocess.run([f"unzip -qq -o /tmp/{r3d_path.name}.zip -d /tmp/{r3d_path.name}"], shell=True)
    datapath = f"/tmp/{r3d_path.name}"

    f = open(os.path.join(datapath, "metadata"), "r")
    metadata = json.load(f)

    intrinsics_rgb, intrinsics_depth = get_intrinsics(metadata)

    color_paths = natsorted(glob.glob(os.path.join(datapath, "rgbd", "*.jpg")))
    depth_paths = natsorted(glob.glob(os.path.join(datapath, "rgbd", "*.depth")))
    conf_paths = natsorted(glob.glob(os.path.join(datapath, "rgbd", "*.conf")))

    colors = jnp.array([load_color(color_paths[i]) for i in range(len(color_paths))])
    depths = jnp.array([load_depth(depth_paths[i]) for i in range(len(color_paths))])
    depths = depths.at[jnp.isnan(depths)].set(0.0)

    pose_data = jnp.array(metadata["poses"])  # (N, 7)
    # NB: Record3D / scipy use "scalar-last" format quaternions (x y z w)
    # https://fzheng.me/2017/11/12/quaternion_conventions_en/
    positions = pose_data[..., 4:] * jnp.array([1, -1, -1])  # (N, 3)
    quaterions = pose_data[..., :4] * jnp.array([-1, 1, 1, -1])  # (N, 4)
    _, _, fx, fy, cx, cy, _, _ = intrinsics_depth

    xyz = jax.vmap(
        lambda p: b3d.xyz_from_depth(p, fx, fy, cx, cy), in_axes=(0,)
    )(depths)
    return b3d.VideoInput(
        rgb=colors,
        xyz=xyz,
        camera_positions=positions,
        camera_quaternions=quaterions,
        camera_intrinsics_rgb=intrinsics_rgb,
        camera_intrinsics_depth=intrinsics_depth,
    )

filename = args.input
video_input = load_r3d_video_input(filename)
result_filename = filename + ".video_input.npz"
print("Writing to ", result_filename)
video_input.save(result_filename)
