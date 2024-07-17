import pykitti
import b3d
import os
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm
import jax
b3d.rr_init("kitti")

basedir = os.path.join(b3d.get_assets_path(), "kitti")
date = '2011_09_26'
drive = '0005'
frames = range(0, 50, 5)
dataset = pykitti.raw(basedir, date, drive, frames=frames)


K = dataset.calib.K_cam0
fx,fy,cx,cy = K[0,0], K[1,1], K[0,2], K[1,2]
rgb = jnp.array(dataset.get_rgb(0)[0])
print(rgb.shape)
height,width = rgb.shape[:2]


rgbs = []
depths = []
for (t,f) in tqdm(enumerate(frames)):
    rgb = jnp.array(dataset.get_rgb(t)[0])
    rgbs.append(rgb)


    velo = dataset.get_velo(t)[:, jnp.array([1, 2, 0])] * jnp.array([-1., -1., 1.])
    b3d.rr.log("velo", b3d.rr.Points3D(velo[...,:3]))

    pixels = b3d.xyz_to_pixel_coordinates(velo[:, :3], fx, fy, cx, cy).astype(jnp.int32)
    depth_image = jnp.zeros((height,width))

    valid_pixels = (
        (0 <= pixels[:, 0])
        * (0 <= pixels[:, 1])
        * (pixels[:, 0] < height)
        * (pixels[:, 1] < width)
        * (velo[:, 2] > 0.0)
    )
    depth_image = depth_image.at[pixels[valid_pixels, 0], pixels[valid_pixels, 1]].set(velo[valid_pixels, 2])
    depths.append(depth_image)
for t in range(len(rgbs)):
    b3d.rr.set_time_sequence("time", t)
    b3d.rr.log("rgb", b3d.rr.Image(rgbs[t]))
    b3d.rr.log("depth", b3d.rr.DepthImage(depths[t]))

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)
import numpy as np

masks = mask_generator.generate(np.array(rgbs[0]))

for t,m in enumerate(masks[:10]):
    b3d.rr.set_time_sequence("time", t)
    b3d.rr.log("depth", b3d.rr.DepthImage(m["segmentation"] * 1.0))

b3d.rr.log("depth/rgb", b3d.rr.Image(rgbs[0]), timeless=True)

CORRECT_MASK_INDEX = 0

mask = jnp.array(masks[0]["segmentation"])

xyzs = b3d.xyz_from_depth(depths[0], fx, fy, cx, cy)[mask]
colors= rgbs[0][mask]
colors = colors[xyzs.sum(-1) > 0]
xyzs = xyzs[xyzs.sum(-1) > 0]

resolution = 0.05
meshes = b3d.mesh.transform_mesh(
    jax.vmap(b3d.mesh.Mesh.cube_mesh)(jnp.ones((xyzs.shape[0],3)) * resolution * 2.0, colors / 255.0),
    b3d.Pose.from_translation(xyzs)[:,None]
)
full_mesh = b3d.mesh.Mesh.squeeze_mesh(meshes)
full_mesh.rr_visualize("mesh")

pose = b3d.Pose.from_translation(full_mesh.vertices.mean(0))
full_mesh.vertices = pose.inv().apply(full_mesh.vertices)
full_mesh.rr_visualize("mesh")

renderer = b3d.RendererOriginal(width, height, fx,fy,cx,cy, 0.01, 100.0)
rgbd = renderer.render_rgbd_from_mesh(full_mesh.transform(pose))
b3d.rr.log("rerender", b3d.rr.Image(rgbd[...,:3]))

b3d.rr.log("rerender/depth", b3d.rr.DepthImage(rgbd[...,3]))

rgbs = jnp.array(rgbs)
depths = jnp.array(depths)
mask.shape


kitti_data_path = b3d.get_assets_path() / "shared_data_bucket/foundation_pose_tracking_datasets/kitti_initial_data.npz"
np.savez(kitti_data_path,
         rgb=np.array(rgbs)/255.0,
         depths=np.array(depths),
            camera_intrinsics=np.array([fx,fy,cx,cy, 0.001, 100.0]),
            object_position=np.array(pose.pos[None,...]),
            object_quaternion=np.array(pose.quat[None,...]),
            mask=np.array(mask[None,...]))

kitti_mesh_path = b3d.get_assets_path() / "shared_data_bucket/foundation_pose_tracking_datasets/kitti_initial_data.obj"
full_mesh.save(kitti_mesh_path)

import numpy as np
import b3d
d = np.load(kitti_data_path)

for (i,v) in d.items():
    print(i, v.shape)