import rerun as rr
import genjax
import os
import numpy as np
import jax.numpy as jnp
import jax
from b3d import Pose
import b3d
from tqdm   import tqdm

PORT = 8812
rr.init("asdf233")
rr.connect(addr=f'127.0.0.1:{PORT}')

path = os.path.join(b3d.get_assets_path(),
#  "shared_data_bucket/input_data/orange_mug_pan_around_and_pickup.r3d.video_input.npz")
 "shared_data_bucket/input_data/ramen_ramen_cream.r3d.video_input.npz")
video_input = b3d.VideoInput.load(path)

image_width, image_height, fx,fy, cx,cy,near,far = np.array(video_input.camera_intrinsics_depth)
image_width, image_height = int(image_width), int(image_height)
fx,fy, cx,cy,near,far = float(fx),float(fy), float(cx),float(cy),float(near),float(far)

rgbs = video_input.rgb / 255.0
# Resize rgbs to be same size as depth.
rgbs_resized = jnp.clip(jax.vmap(jax.image.resize, in_axes=(0, None, None))(
    rgbs[::3], (video_input.xyz.shape[1], video_input.xyz.shape[2], 3), "linear"
), 0.0, 1.0)



import torch
from carvekit.api.high import HiInterface

# Check doc strings for more information
interface = HiInterface(object_type="hairs-like",  # Can be "object" or "hairs-like".
                        batch_size_seg=5,
                        batch_size_matting=1,
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                        seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
                        matting_mask_size=2048,
                        trimap_prob_threshold=231,
                        trimap_dilation=30,
                        trimap_erosion_iters=5,
                        fp16=False)

T = 0
output_images = interface([b3d.get_rgb_pil_image(rgbs_resized[T])])
mask  = jnp.array([jnp.array(output_image)[..., -1] > 0.5 for output_image in output_images])[0]

rr.log("/img", rr.Image(rgbs_resized[T]))
rr.log("/img/mask", rr.Image(jnp.tile((mask * 1.0)[...,None],(1,1,3))))

for t in range(len(rgbs_resized)):
    rr.set_time_sequence("frame", t)
    rr.log(f"/img", rr.Image(rgbs_resized[t]))


