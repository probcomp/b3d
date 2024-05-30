import jax.numpy as jnp
import jax
import trimesh
import rerun as rr
import b3d
import optax
import os
from functools import partial
from tqdm import tqdm
import numpy as np

def map_nested_fn(fn):
  '''Recursively apply `fn` to the key-value pairs of a nested dict.'''
  def map_fn(nested_dict):
    return {k: (map_fn(v) if isinstance(v, dict) else fn(k, v))
            for k, v in nested_dict.items()}
  return map_fn
label_fn = map_nested_fn(lambda k, _: k)


# Load date
# path = os.path.join(
#     b3d.get_root_path(),
#     "assets/shared_data_bucket/input_data/static_room_pan_around.r3d.video_input.npz",
# )
# video_input = b3d.VideoInput.load(path)
# data = jnp.load(os.path.join(
#     b3d.get_root_path(),
#     "assets/shared_data_bucket/input_data/cotracker_static_outputs.npz"
# ))

# path = os.path.join(
#     b3d.get_root_path(),
#     "assets/shared_data_bucket/input_data/bubly_manipulation.r3d.video_input.npz",
# )
# video_input = b3d.VideoInput.load(path)
# data = jnp.load(os.path.join(
#     b3d.get_root_path(),
#     "assets/shared_data_bucket/input_data/cotracker_bubly_outputs.npz"
# ))

path = os.path.join(
    b3d.get_root_path(),
    "assets/shared_data_bucket/input_data/royce_static_to_dynamic.r3d.video_input.npz",
)
video_input = b3d.VideoInput.load(path)
data = jnp.load(path + "cotracker_output.npz")



first_frame_rgb = (video_input.rgb[0] / 255.0)

# Get intrinsics
image_width, image_height, fx, fy, cx, cy, near, far = np.array(
    video_input.camera_intrinsics_rgb
)
image_width, image_height = int(image_width), int(image_height)
fx, fy, cx, cy, near, far = (
    float(fx),
    float(fy),
    float(cx),
    float(cy),
    float(near),
    float(far),
)


rr.init("demo")
rr.connect("127.0.0.1:8812")


def model(params, fx, fy, cx, cy):
    xyz_in_camera_frame = jax.vmap(lambda i: b3d.Pose(params["position"][i], params["quaternion"][i]).apply(params["xyz"]))(jnp.arange(params["position"].shape[0]))
    pixel_coords = b3d.xyz_to_pixel_coordinates(xyz_in_camera_frame, fx, fy, cx, cy)
    return pixel_coords
model_jit = jax.jit(model)

def _pixel_coordinates_to_image(pixel_coords, image_height, image_width):
    img = jnp.zeros((image_height, image_width))
    img = img.at[jnp.round(pixel_coords[:, 0]).astype(jnp.int32), jnp.round(pixel_coords[:, 1]).astype(jnp.int32)].set(jnp.arange(len(pixel_coords))+1 )
    return img
pixel_coordinates_to_image = jax.vmap(_pixel_coordinates_to_image, in_axes=(0, None, None))
pixel_coordinates_to_image_jit = jax.jit(pixel_coordinates_to_image)

def loss_function(params, gt_info):
    gt_pixel_coordinates, gt_visibility = gt_info
    pixel_coords = model(params, fx,fy, cx,cy)
    return jnp.mean(gt_visibility[...,None] * (pixel_coords - gt_pixel_coordinates)**2)
loss_func_grad = jax.jit(jax.value_and_grad(loss_function, argnums=(0,)))

@partial(jax.jit, static_argnums=(0,))
def update_params(tx, params, gt_image, state):
    loss, (gradients,) = loss_func_grad(params, gt_image)
    updates, state = tx.update(gradients, state, params)
    params = optax.apply_updates(params, updates)
    return params, state, loss

pred_tracks = jnp.array(data["pred_tracks"])[0]
pred_visibility = jnp.array(data["pred_visibility"])[0]

import matplotlib.pyplot as plt
from matplotlib import colormaps

colors = colormaps["rainbow"](jnp.linspace(0, 1, len(pred_tracks[0])))
colors = colormaps["rainbow"](jnp.linspace(0, 1, len(pred_tracks[0])))
for t in range(len(pred_tracks)):
    rr.set_time_sequence("input", t)
    rr.log("rgb", rr.Image(video_input.rgb[::3][t] / 255.0))
    rr.log("rgb/points", rr.Points2D(jnp.transpose(pred_tracks[t,...,jnp.array([0,1])],(1,0)),
                                 colors=colors * pred_visibility[t][:,None] + (1.0 - pred_visibility[t])[:,None]))


T = 5


# pil_images = [
#     b3d.get_rgb_pil_image(rgb / 255.0)
#     for rgb in video_input.rgb[:200][::T]
# ]
# b3d.make_video_from_pil_images(pil_images, "video.mp4")
gt_pixel_coordinates = pred_tracks[:,...,jnp.array([1,0])][::T]
gt_pixel_coordinate_colors = first_frame_rgb[
    jnp.round(gt_pixel_coordinates[0, :, 0]).astype(jnp.int32),
    jnp.round(gt_pixel_coordinates[0, :, 1]).astype(jnp.int32),
]
gt_visibility = pred_visibility[::T]

sample_gaussian_vmf_multiple = jax.jit(jax.vmap(b3d.Pose.sample_gaussian_vmf_pose, in_axes=(0, None, None, None)))
N = len(gt_pixel_coordinates)



poses = sample_gaussian_vmf_multiple(jax.random.split(jax.random.PRNGKey(1111110), N), b3d.Pose.from_translation(jnp.array([0.0, 0.0, 6.0])), 0.01, 0.01)

tx = optax.multi_transform(
    {
        'xyz': optax.adam(2e-2),
        'position': optax.adam(1e-2),
        'quaternion': optax.adam(1e-2),
    },
    label_fn
)
params = {
    "xyz": jax.random.uniform(jax.random.PRNGKey(1000), (len(gt_pixel_coordinates[0]),3))*0.1,
    "position": poses.pos,
    "quaternion": poses.quat,
}
state = tx.init(params)

pixel_coords = model(params, fx,fy, cx,cy)
print(jnp.abs(pixel_coords - gt_pixel_coordinates))

rr.set_time_sequence("frame", 0)
rr.log("gt", rr.DepthImage(pixel_coordinates_to_image(gt_pixel_coordinates, image_height, image_width)[0]), timeless=True)
rr.log("overlay", rr.DepthImage(pixel_coordinates_to_image(gt_pixel_coordinates, image_height, image_width)[0]), timeless=True)



params_over_time = [params]
pbar = tqdm(range(2000))
for t in pbar:
    params, state, loss = update_params(tx, params, (gt_pixel_coordinates, gt_visibility), state)
    # pbar.set_description(f"Loss: {loss}")
    # params_over_time.append(params)


rr.log("/", rr.ViewCoordinates.RIGHT_HAND_X_UP, timeless=True)

STRIDE = 10
for (t, params) in enumerate(params_over_time[::STRIDE]):
    rr.set_time_sequence("frame", t)
    pixel_coords = model_jit(params, fx,fy, cx,cy)
    reconstruction = pixel_coordinates_to_image(pixel_coords, image_height, image_width)[0]
    error = jnp.abs(pixel_coords - gt_pixel_coordinates).sum(-1) * gt_visibility
    average_error = error.sum(0) / gt_visibility.sum(0)
    # rr.log("reconstruction", rr.DepthImage(reconstruction))
    # rr.log("overlay/reconstruction", rr.DepthImage(reconstruction))
    # i = 0
    normalized_error = (average_error - average_error.min()) / (average_error.max() - average_error.min())
    rr.log("xyz/overlay", rr.Points3D(params["xyz"], colors = gt_pixel_coordinate_colors))

    rr.log("xyz/overlay/outliers", rr.Points3D(params["xyz"], colors = colormaps["jet"](normalized_error)))

    for camera_id in range(params["position"].shape[0]):
        camera_pose = b3d.Pose(params["position"][camera_id], params["quaternion"][camera_id]).inv()
        rr.log(
            f"/camera/{camera_id}/",
            rr.Transform3D(translation=camera_pose.position, rotation=rr.Quaternion(xyzw=camera_pose.xyzw)),
        )
        rr.log(
            f"/camera/{camera_id}/",
            rr.Pinhole(
                resolution=[0.1,0.1],
                focal_length=0.1,
            ),
        )

