import jax.numpy as jnp
import jax
import trimesh
import rerun as rr
import b3d
import optax
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

box_mesh = trimesh.creation.box(jnp.ones(3))
object_vertices = jnp.array(box_mesh.vertices)


box_mesh = trimesh.creation.box(jnp.ones(3))
object_vertices = jnp.array(box_mesh.vertices)

import os
mesh_path = os.path.join(
    b3d.get_root_path(), "assets/shared_data_bucket/025_mug/textured.obj"
)
mesh = trimesh.load(mesh_path)
vertices = jnp.array(mesh.vertices) * 10.0
object_vertices = vertices - jnp.mean(vertices, axis=0)
object_vertices = object_vertices[jax.random.choice(jax.random.PRNGKey(10), jnp.arange(object_vertices.shape[0]), (500,), replace=False)]


# Load date
path = os.path.join(
    b3d.get_root_path(),
    "assets/shared_data_bucket/input_data/static_room_pan_around.r3d.video_input.npz",
)
video_input = b3d.VideoInput.load(path)


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

def _pixel_coordinates_to_image(pixel_coords, image_height, image_width):
    img = jnp.zeros((image_height, image_width))
    img = img.at[jnp.round(pixel_coords[:, 0]).astype(jnp.int32), jnp.round(pixel_coords[:, 1]).astype(jnp.int32)].set(jnp.arange(len(pixel_coords))+1 )
    return img
pixel_coordinates_to_image = jax.vmap(_pixel_coordinates_to_image, in_axes=(0, None, None))

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

data = jnp.load("cotracker_output.npz")
pred_tracks = jnp.array(data["pred_tracks"])[0]
pred_visibility = jnp.array(data["pred_visibility"])[0]

for t in range(len(pred_tracks[0])):
    rr.set_time_sequence("t", t)
    rr.log("points", rr.Points2D(jnp.transpose(pred_tracks[t,...,jnp.array([0,1])],(1,0))))


T = 10


pil_images = [
    b3d.get_rgb_pil_image(rgb / 255.0)
    for rgb in video_input.rgb[:200][::T]
]
b3d.make_video_from_pil_images(pil_images, "video.mp4")
gt_pixel_coordinates = pred_tracks[:,...,jnp.array([1,0])][::T]
gt_pixel_coordinate_colors = first_frame_rgb[
    jnp.round(gt_pixel_coordinates[0, :, 0]).astype(jnp.int32),
    jnp.round(gt_pixel_coordinates[0, :, 1]).astype(jnp.int32),
]
gt_visibility = pred_visibility[::T]

sample_gaussian_vmf_multiple = jax.jit(jax.vmap(b3d.Pose.sample_gaussian_vmf_pose, in_axes=(0, None, None, None)))
N = len(gt_pixel_coordinates)
poses = sample_gaussian_vmf_multiple(jax.random.split(jax.random.PRNGKey(11110), N), b3d.Pose.from_translation(jnp.array([0.0, 0.0, 3.0])), 0.001, 1.0)

tx = optax.multi_transform(
    {
        'xyz': optax.adam(5e-2),
        'position': optax.adam(2e-2),
        'quaternion': optax.adam(2e-2),
    },
    label_fn
)
params = {
    "xyz": jax.random.uniform(jax.random.PRNGKey(10), (len(gt_pixel_coordinates[0]),3)) * 0.01,
    "position": poses.pos,
    "quaternion": poses.quat,
}
state = tx.init(params)

pixel_coords = model(params, fx,fy, cx,cy)
print(jnp.abs(pixel_coords - gt_pixel_coordinates))

rr.set_time_sequence("frame", 0)
rr.log("gt", rr.DepthImage(pixel_coordinates_to_image(gt_pixel_coordinates, image_height, image_width)[0]), timeless=True)
rr.log("overlay", rr.DepthImage(pixel_coordinates_to_image(gt_pixel_coordinates, image_height, image_width)[0]), timeless=True)

pbar = tqdm(range(1000))
for t in pbar:
    params, state, loss = update_params(tx, params, (gt_pixel_coordinates, gt_visibility), state)
    pbar.set_description(f"Loss: {loss}")
    rr.set_time_sequence("frame", t)
    pixel_coords = model(params, fx,fy, cx,cy)
    reconstruction = pixel_coordinates_to_image(pixel_coords, image_height, image_width)[0]
    rr.log("reconstruction", rr.DepthImage(reconstruction))
    rr.log("overlay/reconstruction", rr.DepthImage(reconstruction))
    i = 0
    rr.log("xyz/overlay", rr.Points3D(b3d.Pose(params["position"][i], params["quaternion"][i]).apply(params["xyz"]), colors = gt_pixel_coordinate_colors))
    # rr.log("xyz", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN)  # Set an up-axis

print(jnp.abs(pixel_coords - gt_pixel_coordinates))
print(jnp.abs(pixel_coords - gt_pixel_coordinates).max())
print(jnp.abs(pixel_coords - gt_pixel_coordinates).max())
