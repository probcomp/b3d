import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np
import os
import trimesh
import b3d
from jax.scipy.spatial.transform import Rotation as Rot
from b3d import Pose
import rerun as rr
import functools
import genjax
from tqdm import tqdm
import jax
import jax.numpy as jnp
import optax
import b3d.differentiable_renderer as rendering
import demos.differentiable_renderer.utils as utils
from functools import partial

rr.init("gradients")
rr.connect("127.0.0.1:8812")

def map_nested_fn(fn):
  '''Recursively apply `fn` to the key-value pairs of a nested dict.'''
  def map_fn(nested_dict):
    return {k: (map_fn(v) if isinstance(v, dict) else fn(k, v))
            for k, v in nested_dict.items()}
  return map_fn

# Set up OpenGL renderer
image_width = 200
image_height = 200
fx = 150.0
fy = 150.0
cx = 100.0
cy = 100.0
near = 0.001
far = 16.0

WINDOW = 5

video_input = b3d.VideoInput.load(
    os.path.join(
        b3d.get_root_path(),
        "assets/shared_data_bucket/input_data/mug_handle_visible.video_input.npz",
    )
)
scaling_factor = 5
T = 0
image_width, image_height, fx, fy, cx, cy, near, far = (
    jnp.array(video_input.camera_intrinsics_depth) / scaling_factor
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
renderer = b3d.Renderer(image_width, image_height, fx, fy, cx, cy, near, far)


_rgb = video_input.rgb[T].astype(jnp.float32) / 255.0
_depth = video_input.xyz[T].astype(jnp.float32)[..., 2]
rgb = jnp.clip(
    jax.image.resize(_rgb, (image_height, image_width, 3), "nearest"), 0.0, 1.0
)
depth = jax.image.resize(_depth, (image_height, image_width), "nearest")


mesh_path = os.path.join(
    b3d.get_root_path(), "assets/shared_data_bucket/ycb_video_models/models/025_mug/textured_simple.obj"
)
mesh = trimesh.load(mesh_path)
object_library = b3d.MeshLibrary.make_empty_library()
object_library.add_trimesh(mesh)


point_cloud = b3d.xyz_from_depth(depth, fx, fy, cx, cy).reshape(-1, 3)

vertex_colors = object_library.attributes
rgb_object_samples = vertex_colors[
    jax.random.choice(jax.random.PRNGKey(0), jnp.arange(len(vertex_colors)), (10,))
]
distances = jnp.abs(rgb[..., None] - rgb_object_samples.T).sum([-1, -2])
# rr.log("image/distances", rr.DepthImage(distances))
# rr.log("img", rr.Image(rgb))

object_center_hypothesis = point_cloud[distances.argmin()]
key = jax.random.PRNGKey(10)
pose = Pose.sample_gaussian_vmf_pose(
    key, Pose.from_translation(object_center_hypothesis), 0.001, 0.01
)

hyperparams = rendering.DEFAULT_HYPERPARAMS
def render(params):
    image = rendering.render_to_average_rgbd(
        renderer,
        b3d.Pose(params["position"], params["quaternion"]).apply(object_library.vertices),
        object_library.faces,
        object_library.attributes,
        background_attribute=jnp.array([0.0, 0.0, 0.0, 0])
    )
    return image

render_jit = jax.jit(render)

gt_image = rgb

def loss_func_rgbd(params, gt):
    image = render(params)
    return jnp.mean(jnp.abs(image[...,:3] - gt[...,:3]))
    #  + jnp.mean(jnp.abs(image[...,3] - gt[...,3]))
loss_func_rgbd_grad = jax.jit(jax.value_and_grad(loss_func_rgbd, argnums=(0,)))

@partial(jax.jit, static_argnums=(0,))
def update_params(tx, params, gt_image, state):
    loss, (gradients,) = loss_func_rgbd_grad(params, gt_image)
    updates, state = tx.update(gradients, state, params)
    params = optax.apply_updates(params, updates)
    return params, state, loss

label_fn = map_nested_fn(lambda k, _: k)

tx = optax.multi_transform(
    {
    'position': optax.adam(1e-2),
    'quaternion': optax.adam(1e-2),
    },
    label_fn
)


params = {
    "position": pose.position,
    "quaternion": pose.quaternion,
}

rr.log("image", rr.Image(gt_image[...,:3]), timeless=True)
rr.log("cloud", rr.Points3D(gt_pose.apply(object_library.vertices)), timeless=True)
rr.log("loss2", rr.SeriesLine(name="loss2"), timeless=True)


pbar = tqdm(range(200))
state = tx.init(params)
images = [render_jit(params)]
for t in pbar:
    params, state, loss = update_params(tx, params, gt_image, state)
    pbar.set_description(f"Loss: {loss}")
    rr.set_time_sequence("frame", t)
    image = render_jit(params)
    rr.log("image/reconstruction", rr.Image(image[...,:3]))
    rr.log("cloud/reconstruction", rr.Points3D(b3d.Pose(params["position"], params["quaternion"]).apply(object_library.vertices)))

    rr.log(
        "loss2",
        rr.Scalar(loss),
        rr.SeriesLine(),
    )
