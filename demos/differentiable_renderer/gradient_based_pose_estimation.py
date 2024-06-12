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
import b3d.likelihoods as likelihoods
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
renderer = b3d.Renderer(image_width, image_height, fx, fy, cx, cy, near, far)

WINDOW = 5

mesh_path = os.path.join(
    b3d.get_root_path(), "assets/shared_data_bucket/ycb_video_models/models/006_mustard_bottle/textured_simple.obj"
)
mesh = trimesh.load(mesh_path)
object_library = b3d.MeshLibrary.make_empty_library()
object_library.add_trimesh(mesh)




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


gt_pose = Pose.from_position_and_target(
    jnp.array([0.3, 0.3, 0.0]), jnp.array([0.0, 0.0, 0.0]),
).inv()
gt_image = render_jit({"position": gt_pose.position, "quaternion": gt_pose.quaternion})

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
    'position': optax.adam(5e-3),
    'quaternion': optax.adam(5e-3),
    },
    label_fn
)

pose = Pose.from_position_and_target(
    jnp.array([0.6, 0.3, 0.6]), jnp.array([0.0, 0.0, 0.0]),
).inv()

params = {
    "position": pose.position,
    "quaternion": pose.quaternion,
}

rr.log("image", rr.Image(gt_image[...,:3]), timeless=True)
rr.log("cloud", rr.Points3D(gt_pose.apply(object_library.vertices)), timeless=True)


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

