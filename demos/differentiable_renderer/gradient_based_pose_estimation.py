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
import b3d.chisight.dense.differentiable_renderer as rendering
import b3d.likelihoods as likelihoods
from b3d.renderer_original import RendererOriginal
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
renderer = RendererOriginal(image_width, image_height, fx, fy, cx, cy, near, far)

WINDOW = 5

mesh_path = os.path.join(
    b3d.get_root_path(), "assets/shared_data_bucket/ycb_video_models/models/006_mustard_bottle/textured_simple.obj"
)
mesh = trimesh.load(mesh_path)
object_library = b3d.MeshLibrary.make_empty_library()
object_library.add_trimesh(mesh)


def render_to_dist_params(renderer, vertices, faces, vertex_attributes, hyperparams=rendering.DEFAULT_HYPERPARAMS):
    """
    Differentiable rendering to parameters for a per-pixel
    categorical distribution over attributes (e.g. RGB or RGBD).

    Args:
    - renderer: b3d.Renderer
    - vertices: (V, 3)
    - faces: (F, 3)
    - vertex_attributes: (F, A) [A=3 for RGB; A=4 for RGBD]
    - hyperparams: DifferentiableRendererHyperparams
    Returns:
    - weights (H, W, U)
    - attributes (H, W, U-1, A)
    For each pixel, the first weight is the weight assigned to the background
    (ie. assigned to not hitting any object).
    The remaining weights are those assigned to some triangles in the scene.
    The attributes measured on those triangles are contained in `attributes`.
    """
    image = renderer.rasterize(
        vertices[None,...], faces
    )
    triangle_id_image = image[0,...,-1].astype(jnp.int32)


    triangle_intersected_padded = jnp.pad(
        triangle_id_image, pad_width=[(hyperparams.WINDOW, hyperparams.WINDOW)], constant_values=-1
    )

    h = rendering.HyperparamsAndIntrinsics(hyperparams, renderer.fx, renderer.fy, renderer.cx, renderer.cy)
    (weights, attributes) = jax.vmap(rendering._get_pixel_attribute_dist_parameters, in_axes=(0, None))(
        b3d.utils.all_pairs(renderer.height, renderer.width),
        (vertices, faces, vertex_attributes, triangle_intersected_padded, h)
    )
    weights = weights.reshape(renderer.height, renderer.width, -1)
    attributes = attributes.reshape(renderer.height, renderer.width, -1, vertex_attributes.shape[1])

    return (weights, attributes)


def render_to_average(
        renderer,
        vertices,
        faces,
        vertex_attributes,
        background_attribute,
        hyperparams=rendering.DEFAULT_HYPERPARAMS
):
    """
    Differentiable rendering to produce an image by averaging
    the categorical distribution over attributes (e.g. RGB or RGBD)
    returned by `render_to_dist_params`.

    Args:
    - renderer: b3d.Renderer
    - vertices: (V, 3)
    - faces: (F, 3)
    - vertex_attributes: (F, A) [A=3 for RGB; A=4 for RGBD]
    - background_attribute: (A,) attribute to assign to pixels not hitting any object
    - hyperparams: DifferentiableRendererHyperparams
    Returns:
    - image (H, W, A)
    """
    weights, attributes = render_to_dist_params(renderer, vertices, faces, vertex_attributes, hyperparams=hyperparams)
    return rendering.dist_params_to_average(weights, attributes, background_attribute)    


def render_to_average_rgbd(
    renderer,
    vertices,
    faces,
    vertex_rgbs,
    background_attribute=jnp.array([0.1, 0.1, 0.1, 0]),
    hyperparams=rendering.DEFAULT_HYPERPARAMS,
):
    """
    Variant of `render_to_average` for rendering RGBD.
    """
    vertex_depths = vertices[:, 2]
    vertex_rgbds = jnp.concatenate([vertex_rgbs, vertex_depths[:, None]], axis=1)
    return render_to_average(renderer, vertices, faces, vertex_rgbds, background_attribute, hyperparams)



hyperparams = rendering.DifferentiableRendererHyperparams(3, 5e-5, 0.25, -1)

def render(params):
    image = render_to_average_rgbd(
        renderer,
        b3d.Pose(params["position"], params["quaternion"]).apply(object_library.vertices),
        object_library.faces,
        object_library.attributes,
        background_attribute=jnp.array([0.0, 0.0, 0.0, 0]),
        hyperparams=hyperparams
    )
    return image

render_jit = jax.jit(render)


vertices, faces = object_library.vertices, object_library.faces
image = renderer.rasterize(
    vertices[None,...], faces
)

gt_pose = Pose.from_position_and_target(
    jnp.array([0.3, 0.3, 0.0]), jnp.array([0.0, 0.0, 0.0]),
).inv()
gt_image = render_jit({"position": gt_pose.position, "quaternion": gt_pose.quaternion})

def loss_func_rgbd(params, gt):
    image = render(params)
    return jnp.mean(jnp.abs(image[...,:3] - gt[...,:3]))
    #  + jnp.mean(jnp.abs(image[...,3] - gt[...,3]))
loss_func_rgbd_grad = jax.value_and_grad(loss_func_rgbd, argnums=(0,))


@partial(jax.jit, static_argnums=(1,))
def step(carry, tx):
    (params, gt_image, state) = carry
    loss, (gradients,) = loss_func_rgbd_grad(params, gt_image)
    updates, state = tx.update(gradients, state, params)
    params = optax.apply_updates(params, updates)
    return ((params, gt_image, state), None)

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
    (params,  gt_image, state),_ = step((params, gt_image, state), tx)
    rr.set_time_sequence("frame", t)
    image = render_jit(params)
    rr.log("image/reconstruction", rr.Image(image[...,:3]))
    rr.log("cloud/reconstruction", rr.Points3D(b3d.Pose(params["position"], params["quaternion"]).apply(object_library.vertices)))

