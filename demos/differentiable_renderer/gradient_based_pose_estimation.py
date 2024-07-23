import os
from functools import partial

import b3d
from jax.scipy.spatial.transform import Rotation as Rot
from b3d import Pose, Mesh
import rerun as rr
import functools
import genjax
from tqdm import tqdm
import jax
import jax.numpy as jnp
import optax
import b3d.chisight.dense.differentiable_renderer as rendering
import demos.differentiable_renderer.utils as utils
from functools import partial

rr.init("gradients")
rr.connect("127.0.0.1:8812")


def map_nested_fn(fn):
    """Recursively apply `fn` to the key-value pairs of a nested dict."""

    def map_fn(nested_dict):
        return {
            k: (map_fn(v) if isinstance(v, dict) else fn(k, v))
            for k, v in nested_dict.items()
        }

    return map_fn


def render_to_dist_params(
    renderer,
    vertices,
    faces,
    vertex_attributes,
    hyperparams=rendering.DEFAULT_HYPERPARAMS,
):
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
    image = renderer.rasterize_many(vertices[None, ...], faces)
    triangle_id_image = image[0, ..., -1].astype(jnp.int32)

    triangle_intersected_padded = jnp.pad(
        triangle_id_image,
        pad_width=[(hyperparams.WINDOW, hyperparams.WINDOW)],
        constant_values=-1,
    )

    h = rendering.HyperparamsAndIntrinsics(
        hyperparams, renderer.fx, renderer.fy, renderer.cx, renderer.cy
    )
    (weights, attributes) = jax.vmap(
        rendering._get_pixel_attribute_dist_parameters, in_axes=(0, None)
    )(
        b3d.all_pairs(renderer.height, renderer.width),
        (vertices, faces, vertex_attributes, triangle_intersected_padded, h),
    )
    weights = weights.reshape(renderer.height, renderer.width, -1)
    attributes = attributes.reshape(
        renderer.height, renderer.width, -1, vertex_attributes.shape[1]
    )

    return (weights, attributes)


def render_to_average(
    renderer,
    vertices,
    faces,
    vertex_attributes,
    background_attribute,
    hyperparams=rendering.DEFAULT_HYPERPARAMS,
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
    weights, attributes = render_to_dist_params(
        renderer, vertices, faces, vertex_attributes, hyperparams=hyperparams
    )
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
    return render_to_average(
        renderer, vertices, faces, vertex_rgbds, background_attribute, hyperparams
    )


hyperparams = rendering.DifferentiableRendererHyperparams(3, 5e-5, 0.25, -1)


def render(params, mesh_params):
    image = render_to_average_rgbd(
        renderer,
        b3d.Pose(params["position"], params["quaternion"]).apply(
            mesh_params["vertices"]
        ),
        mesh_params["faces"],
        mesh_params["vertex_attributes"],
        background_attribute=jnp.array([0.0, 0.0, 0.0, 0]),
        hyperparams=hyperparams,
    )
    return image


WINDOW = 5


ycb_dir = os.path.join(b3d.get_assets_path(), "bop/ycbv")

# image_ids = [image] if image is not None else range(1, num_scenes, FRAME_RATE)
scene_id = 48
print(f"Scene {scene_id}")
num_scenes = b3d.io.data_loader.get_ycbv_num_test_images(ycb_dir, scene_id)
image_ids = range(1, num_scenes + 1, 50)
all_data = b3d.io.get_ycbv_test_images(ycb_dir, scene_id, image_ids)

meshes = [
    Mesh.from_obj_file(
        os.path.join(ycb_dir, f'models/obj_{f"{id + 1}".rjust(6, "0")}.ply')
    ).scale(0.001)
    for id in all_data[0]["object_types"]
]

height, width = all_data[0]["rgbd"].shape[:2]
fx, fy, cx, cy = all_data[0]["camera_intrinsics"]
scaling_factor = 0.3
renderer = b3d.renderer.renderer_original.RendererOriginal(
    width * scaling_factor,
    height * scaling_factor,
    fx * scaling_factor,
    fy * scaling_factor,
    cx * scaling_factor,
    cy * scaling_factor,
    0.01,
    2.0,
)

IDX = 1
mesh = meshes[IDX]

render_jit = jax.jit(render)

mesh_params = {
    "vertices": mesh.vertices,
    "faces": mesh.faces,
    "vertex_attributes": mesh.vertex_attributes,
}
gt_pose = Pose.from_position_and_target(
    jnp.array([0.3, 0.3, 0.0]),
    jnp.array([0.0, 0.0, 0.0]),
).inv()
gt_image = b3d.resize_image(all_data[0]["rgbd"], renderer.height, renderer.width)


def loss_func_rgbd(params, mesh_params, gt):
    image = render(params, mesh_params)
    rendered_depth = image[..., 3]
    rendered_areas = (rendered_depth / fx) * (rendered_depth / fy)
    return jnp.mean(jnp.abs(image[..., :3] - gt[..., :3]) * rendered_areas[..., None])
    #  + jnp.mean(jnp.abs(image[...,3] - gt[...,3]))


loss_func_rgbd_grad = jax.value_and_grad(loss_func_rgbd, argnums=(0,))


@partial(jax.jit, static_argnums=(1,))
def step(carry, tx):
    (params, gt_image, state) = carry
    loss, (gradients,) = loss_func_rgbd_grad(params, mesh_params, gt_image)
    updates, state = tx.update(gradients, state, params)
    params = optax.apply_updates(params, updates)
    return ((params, gt_image, state), None)


label_fn = map_nested_fn(lambda k, _: k)

tx = optax.multi_transform(
    {
        "position": optax.adam(5e-3),
        "quaternion": optax.adam(5e-3),
    },
    label_fn,
)

pose = all_data[0]["camera_pose"].inv() @ all_data[0]["object_poses"][IDX]

params = {
    "position": pose.position,
    "quaternion": pose.quaternion,
}

rr.log("image", rr.Image(gt_image[..., :3]), timeless=True)
rr.log("cloud", rr.Points3D(gt_pose.apply(mesh.vertices)), timeless=True)

pbar = tqdm(range(200))
state = tx.init(params)
images = [render_jit(params, mesh_params)]
for t in pbar:
    (params, gt_image, state), _ = step((params, gt_image, state), tx)
    rr.set_time_sequence("frame", t)
    image = render_jit(params, mesh_params)
    pbar.set_description(f"Loss: {loss_func_rgbd(params, mesh_params, gt_image)}")
    rr.log("image/reconstruction", rr.Image(image[..., :3]))
    rr.log(
        "cloud/reconstruction",
        rr.Points3D(
            b3d.Pose(params["position"], params["quaternion"]).apply(mesh.vertices)
        ),
    )
