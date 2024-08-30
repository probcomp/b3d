### IMPORTS ###

import importlib
import os

import b3d
import b3d.chisight.dynamic_object_model.kfold_image_kernel as kik
import jax
import jax.numpy as jnp
import rerun as rr
from b3d import Mesh

b3d.rr_init("kfold_image_kernel2")

### Loading data ###

scene_id = 55
FRAME_RATE = 50
ycb_dir = os.path.join(b3d.get_assets_path(), "bop/ycbv")
print(f"Scene {scene_id}")
b3d.reload(b3d.io.data_loader)
num_scenes = b3d.io.data_loader.get_ycbv_num_test_images(ycb_dir, scene_id)

# image_ids = [image] if image is not None else range(1, num_scenes, FRAME_RATE)
image_ids = range(1, num_scenes + 1, FRAME_RATE)
all_data = b3d.io.data_loader.get_ycbv_test_images(ycb_dir, scene_id, image_ids)

meshes = [
    Mesh.from_obj_file(
        os.path.join(ycb_dir, f'models/obj_{f"{id + 1}".rjust(6, "0")}.ply')
    ).scale(0.001)
    for id in all_data[0]["object_types"]
]

image_height, image_width = all_data[0]["rgbd"].shape[:2]
fx, fy, cx, cy = all_data[0]["camera_intrinsics"]
scaling_factor = 1.0
renderer = b3d.renderer.renderer_original.RendererOriginal(
    image_width * scaling_factor,
    image_height * scaling_factor,
    fx * scaling_factor,
    fy * scaling_factor,
    cx * scaling_factor,
    cy * scaling_factor,
    0.01,
    2.0,
)
b3d.viz_rgb(all_data[0]["rgbd"])

######

T = 0
OBJECT_INDEX = 2

template_pose = (
    all_data[T]["camera_pose"].inv() @ all_data[T]["object_poses"][OBJECT_INDEX]
)
rendered_rgbd = renderer.render_rgbd_from_mesh(
    meshes[OBJECT_INDEX].transform(template_pose)
)
xyz_rendered = b3d.xyz_from_depth(rendered_rgbd[..., 3], fx, fy, cx, cy)

fx, fy, cx, cy = all_data[T]["camera_intrinsics"]
xyz_observed = b3d.xyz_from_depth(all_data[T]["rgbd"][..., 3], fx, fy, cx, cy)
mask = (
    all_data[T]["masks"][OBJECT_INDEX]
    * (xyz_observed[..., 2] > 0)
    * (jnp.linalg.norm(xyz_rendered - xyz_observed, axis=-1) < 0.01)
)
model_vertices = template_pose.inv().apply(xyz_rendered[mask])
model_colors = vertex_attributes = all_data[T]["rgbd"][..., :3][mask]

vertices = xyz_rendered[mask]
vertices.shape

model_rgbd = all_data[T]["rgbd"][mask]
model_rgbd.shape

intrinsics = {
    "fx": fx // 4,
    "fy": fy // 4,
    "cx": cx // 4,
    "cy": cy // 4,
    "height": image_height // 4,
    "width": image_width // 4,
    "near": 0.001,
    "far": 100.0,
}

#####

importlib.reload(kik)

# key = jax.random.PRNGKey(0)
# img = kik.raycast_to_image_nondeterministic(
#     key, intrinsics, vertices, 20
# )
# h, w = intrinsics["height"], intrinsics["width"]
# smp = kik.mapped_pixel_distribution.sample(
#     key, h, w, img,
#     model_rgbd,
#     0.03 * jnp.ones(vertices.shape[0]),
#     0.1 * jnp.ones(vertices.shape[0]),
#     0.01, 0.04, intrinsics["near"], intrinsics["far"]
# )

image_kernel = kik.ImageDistribution(5)


def get_sample(key):
    sample, _ = image_kernel.random_weighted(
        key,
        intrinsics,
        vertices,
        model_rgbd,
        # color outlier probs
        0.003 * jnp.ones(vertices.shape[0]),
        # depth outlier probs
        0.001 * jnp.ones(vertices.shape[0]),
        # color scale
        0.01,
        # depth scale
        0.04,
    )
    return sample


samples_20 = jax.vmap(get_sample)(jax.random.split(jax.random.PRNGKey(0), 20))
for t, sample in enumerate(samples_20):
    rr.set_time_sequence("samples", t)
    rr.log("sample/rgb", rr.Image(sample[:, :, :3]))
    rr.log("sample/depth", rr.DepthImage(sample[:, :, 3]))

image_kernel.estimate_logpdf(
    jax.random.PRNGKey(10),
    samples_20[0],
    intrinsics,
    vertices,
    model_rgbd,
    # color outlier probs
    0.003 * jnp.ones(vertices.shape[0]),
    # depth outlier probs
    0.001 * jnp.ones(vertices.shape[0]),
    # color scale
    0.01,
    # depth scale
    0.04,
)
