import os

import jax
import jax.numpy as jnp
from genjax import Pytree

import b3d
from b3d import Mesh


def load_scene(scene_id, FRAME_RATE=50, subdir="train_real"):
    num_scenes = b3d.io.data_loader.get_ycbv_num_images(scene_id, subdir=subdir)

    image_ids = range(1, num_scenes + 1, FRAME_RATE)
    all_data = b3d.io.get_ycbv_data(scene_id, image_ids, subdir=subdir)

    ycb_dir = os.path.join(b3d.utils.get_assets_path(), "bop/ycbv")
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
        4.0,
    )

    intrinsics = {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "image_height": Pytree.const(image_height),
        "image_width": Pytree.const(image_width),
        "near": 0.01,
        "far": 3.0,
    }

    initial_object_poses = all_data[0]["object_poses"]

    return all_data, meshes, renderer, intrinsics, initial_object_poses


def load_object_given_scene(all_data, meshes, renderer, OBJECT_INDEX):
    T = 0
    fx, fy, cx, cy = all_data[T]["camera_intrinsics"]

    template_pose = (
        all_data[T]["camera_pose"].inv() @ all_data[T]["object_poses"][OBJECT_INDEX]
    )
    rendered_rgbd = renderer.render_rgbd_from_mesh(
        meshes[OBJECT_INDEX].transform(template_pose)
    )
    xyz_rendered = b3d.xyz_from_depth(rendered_rgbd[..., 3], fx, fy, cx, cy)

    xyz_observed = b3d.xyz_from_depth(all_data[T]["rgbd"][..., 3], fx, fy, cx, cy)
    mask = (
        all_data[T]["masks"][OBJECT_INDEX]
        * (xyz_observed[..., 2] > 0)
        * (jnp.linalg.norm(xyz_rendered - xyz_observed, axis=-1) < 0.01)
    )
    model_vertices = template_pose.inv().apply(xyz_rendered[mask])
    model_colors = all_data[T]["rgbd"][..., :3][mask]

    subset = jax.random.permutation(jax.random.PRNGKey(0), len(model_vertices))[
        : min(10000, len(model_vertices))
    ]
    model_vertices = model_vertices[subset]
    model_colors = model_colors[subset]

    return (template_pose, model_vertices, model_colors)


def get_initial_state(template_pose, model_vertices, model_colors, hyperparams):
    num_vertices = model_vertices.shape[0]
    return {
        "pose": template_pose,
        "colors": model_colors,
        "visibility_prob": jnp.ones(num_vertices)
        * hyperparams["visibility_prob_kernel"].support[-1],
        "depth_nonreturn_prob": jnp.ones(num_vertices)
        * hyperparams["depth_nonreturn_prob_kernel"].support[0],
        "depth_scale": hyperparams["depth_scale_kernel"].support[0],
        "color_scale": hyperparams["color_scale_kernel"].support[0],
    }
