### IMPORTS ###

import os

import b3d.chisight.dynamic_object_model.kfold_image_kernel as kik
import jax
import jax.numpy as jnp

import b3d
from b3d import Mesh

b3d.rr_init("kfold_image_kernel2")


def test_sampling_on_real_data():
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
            os.path.join(ycb_dir, f"models/obj_{f'{id + 1}'.rjust(6, '0')}.ply")
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

    image_kernel = kik.KfoldMixturePointsToImageKernel(5)

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
    assert samples_20.shape == (20, image_height // 4, image_width // 4, 4)

    assert (
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
        ).shape
        == ()
    )
