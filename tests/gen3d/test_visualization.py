import os

import b3d
import b3d.chisight.gen3d.model
import b3d.chisight.gen3d.settings
import b3d.io.data_loader
import genjax
import jax
import jax.numpy as jnp
from b3d import Mesh
from b3d.chisight.gen3d.model import (
    make_colors_choicemap,
    make_depth_nonreturn_prob_choicemap,
    make_visibility_prob_choicemap,
)
from genjax import Pytree


def test_visualization():
    b3d.rr_init("test_visualization")
    scene_id = 49
    ycb_dir = os.path.join(b3d.get_assets_path(), "bop/ycbv")
    all_data = b3d.io.data_loader.get_ycbv_test_images(ycb_dir, scene_id, [1])
    OBJECT_INDEX = 0
    id = all_data[0]["object_types"][OBJECT_INDEX]

    mesh = Mesh.from_obj_file(
        os.path.join(ycb_dir, f'models/obj_{f"{id + 1}".rjust(6, "0")}.ply')
    ).scale(0.001)

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

    T = 0
    b3d.rr_set_time(T)

    template_pose = (
        all_data[T]["camera_pose"].inv() @ all_data[T]["object_poses"][OBJECT_INDEX]
    )
    rendered_rgbd = renderer.render_rgbd_from_mesh(mesh.transform(template_pose))
    xyz_rendered = b3d.xyz_from_depth(rendered_rgbd[..., 3], fx, fy, cx, cy)

    fx, fy, cx, cy = all_data[T]["camera_intrinsics"]
    xyz_observed = b3d.xyz_from_depth(all_data[T]["rgbd"][..., 3], fx, fy, cx, cy)
    mask = (
        all_data[T]["masks"][OBJECT_INDEX]
        * (xyz_observed[..., 2] > 0)
        * (jnp.linalg.norm(xyz_rendered - xyz_observed, axis=-1) < 0.01)
    )
    model_vertices = template_pose.inv().apply(xyz_rendered[mask])
    model_colors = all_data[T]["rgbd"][..., :3][mask]

    ### Set up inference hyperparams ###

    hyperparams = b3d.chisight.gen3d.settings.hyperparams

    hyperparams["intrinsics"] = {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "image_height": Pytree.const(image_height),
        "image_width": Pytree.const(image_width),
        "near": 0.01,
        "far": 10.0,
    }
    hyperparams["vertices"] = model_vertices

    num_vertices = model_vertices.shape[0]
    previous_state = {
        "pose": template_pose,
        "colors": model_colors,
        "visibility_prob": jnp.ones(num_vertices)
        * hyperparams["visibility_prob_kernel"].support[-1],
        "depth_nonreturn_prob": jnp.ones(num_vertices)
        * hyperparams["depth_nonreturn_prob_kernel"].support[0],
        "depth_scale": hyperparams["depth_scale_kernel"].support[0],
        "color_scale": hyperparams["color_scale_kernel"].support[0],
    }

    choicemap = (
        genjax.ChoiceMap.d(
            {
                "pose": previous_state["pose"],
                "color_scale": previous_state["color_scale"],
                "depth_scale": previous_state["depth_scale"],
                "rgbd": all_data[T]["rgbd"],
            }
        )
        ^ make_visibility_prob_choicemap(previous_state["visibility_prob"])
        ^ make_colors_choicemap(previous_state["colors"])
        ^ make_depth_nonreturn_prob_choicemap(previous_state["depth_nonreturn_prob"])
    )

    key = jax.random.PRNGKey(0)
    trace, _ = b3d.chisight.gen3d.model.dynamic_object_generative_model.importance(
        key, choicemap, (hyperparams, previous_state)
    )
    b3d.chisight.gen3d.model.viz_trace(
        trace,
        T,
        ground_truth_vertices=mesh.vertices,
        ground_truth_pose=all_data[T]["camera_pose"].inv()
        @ all_data[T]["object_poses"][OBJECT_INDEX],
    )
