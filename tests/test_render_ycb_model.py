import os

import b3d
import b3d.bayes3d as bayes3d
import jax.numpy as jnp
import trimesh


def test_renderer_full(renderer):
    mesh_path = os.path.join(
        b3d.get_root_path(),
        "assets/shared_data_bucket/ycb_video_models/models/003_cracker_box/textured_simple.obj",
    )
    mesh = trimesh.load(mesh_path)

    object_library = bayes3d.MeshLibrary.make_empty_library()
    object_library.add_trimesh(mesh)

    pose = b3d.Pose.from_position_and_target(
        jnp.array([0.2, 0.2, 0.0]), jnp.array([0.0, 0.0, 0.0])
    ).inv()

    rgb, _depth = renderer.render_attribute(
        pose[None, ...],
        object_library.vertices,
        object_library.faces,
        jnp.array([[0, len(object_library.faces)]]),
        object_library.attributes,
    )
    b3d.get_rgb_pil_image(rgb).save(
        b3d.get_root_path() / "assets/test_results/test_ycb.png"
    )
    assert rgb.sum() > 0
