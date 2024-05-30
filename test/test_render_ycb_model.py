import os
import jax.numpy as jnp
import trimesh
import b3d
import rerun as rr

PORT = 8812
rr.init("real")
rr.connect(addr=f"127.0.0.1:{PORT}")

def test_renderer_full(renderer):
    mesh_path = os.path.join(
        b3d.get_root_path(),
        "assets/shared_data_bucket/ycb_video_models/models/003_cracker_box/textured_simple.obj",
    )
    mesh = trimesh.load(mesh_path)

    object_library = b3d.MeshLibrary.make_empty_library()
    object_library.add_trimesh(mesh)

    pose = b3d.Pose.from_position_and_target(
        jnp.array([0.2, 0.2, 0.2]), jnp.array([0.0, 0.0, 0.0])
    ).inv()

    rgb, depth = renderer.render_attribute(
        pose[None, ...],
        object_library.vertices,
        object_library.faces,
        jnp.array([[0, len(object_library.faces)]]),
        object_library.attributes,
    )
    b3d.get_rgb_pil_image(rgb).save(b3d.get_root_path() / "assets/test_results/test_ycb.png")
    assert rgb.sum() > 0

def test_renderer_normal_full(renderer):
    mesh_path = os.path.join(
        b3d.get_root_path(),
        "assets/shared_data_bucket/ycb_video_models/models/003_cracker_box/textured_simple.obj",
    )
    mesh = trimesh.load(mesh_path)

    object_library = b3d.MeshLibrary.make_empty_library()
    object_library.add_trimesh(mesh)

    pose = b3d.Pose.from_position_and_target(
        jnp.array([0.2, 0.2, 0.2]), jnp.array([0.0, 0.0, 0.0])
    ).inv()

    rgb, depth, normal = renderer.render_attribute_normal(
        pose[None, ...],
        object_library.vertices,
        object_library.faces,
        jnp.array([[0, len(object_library.faces)]]),
        object_library.attributes,
    )

    b3d.get_rgb_pil_image((normal+1)/2).save(b3d.get_root_path() / "assets/test_results/test_ycb_normal.png")

    point_im = b3d.utils.unproject_depth(depth, renderer)
    rr.log("pc", rr.Points3D(point_im.reshape(-1,3), colors=rgb.reshape(-1,3)))
    rr.log("arrows", rr.Arrows3D(origins=point_im[::5,::5,:].reshape(-1,3), vectors=normal[::5,::5,:].reshape(-1,3)/100))

    assert jnp.abs(normal).sum() > 0
