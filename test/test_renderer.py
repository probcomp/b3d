import b3d
import jax.numpy as jnp


def test_renderer_full(renderer):
    vertices = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    faces = jnp.array(
        [
            [0, 1, 2],
        ]
    )

    vertex_colors = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    poses = b3d.Pose.from_translation(jnp.array([0.0, 0.0, 5.1]))[None, ...]

    uvs, object_ids, triangle_ids, zs = renderer.rasterize(
        poses, vertices, faces, jnp.array([[0, len(faces)]])
    )
    rgb, depth = renderer.render_attribute(
        poses, vertices, faces, jnp.array([[0, len(faces)]]), vertex_colors
    )
    b3d.get_rgb_pil_image(rgb).save(b3d.get_root_path() / "assets/test_renderer.png")
