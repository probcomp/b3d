import b3d
import jax
import jax.numpy as jnp

import demos.mesh_fitting.tessellation as t



def initialize_mesh_using_depth(video_input):
    width, height, fx, fy, cx, cy, near, far = jnp.array(
        video_input.camera_intrinsics_depth
    )
    width, height = int(width), int(height)
    fx, fy, cx, cy, near, far = (
        float(fx),
        float(fy),
        float(cx),
        float(cy),
        float(near),
        float(far),
    )
    renderer = b3d.Renderer(width, height, fx, fy, cx, cy, near, far)
    rgbs_full_resolution = video_input.rgb[::4] / 255.0
    rgbs = jnp.clip(
        jax.vmap(jax.image.resize, in_axes=(0, None, None))(
            rgbs_full_resolution,
            (video_input.xyz.shape[1], video_input.xyz.shape[2], 3),
            "linear",
        ),
        0.0,
        1.0,
    )
    depths = video_input.xyz[::4][:, :, :, 3]
    rgbds = jnp.concatenate([rgbs, depths[..., None]], axis=-1)

    vertices_2D, faces, triangle_rgbds = t.generate_tessellated_2D_mesh_from_rgb_image(
        rgbds[0], scaledown=2
    )

    MAX_N_FACES = 7

    def get_faces_for_vertex(i):
        return jnp.where(faces == i, size=MAX_N_FACES, fill_value=-1)[0]

    vertex_to_faces = jax.vmap(get_faces_for_vertex)(jnp.arange(vertices_2D.shape[0]))
    # Check we had 1 more padding than we needed, for each vertex
    assert jnp.all(jnp.any(vertex_to_faces == -1, axis=1))

    def get_vertex_depth(v):
        face_indices = vertex_to_faces[v]
        face_indices_safe = jnp.where(face_indices == -1, 0, face_indices)
        depths = jnp.where(
            face_indices != -1, triangle_rgbds[face_indices_safe, 3], 0.0
        )
        n_valid = jnp.sum(depths != 0)
        return jnp.sum(depths) / (n_valid + 1e-5)

    vertex_depths = jax.vmap(get_vertex_depth)(jnp.arange(vertices_2D.shape[0]))

    vertices_3D = jnp.hstack(
        (
            (vertices_2D - jnp.array([cx, cy]))
            * vertex_depths[:, None]
            / jnp.array([fx, fy]),
            vertex_depths[:, None],
        )
    )

    return (vertices_3D, faces, triangle_rgbds, renderer, rgbs)
