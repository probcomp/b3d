import jax.numpy as jnp
from functools import partial
import numpy as np
from collections import namedtuple
import genjax
from PIL import Image
import subprocess
import jax

import inspect
from inspect import signature
import genjax

def xyz_from_depth(
    z: "Depth Image", 
    fx, fy, cx, cy
):
    v, u = jnp.mgrid[: z.shape[0], : z.shape[1]] + 0.5
    x = (u - cx) / fx
    y = (v - cy) / fy
    xyz = jnp.stack([x, y, jnp.ones_like(x)], axis=-1) * z[..., None]
    return xyz


@partial(jnp.vectorize, signature='(k)->(k)')
def rgb_to_lab(rgb):
    # Convert sRGB to linear RGB
    rgb = jnp.clip(rgb, 0, 1)
    mask = rgb > 0.04045
    rgb = jnp.where(mask, jnp.power((rgb + 0.055) / 1.055, 2.4), rgb / 12.92)

    # RGB to XYZ
    # https://en.wikipedia.org/wiki/SRGB#The_forward_transformation_(CIE_XYZ_to_sRGB)
    rgb_to_xyz = jnp.array([[0.4124564, 0.3575761, 0.1804375],
                            [0.2126729, 0.7151522, 0.0721750],
                            [0.0193339, 0.1191920, 0.9503041]])
    xyz = jnp.dot(rgb, rgb_to_xyz.T)

    # XYZ to LAB
    # https://en.wikipedia.org/wiki/CIELAB_color_space#From_CIEXYZ_to_CIELAB
    xyz_ref = jnp.array([0.95047, 1.0, 1.08883])  # D65 white point
    xyz_normalized = xyz / xyz_ref
    mask = xyz_normalized > 0.008856
    xyz_f = jnp.where(mask, jnp.power(xyz_normalized, 1/3), 7.787 * xyz_normalized + 16/116)

    L = 116 * xyz_f[1] - 16
    a = 500 * (xyz_f[0] - xyz_f[1])
    b = 200 * (xyz_f[1] - xyz_f[2])

    lab = jnp.stack([L, a, b], axis=-1)
    return lab



def make_mesh_from_point_cloud_and_resolution(
    grid_centers, grid_colors, resolutions
):
    box_mesh = trimesh.creation.box(jnp.ones(3))
    base_vertices, base_faces = jnp.array(box_mesh.vertices), jnp.array(box_mesh.faces)

    def process_ith_ball(
        i,
        positions,
        colors,
        base_vertices,
        base_faces,
        resolutions          
    ):
        transform = Pose.from_translation(positions[i])
        new_vertices = base_vertices * resolutions[i]
        new_vertices = transform.apply(new_vertices)
        return (
            new_vertices,
            base_faces + i*len(new_vertices),
            jnp.tile(colors[i][None,...],(len(base_vertices),1)),
            jnp.tile(colors[i][None,...],(len(base_faces),1)),
        )

    vertices_, faces_, vertex_colors_, face_colors_ = jax.vmap(
        process_ith_ball, in_axes=(0, None, None, None, None, None)
    )(
        jnp.arange(len(grid_centers)),
        grid_centers,
        grid_colors,
        base_vertices,
        base_faces,
        resolutions * 1.0
    )

    vertices = jnp.concatenate(vertices_, axis=0)
    faces = jnp.concatenate(faces_, axis=0)
    vertex_colors = jnp.concatenate(vertex_colors_, axis=0)
    face_colors = jnp.concatenate(face_colors_, axis=0)
    return vertices, faces, vertex_colors, face_colors



def get_rgb_pil_image(image, max=1.0):
    """Convert an RGB image to a PIL image.

    Args:
        image (np.ndarray): RGB image. Shape (H, W, 3).
        max (float): Maximum value for colormap.
    Returns:
        PIL.Image: RGB image visualized as a PIL image.
    """
    image = np.clip(image, 0.0, max)
    if image.shape[-1] == 3:
        image_type = "RGB"
    else:
        image_type = "RGBA"

    img = Image.fromarray(
        np.rint(image / max * 255.0).astype(np.int8),
        mode=image_type,
    ).convert("RGB")
    return img



def make_onehot(n, i, hot=1, cold=0):
    return tuple(cold if j != i else hot for j in range(n))


def multivmap(f, args=None):
    if args is None:
        args = (True,) * len(inspect.signature(f).parameters)
    multivmapped = f
    for i, ismapped in reversed(list(enumerate(args))):
        if ismapped:
            multivmapped = jax.vmap(
                multivmapped, in_axes=make_onehot(len(args), i, hot=0, cold=None)
            )
    return multivmapped


Enumerator = namedtuple(
    "Enumerator",
    [
        "update_choices",
        "update_choices_with_weight",
        "update_choices_get_score",
        "enumerate_choices",
        "enumerate_choices_with_weights",
        "enumerate_choices_get_scores",
    ],
)


def make_enumerator(
    addresses,
):
    def enumerator(trace, key, *args):
        return trace.update(
            key,
            genjax.choice_map({addr: c for (addr, c) in zip(addresses, args)}),
            genjax.Diff.tree_diff_unknown_change(trace.get_args())
        )[0]

    def enumerator_with_weight(trace, key, *args):
        return trace.update(
            key,
            genjax.choice_map({addr: c for (addr, c) in zip(addresses, args)}),
            genjax.Diff.tree_diff_unknown_change(trace.get_args())
        )[:2]

    def enumerator_score(trace, key, *args):
        return enumerator(trace, key, *args).get_score()

    return Enumerator(
        jax.jit(enumerator),
        jax.jit(enumerator_with_weight),
        jax.jit(enumerator_score),
        jax.jit(
            multivmap(
                enumerator,
                (
                    False,
                    False,
                )
                + (True,) * len(addresses),
            )
        ),
        jax.jit(
            multivmap(
                enumerator_with_weight,
                (
                    False,
                    False,
                )
                + (True,) * len(addresses),
            )
        ),
        jax.jit(
            multivmap(
                enumerator_score,
                (
                    False,
                    False,
                )
                + (True,) * len(addresses),
            )
        ),
    )

