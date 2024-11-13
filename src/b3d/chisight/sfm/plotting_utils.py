import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
from matplotlib.collections import LineCollection
from sklearn.utils import Bunch

from b3d.utils import downsize_images


def quick_plot_images(imgs, ax=None, figsize=(3, 3), downsize=10):
    """Plot an overview of a list of images."""
    n = imgs.shape[0]
    figsize = (figsize[0] * n, figsize[1])
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_aspect(1)
        ax.axis("off")

    imgs_ = downsize_images(imgs, downsize)
    ax.imshow(np.concatenate(imgs_, axis=1))

    return ax.get_figure(), ax


def create_box_mesh(dims=np.array([1.0, 1.0, 1.0])):
    """
    Create a box mesh (tuple of vertices, triangle indices, and vertex normals) with the given dimensions.
    """
    # Define the 8 vertices of the box
    w, h, d = dims / 2.0
    vertex_positions = np.array(
        [
            [-w, -h, -d],
            [w, -h, -d],
            [w, h, -d],
            [-w, h, -d],
            [-w, -h, d],
            [w, -h, d],
            [w, h, d],
            [-w, h, d],
        ]
    )

    # Define the 12 triangles (two per face)
    triangle_indices = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],  # Front face
            [4, 5, 6],
            [4, 6, 7],  # Back face
            [0, 1, 5],
            [0, 5, 4],  # Bottom face
            [2, 3, 7],
            [2, 7, 6],  # Top face
            [0, 3, 7],
            [0, 7, 4],  # Left face
            [1, 2, 6],
            [1, 6, 5],  # Right face
        ]
    )
    vertex_normals = vertex_positions

    return vertex_positions, triangle_indices, vertex_normals


def create_box_mesh2(dims=np.array([1.0, 1.0, 1.0])):
    """
    Create a box mesh (tuple of vertices, triangle indices, and vertex normals) with the given dimensions.
    This version has a bigger front face.
    """
    # Define the 8 vertices of the box
    w, h, d = dims / 2.0
    vertex_positions = np.array(
        [
            [-w, -h, -d],
            [w, -h, -d],
            [w, h, -d],
            [-w, h, -d],
            [-2 * w, -2 * h, d],
            [2 * w, -2 * h, d],
            [2 * w, 2 * h, d],
            [-2 * w, 2 * h, d],
        ]
    )

    # Define the 12 triangles (two per face)
    triangle_indices = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],  # Front face
            [4, 5, 6],
            [4, 6, 7],  # Back face
            [0, 1, 5],
            [0, 5, 4],  # Bottom face
            [2, 3, 7],
            [2, 7, 6],  # Top face
            [0, 3, 7],
            [0, 7, 4],  # Left face
            [1, 2, 6],
            [1, 6, 5],  # Right face
        ]
    )
    vertex_normals = vertex_positions

    return vertex_positions, triangle_indices, vertex_normals


def create_pose_bunch(
    p, c=jnp.array([0.7, 0.7, 0.7]), s=1.0, dims=np.array([0.2, 0.2, 1.0])
):
    vs, fs, ns = create_box_mesh2(dims=s * dims)

    if c is None:
        c = jnp.array([0.7, 0.7, 0.7])
    cs = c[None, :] * jnp.ones((vs.shape[0], 1))

    return Bunch(
        vertex_positions=p(vs),
        triangle_indices=fs,
        vertex_normals=p.rot.apply(ns),
        vertex_colors=cs,
    )


def log_pose(
    s, p, c=jnp.array([0.7, 0.7, 0.7]), scale=1.0, dims=np.array([0.2, 0.2, 1.0])
):
    rr.log(
        s,
        rr.Mesh3D(
            **create_pose_bunch(p, c=c, s=scale, dims=dims),
            # mesh_material=rr.components.Material(albedo_factor=[255, 255, 255]),
        ),
    )


def plot_segments(segs, c=None, ax=None, lw=1, alpha=1.0):
    """
    Example:
    ```
    N = 10
    segs = np.random.rand(N, 2, 2)
    cols = np.random.rand(N, 3)

    plot_segments(segs, c=cols, alpha=1.)
    ```
    """
    if ax is None:
        ax = plt.gca()

    if c is None:
        c = "r"

    # Create a LineCollection object
    lc = LineCollection(segs, colors=c, linewidths=lw, alpha=alpha)
    ax.add_collection(lc)
    return ax


def intersect_line_with_image(ell, h, w):
    left = jnp.array([0, -ell[2] / ell[1]])
    right = jnp.array([w, -(w * ell[0] + ell[2]) / ell[1]])
    bottom = jnp.array([-ell[2] / ell[0], 0])
    top = jnp.array([-(h * ell[1] + ell[2]) / ell[0], h])
    points = [left, right, bottom, top]
    points = [p for p in points if 0 <= p[0] <= w and 0 <= p[1] <= h]
    if len(points) == 2:
        return jnp.array(points), True
    else:
        return jnp.array([jnp.inf, jnp.inf]), False
