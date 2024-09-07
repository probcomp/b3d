import jax.numpy as jnp
import numpy as np
import rerun as rr
from sklearn.utils import Bunch


def create_box_mesh(dims=np.array([1.0, 1.0, 1.0])):
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
