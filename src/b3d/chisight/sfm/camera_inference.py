"""
Computation of the camera matrix from world points and their projections.

References:
> Terzakis--Lourakis, "A Consistently Fast and Globally Optimal Solution to the Perspective-n-Point Problem"
> Hartley--Zisserman, "Multiple View Geometry in Computer Vision", 2nd ed.
"""

from jax import numpy as jnp

from b3d.types import Matrix3x4, Point3D


def solve_camera_projection_constraints(Xs: Point3D, ys: Point3D) -> Matrix3x4:
    """
    Solve for the camera projection matrix given 3D points and their 2D projections,
    as described in Chapter 7 ("Computation of the Camera Matrix P") of
    > Hartley--Zisserman, "Multiple View Geometry in Computer Vision" (2nd ed).

    Args:
        Xs: 3D points in world coordinates, shape (N, 3).
        ys: Normalized image coordinates, shape (N, 2).

    Returns:
        Camera projection matrix, shape (3, 4).
    """
    # We change notation from B3D notation
    # to Hartley--Zisserman, for easy of comparison
    X = Xs
    x = ys[:, 0]
    y = ys[:, 1]
    w = ys[:, 2]
    n = Xs.shape[0]

    # Cf. Hartley--Zisserman, Eqn. 7.1
    _0 = jnp.zeros(3)
    A = jnp.concatenate(
        [
            jnp.block(
                [
                    [_0, -w[i] * X[i], y[i] * X[i]],
                    [w[i] * X[i], _0, -x[i] * X[i]],
                    [-y[i] * X[i], x[i] * X[i], _0],
                ]
            )
            for i in jnp.arange(n)
        ],
        axis=0,
    )

    _, _, vt = jnp.linalg.svd(A)
    P = vt[-1].reshape(3, 4)
    return P
