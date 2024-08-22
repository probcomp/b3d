"""
Implementation of the eight point algorithm and other relevant pieces.

Notation and terminology:
    Normalized image coordinates: "Normalized" means "intrinsics-free" coordinates, which 
        in our language are just the coordinates in the camera frame.
    `x`: World coordinates
    `y`: Camera coordinates (3D), i.e. normalized image coordinates.
    `uv` or sometimes only `u`: Sensor coordinates (2D)
    `intr`: Camera intrinsics

Example:
```
from b3d.sfm.eight_point import normalized_eight_point, poses_from_essential

# Load data and so on
...

# Choose a pair of frames
t0 = 0
t1 = 1

# Choose a subset of keypoints to run the algorithm on
key = keysplit(key)
sub = jax.random.choice(key, jnp.where(vis[t0]*vis[t1])[0], (8,), replace=False)

# Normalized image coordinates from sensor coordinates
ys0 = camera.camera_from_screen(uvs[t0,sub], intr)
ys1 = camera.camera_from_screen(uvs[t1,sub], intr)

# Estimate essential matrix and extract 
# possible choices for the second camera pose
E = normalized_eight_point(ys0, ys1)
ps = poses_from_essential(E)
```
"""
from typing import Tuple

import jax
from jax import numpy as jnp

from b3d.pose import Pose
from b3d.types import Array, Int, Matrix3x3, Matrix3x4, Point3D


def cross_product_matrix(a) -> Matrix3x3:
    """
    Returns matrix A such that for any vector b, 
    the cross product of a and b is given by Ab.
    """
    # > https://en.wikipedia.org/wiki/Cross_product#Conversion_to_matrix_multiplication
    return jnp.array([
        [    0, -a[2],  a[1]],
        [ a[2],     0, -a[0]],
        [-a[1],  a[0],     0]
    ])

def essential_from_pose(p) -> Matrix3x3:
    """
    Essential matrix from B3D camera pose.
    
    Args:
        `p`: Camera pose

    Returns:
        Essential matrix
    """
    # Two things to note:
    # 
    # 1) A camera projection matrix [R | t] that maps 
    #   *world* coordinates x to *camera* coordinates y = Rx + t
    #   corresponds to a B3D camera pose with rotation `R.T` and 
    #   translation `- R.T t` (this is the inverse of [R | t]). 
    # 
    # 2) Recall that the essential matrix for a camera 
    #   projection matrix [R | t] is given by E = [t] R, 
    #   where [t] denotes the matrix representation of the 
    #   cross product with t
    # 
    # Therefore, the essential matrix for a B3D camera pose p = Pose(x,q)
    # is given by E = [-Q^T x] Q^T, where x is the position and Q 
    # is the rotation matrix.
    x = p.pos
    Q = p.rot.as_matrix()
    return cross_product_matrix(- Q.T @ x) @ Q.T

def camera_projection_from_pose(p) -> Tuple[Matrix3x3, Point3D]:
    """
    Returns camera projection data R,t from from B3D camera pose.
    
    Args:
        `p`: Camera pose

    Returns:
        Camera projection matrix entries R,t
    """
    # Note: A camera projection matrix [R | t] that maps 
    #   *world* coordinates x to *camera* coordinates y = Rx + t
    #   corresponds to a B3D camera pose with rotation `R.T` and 
    #   translation `- R.T t` (this is the inverse of [R | t]). 
    x = p.pos
    Q = p.rot.as_matrix()
    return Q.T, - Q.T @ x

def camera_projection_matrix(p: Pose) -> Matrix3x4:
    """
    Returns camera projection matrix P = [R | t] from from B3D camera pose.
    
    Args:
        `p`: Camera pose

    Returns:
        Camera projection matrix P = [R | t]
    """
    # Note: A camera projection matrix [R | t] that maps 
    #   *world* coordinates x to *camera* coordinates y = Rx + t
    #   corresponds to a B3D camera pose with rotation `R.T` and 
    #   translation `- R.T t` (this is the inverse of [R | t]). 
    x = p.pos
    Q = p.rot.as_matrix()
    return jnp.concatenate([Q.T, (- Q.T @ x)[:, None]], axis=1)


def poses_from_essential(E: Matrix3x3) -> Pose:
    """
    Extract the 4 possible choices for the second camera matrix 
    from essential matrix.

    Args:
        `E`: Essential matrix
    
    Returns:
        Stacked camera poses
    """
    # According to [Hartley-Zisserman, "Multiple View Geometry" (2nd ed.), Result 9.19.], 
    # for a given essential matrix E = U diag(1, 1, 0)VT, and ﬁrst camera matrix P = [I | 0], 
    # there are four possible choices for the second camera matrix P, namely
    #   P = [UWVT | +u3] or [UWVT |−u3] or [UWTVT | +u3] or [UWTVT |−u3].
    u, _, vt = jnp.linalg.svd(E)
    x = u[:, -1]
    w = jnp.array([
        [ 0., -1., 0.],
        [ 1.,  0., 0.],
        [ 0.,  0., 1.]
    ])
    r0 = u @ w   @ vt
    r1 = u @ w.T @ vt

    # Make sure these are orientation preserving; det(u) or det(vt) may be -1
    # but we are enforcing the equality of the first two singular values so
    # we can flip the first two rows / columns to flip the sign.
    r0 *= jnp.sign(jnp.linalg.det(r0))
    r1 *= jnp.sign(jnp.linalg.det(r1))

    # Recall that a **camera projection** matrix [R | t] that maps 
    # world coordinates x to camera coordinates y = [R | t] hom(x) 
    # corresponds to a B3D **camera pose** with rotation `R.T` and 
    # translation `- R.T t`.  
    return Pose.stack_poses([
        Pose.from_pos_matrix(-r0.T @ x, r0.T),
        Pose.from_pos_matrix(-r1.T @ x, r1.T),
        Pose.from_pos_matrix( r0.T @ x, r0.T),
        Pose.from_pos_matrix( r1.T @ x, r1.T),
    ])

extract_poses = poses_from_essential

def solve_epipolar_constraints(ys0: Point3D, ys1: Point3D) -> Matrix3x3:
    """
    Returns the essential matrix that minimizes 
    the epipolar constraint `y1.T E y0`.
    """
    # We want to solve `y1.T E y0 = 0`, which can be 
    # rewritten as `(y0 kronecker Y1).T vec(E) = 0`,
    # where "kronecker" denotes the Kronecker product and 
    # vec(E) is the vectorized form of E.
    # 
    # Useful references:
    # > https://en.wikipedia.org/wiki/Eight-point_algorithm#Step_1:_Formulating_a_homogeneous_linear_equation
    # > https://en.wikipedia.org/wiki/Kronecker_product
    Y = jax.vmap(jnp.kron)(ys1, ys0)
    _, _, vt = jnp.linalg.svd(Y) 
    # Solution is the minimal right singular vector
    e = vt[-1].reshape((3, 3))
    return e

def epipolar_errors(E: Matrix3x3, ys0: Point3D, ys1: Point3D) -> Array:
    """
    Compute the epipolar errors for a given essential matrix and 
    normalized image coordinates.
    
    Args:
        `E`: Essential matrix
        `ys0`: Normalized image coordinates at time 0
        `ys1`: Normalized image coordinates at time 1

    Returns:
        Array of Epipolar errors
    """
    return jax.vmap(lambda y0, y1: jnp.abs(y1.T @ E @ y0))(ys0, ys1)

def enforce_internal_constraint(E_est:Matrix3x3) -> Matrix3x3:
    """Enforce the fundamental matrix rank constraint."""
    # > https://en.wikipedia.org/wiki/Eight-point_algorithm#Step_3:_Enforcing_the_internal_constraint
    u, s, vt = jnp.linalg.svd(E_est)
    return u @ jnp.diag(jnp.array([s[0], s[1], 0.])) @ vt  

def normalize_hartley(x: Array) -> tuple[Array, Array]:
    """
    Normalize a homogeneous batch to mean zero and range from -1 to 1;
    as suggested in
    > Hartley, "In defense of the eight point algorithm", 1997.
    """
    x /= x[:, -1:]
    u = x[:, :-1]
    means = jnp.mean(u, axis=0)
    maxes = jnp.max(jnp.abs(u - means), axis=0)
    normalizer = jnp.diag(jnp.append(1 / maxes, 1.)).at[:-1, -1].set(
        -means / maxes)
    return (x @ normalizer.T, normalizer)

def _normalized_eight_point(ys0, ys1):
    """
    Normalized 8-point algorithm estimating the essential matrix, 
    as described in 
    > Hartley-Zisserman, "Multiple view geometry in computer vision", 2nd ed., Algorithm 11.1.

    Args:
        `ys0`: Normalized 3D image coordinates at time 0 
        `ys1`: Normalized 3D image coordinates at time 1
    
    Returns:
        Estimated essential matrix
    """
    ys0, T0 = normalize_hartley(ys0)
    ys1, T1 = normalize_hartley(ys1)
    E = solve_epipolar_constraints(ys0, ys1) 
    E = enforce_internal_constraint(E)
    E = T1.T @ E @ T0
    return E

normalized_eight_point = jax.jit(_normalized_eight_point)

def _triangulate_linear_hartley(cam0, cam1, y0, y1) -> Point3D:
    """
    Linear triangulation method as described in 
    > Hartley-Zisserman, "Multiple view geometry in computer vision", 2nd ed.; Section 12.2
    
    Args:
        `cam0`: Camera pose at time 0
        `cam1`: Camera pose at time 1
        `y0`: Normalized image coordinates of keypoint at time 0 
        `y1`: Normalized image coordinates of keypoint at time 1

    Returns:
        Inferred world point.
    """
    # We follow Section 12.2 in Hartley-Zisserman.
    # First we map to notation from Hartley-Zisserman. 
    # Here "underscore" reads "prime", i.e. 
    # `x_` translates to `x'`
    P  = camera_projection_matrix(cam0)
    P_ = camera_projection_matrix(cam1)
    x , y  = y0[:2]
    x_, y_ = y1[:2]
    A = jnp.array([
        x *P[2] - P[0],
        y *P[2] - P[1],
        x_*P_[2] - P_[0],
        y_*P_[2] - P_[1]
    ])
    _, _, vt = jnp.linalg.svd(A)
    X = vt[-1]    
    # TODO: Obviously there is a problem when X[3] is zero. Review this.
    #   Hartley-Zisserman address this I think.
    return X[:3] / X[3]

triangulate_linear_hartley = jax.vmap(_triangulate_linear_hartley, (None,None,0,0))

def in_front_count(cam0, cam1, xs_world: Point3D) -> Int:
  """Count the world points that are in front of both cameras."""
  ys0 = cam0.inv()(xs_world)
  ys1 = cam1.inv()(xs_world)
  return jnp.sum((ys0[:,2] > 0) & (ys1[:,2] > 0))

def find_best_chirality(cams, ys0, ys1):
   xss = jax.vmap(triangulate_linear_hartley, (None,0,None,None))(Pose.id(), cams, ys0, ys1)
   counts = jax.vmap(in_front_count, (None,0,0))(Pose.id(), cams, xss)
   i = jnp.argmax(counts)
   return cams[i], xss[i]

# TODO:
#   - Check triangulation angles ("orthogonality score")
#   - Asses posterior

def _triangulate_linear_midpoint(cam0, cam1, y0, y1):
    """
    Returns the mid point of the line segment 
    between the two rays through the keypoints.

    Args:
        `cam0`: Camera pose at time 0
        `cam1`: Camera pose at time 1
        `y0`: Camera coordinates of keypoint at time 0 
        `y1`: Camera coordinates of keypoint at time 1

    Returns:
        Inferred world point.
    """
    # We need to solve
    #   c0 + s0*v0 = c1 + s1*v1, 
    # where ci are the camera positions in the world and vi 
    # are world vectors through the image keypoints.
    v0 = cam0(y0) - cam0.pos
    v1 = cam1(y1) - cam1.pos

    V = jnp.stack([v0, -v1], axis=1)
    c = cam1.pos - cam0.pos

    s = jnp.linalg.pinv(V)@c[:,None]

    xs = jnp.array([
        cam0.pos + s[0]*v0,
        cam1.pos + s[1]*v1
    ])
    x = xs.mean(0)
    return x

triangulate_linear_midpoint = jax.vmap(_triangulate_linear_midpoint, (None,None,0,0))