import jax
import jax.numpy as jnp

from .core import Pose


def angle_axis_helper_edgecase(newZ):
    zUnit = jnp.array([0.0, 0.0, 1.0])
    axis = jnp.array([0.0, 1.0, 0.0])
    geodesicAngle = jax.lax.cond(
        jnp.allclose(newZ, zUnit, atol=1e-3),
        lambda: 0.0,
        lambda: jnp.pi,  # noqa: E731
    )
    return axis, geodesicAngle


def angle_axis_helper(newZ):
    zUnit = jnp.array([0.0, 0.0, 1.0])
    axis = jnp.cross(zUnit, newZ)
    theta = jax.lax.asin(jax.lax.clamp(-1.0, jnp.linalg.norm(axis), 1.0))
    geodesicAngle = jax.lax.cond(
        jnp.dot(zUnit, newZ) > 0,
        lambda: theta,
        lambda: jnp.pi - theta,  # noqa: E731
    )
    return axis, geodesicAngle


def axis_angle_to_quaternion(axis, angle):
    # Normalize the axis to ensure it's a unit vector
    direction = axis / jnp.linalg.norm(axis)

    # Half-angle for quaternion computation
    half_angle = angle / 2.0
    sina = jnp.sin(half_angle)
    cosa = jnp.cos(half_angle)

    # Quaternion components
    q_w = cosa  # Scalar part
    q_x = direction[0] * sina  # x-component
    q_y = direction[1] * sina  # y-component
    q_z = direction[2] * sina  # z-component

    # Return quaternion (scalar last)
    return jnp.array([q_x, q_y, q_z, q_w])


def geodesicHopf_rotate_within_axis(newZ, planarAngle):
    """
    Rotate an axis in 3D space by a planar angle
    """
    # newZ should be a normalized vector
    # returns a 4x4 quaternion
    zUnit = jnp.array([0.0, 0.0, 1.0])

    # todo: implement cases where newZ is approx. -zUnit or approx. zUnit
    axis, geodesicAngle = jax.lax.cond(
        jnp.allclose(jnp.abs(newZ), zUnit, atol=1e-3),
        angle_axis_helper_edgecase,
        angle_axis_helper,
        newZ,
    )
    left_pose = Pose(
        jnp.array([0, 0, 0]), axis_angle_to_quaternion(axis, geodesicAngle)
    )
    right_pose = Pose(
        jnp.array([0, 0, 0]), axis_angle_to_quaternion(zUnit, planarAngle)
    )

    return left_pose @ right_pose


def fibonacci_sphere(samples_in_range, phi_range=jnp.pi):
    ga = jnp.pi * (jnp.sqrt(5) - 1)  # golden angle
    eps = 1e-10
    min_y = jnp.cos(phi_range)

    samples = jnp.round(samples_in_range * (2 / (1 - min_y + eps)))

    def fib_point(i):
        y = 1 - (i / (samples - 1)) * 2  # goes from 1 to -1
        radius = jnp.sqrt(1 - y * y)
        theta = ga * i
        x = jnp.cos(theta) * radius
        z = jnp.sin(theta) * radius
        return jnp.array([x, z, y])

    fib_sphere = jax.vmap(fib_point, in_axes=(0))
    points = jnp.arange(samples_in_range)
    return fib_sphere(points)


def make_rotation_grid_enumeration(
    num_fibonacci_sphere_points,  # 5
    num_axis_rotations,  # 2
    min_rot_angle,
    max_rot_angle,
    sphere_angle_range,
) -> Pose:
    """
    Generate uniformly spaced rotation proposals around a constrained region of SO(3)

    Params:
    num_fibonacci_sphere_points: number of rotation axes to sample, on the region of the fibonacci sphere specified by `sphere_angle_range`
    num_axis_rotations: number of in-axis rotations to sample, in the interval [min_rot_angle, max_rot_angle]
    min_rot_angle, max_rot_angle: the minimum and maximum rotation angle values; max_rot_angle - min_rot_angle leq 2*pi
    sphere_angle_range: the maximum phi angle (in spherical coordinates) that bounds the region of the fibonacci sphere to sample rotation axes from; sphere_angle_range leq pi

    Returns:
    rotation proposals: b3d.Pose type with length (fib_sample*rot_sample)
    """
    unit_sphere_directions = fibonacci_sphere(
        num_fibonacci_sphere_points, sphere_angle_range
    )
    in_axis_spin = jnp.linspace(min_rot_angle, max_rot_angle, num_axis_rotations)

    inner_vmap = lambda axis_rot: jax.vmap(  # noqa:E731
        geodesicHopf_rotate_within_axis, in_axes=(None, 0)
    )(axis_rot, in_axis_spin)

    _proposals = jax.vmap(inner_vmap, in_axes=(0,))(unit_sphere_directions)
    proposals = _proposals.reshape(num_fibonacci_sphere_points * num_axis_rotations)

    return proposals


def make_translation_grid_enumeration(
    min_x, min_y, min_z, max_x, max_y, max_z, num_x, num_y, num_z
):
    """
    Generate uniformly spaced translation proposals in a 3D box
    Args:
        min_x, min_y, min_z: minimum x, y, z values
    """
    deltas = jnp.stack(
        jnp.meshgrid(
            jnp.linspace(min_x, max_x, num_x),
            jnp.linspace(min_y, max_y, num_y),
            jnp.linspace(min_z, max_z, num_z),
        ),
        axis=-1,
    )
    deltas = deltas.reshape((-1, 3), order="F")

    return Pose(deltas, jnp.tile(Pose.identity_quaternion, (num_x * num_y * num_z, 1)))


def pose_grid(
    pose_center: Pose,
    min_x: float,
    min_y: float,
    min_z: float,
    max_x: float,
    max_y: float,
    max_z: float,
    nx: int,
    ny: int,
    nz: int,
    min_r: float,
    max_r: float,
    d_sph: float,
    nfib: int,
    naxis: int,
) -> jnp.ndarray:
    tr_grid = make_translation_grid_enumeration(
        min_x, min_y, min_z, max_x, max_y, max_z, nx, ny, nz
    )
    rot_grid = make_rotation_grid_enumeration(nfib, naxis, min_r, max_r, d_sph)

    compose_pose = lambda tr, rot: pose_center.compose(tr).compose(rot)  # noqa:E731
    inner_vmap = lambda tr: jax.vmap(compose_pose, in_axes=(None, 0))(  # noqa:E731
        tr, rot_grid
    )
    _total_grid = jax.vmap(inner_vmap, in_axes=(0,))(tr_grid)  # vmap over tr

    total_grid = _total_grid.reshape(-1)

    return total_grid


if __name__ == "__main__":
    import time

    xyz0 = jnp.array([0.0, 0.0, 0.0])
    rot0 = jnp.array([0.0, 0.0, 0.0, 1.0])
    pose0 = Pose(xyz0, rot0)

    # grid size (translation)
    nx, ny, nz = 5, 5, 5
    ntr = nx * ny * nz

    # grid size (rotation)
    nfib, naxis = 10, 10
    nrot = nfib * naxis

    pose_grid_jit = jax.jit(
        pose_grid, static_argnames=("nx", "ny", "nz", "nfib", "naxis")
    )
    grid = pose_grid_jit(
        pose0,
        min_x=-1,
        min_y=-1,
        min_z=-1,
        max_x=1,
        max_y=1,
        max_z=1,
        nx=nx,
        ny=ny,
        nz=ny,
        min_r=-jnp.pi,
        max_r=jnp.pi,
        d_sph=jnp.pi,
        nfib=nfib,
        naxis=naxis,
    )

    # sanity check sizes
    assert isinstance(
        grid, Pose
    ), f"Wrong return type; expected b3d.Pose, got {type(grid)}"
    assert grid.pos.shape == (
        ntr * nrot,
        3,
    ), f"Wrong shape for pos; expected {(ntr * nrot, 3)}, got {grid.pos.shape}"
    assert grid.quaternion.shape == (
        ntr * nrot,
        4,
    ), f"Wrong shape for quat; expected {(ntr * nrot, 4)}, got {grid.quaternion.shape}"

    start = time.time()
    timed_grid = pose_grid_jit(
        pose0,
        min_x=-1,
        min_y=-1,
        min_z=-1,
        max_x=1,
        max_y=1,
        max_z=1,
        nx=nx,
        ny=ny,
        nz=ny,
        min_r=-jnp.pi,
        max_r=jnp.pi,
        d_sph=jnp.pi,
        nfib=nfib,
        naxis=naxis,
    )
    end = time.time()
    print(f"Time taken: {(end-start)*1000} milliseconds for {ntr*nrot} poses")
