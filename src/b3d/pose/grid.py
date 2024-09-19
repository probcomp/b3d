import os

import jax
import jax.numpy as jnp

# for debug+test only
import rerun as rr
from jax.scipy.spatial.transform import Rotation

import b3d
from b3d import Mesh

from .core import Pose


def viz_rotation(pose, vertices, t=0, channel="scene/transformed_mesh"):
    """visualize a rotation of a given mesh vertex set in rerun"""
    b3d.rr_set_time(t)

    # render
    b3d.rr_log_cloud(
        pose.apply(vertices),
        channel,
    )
    rr.log(
        "info",
        rr.TextDocument(
            f"""
            translation: {pose.pos}
            rotation (xyzw): {pose.quaternion}
            """.strip(),
            media_type=rr.MediaType.MARKDOWN,
        ),
    )
    b3d.rr_log_pose(pose, "pose/xfm_pose")


def viz_from_grid(pose_grid, rerun_session_name="grid_test", ycb_obj_id=13):
    # load vertices
    ycb_dir = os.path.join(b3d.get_assets_path(), "bop/ycbv")
    mesh = Mesh.from_obj_file(
        os.path.join(ycb_dir, f'models/obj_{f"{ycb_obj_id + 1}".rjust(6, "0")}.ply')
    ).scale(0.001)
    cam_pose = Pose(
        position=mesh.vertices.mean(axis=0), quaternion=jnp.array([0, 0, 0, 1])
    )  # center the mesh for cleaner viz
    mesh_vertices = cam_pose.inv().apply(mesh.vertices)

    # setup viz
    b3d.rr_init(rerun_session_name)
    b3d.rr_log_pose(Pose.identity(), "pose/default_pose")
    viz_rotation(Pose.identity(), mesh_vertices, 0, "pose/default_pose")

    # visualize
    for t, pose_viz in enumerate(pose_grid):
        viz_rotation(pose_viz, mesh_vertices, t + 1)


def rr_log_pose_arrows_grid(pose_grid, channel="pose_grid", scale=0.02):
    origins = jnp.tile(pose_grid.pos, (3, 1))
    colors = jnp.tile(jnp.eye(3), (len(origins), 1))
    vectors = jax.vmap(lambda pose: pose.as_matrix()[:3, :3].T)(pose_grid) * scale
    vectors = vectors.reshape(-1, 3)
    rr.log(channel, rr.Arrows3D(origins=origins, vectors=vectors, colors=colors))


def make_rotation_grid_enumeration(
    min_angle,
    max_angle,
    n_alpha,
    n_beta,
    n_gamma,
) -> Pose:
    """
    Enumerate rotations via euler angles uniformly gridded in the range [min_angle, max_angle]
    for each axis (angle of axes: X = alpha, Y = beta, Z = gamma)
    """
    alphas = jnp.linspace(min_angle, max_angle, n_alpha)
    betas = jnp.linspace(min_angle, max_angle, n_beta)
    gammas = jnp.linspace(min_angle, max_angle, n_gamma)

    # nest vmap over all axes
    def _inner_proposal(alpha, beta, gamma):
        return Rotation.from_euler("ZYX", jnp.array([gamma, beta, alpha]))

    _proposal_z = lambda gamma, beta: jax.vmap(  # noqa:E731
        _inner_proposal, in_axes=(None, None, 0)
    )(gamma, beta, alphas)
    _proposal_zy = lambda gamma: jax.vmap(_proposal_z, in_axes=(None, 0))(gamma, betas)  # noqa:E731
    proposal_zyx = jax.vmap(_proposal_zy, in_axes=(0,))(gammas)

    return Pose(
        jnp.zeros((n_alpha * n_beta * n_gamma, 3)),
        proposal_zyx.as_quat().reshape(-1, 4),
    )


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
    min_euler_angle: float,
    max_euler_angle: float,
    n_xrot: int,
    n_yrot: int,
    n_zrot: int,
) -> jnp.ndarray:
    tr_grid = make_translation_grid_enumeration(
        min_x, min_y, min_z, max_x, max_y, max_z, nx, ny, nz
    )
    rot_grid = make_rotation_grid_enumeration(
        min_euler_angle, max_euler_angle, n_xrot, n_yrot, n_zrot
    )

    compose_pose = lambda tr, rot: pose_center.compose(tr).compose(rot)  # noqa:E731
    inner_vmap = lambda tr: jax.vmap(compose_pose, in_axes=(None, 0))(  # noqa:E731
        tr, rot_grid
    )
    _total_grid = jax.vmap(inner_vmap, in_axes=(0,))(tr_grid)  # vmap over tr

    total_grid = _total_grid.reshape(-1)

    return total_grid


if __name__ == "__main__":
    # center pose
    xyz0 = jnp.array([0.0, 0.0, 0.0])
    rot0 = jnp.array([0.0, 0.0, 0.0, 1.0])
    pose0 = Pose(xyz0, rot0)

    # translation grid
    nx = ny = nz = 5
    ntr = nx * ny * nz
    print(f"Generating {ntr} translations")
    min_x, min_y, min_z = -0.1, -0.1, 0.001
    max_x, max_y, max_z = 0.1, 0.1, 0.2

    # rotation grid
    n_alpha = n_beta = n_gamma = 5
    nrot = n_alpha * n_beta * n_gamma
    print(f"Generating {nrot} rotations")
    min_euler, max_euler = -jnp.pi / 6, jnp.pi / 6

    ##################################
    # Translation / Rotation grids
    ##################################
    tr_grid = make_translation_grid_enumeration(
        min_x, min_y, min_z, max_x, max_y, max_z, nx, ny, nz
    )
    rot_grid = make_rotation_grid_enumeration(
        min_euler, max_euler, n_alpha, n_beta, n_gamma
    )

    ##################################
    # Whole pose grid
    ##################################

    pose_grid_jit = jax.jit(
        pose_grid, static_argnames=("nx", "ny", "nz", "n_xrot", "n_yrot", "n_zrot")
    )
    grid = pose_grid_jit(
        pose0,
        min_x=min_x,
        min_y=min_y,
        min_z=min_z,
        max_x=max_x,
        max_y=max_y,
        max_z=max_z,
        nx=nx,
        ny=ny,
        nz=ny,
        min_euler_angle=min_euler,
        max_euler_angle=max_euler,
        n_xrot=n_alpha,
        n_yrot=n_beta,
        n_zrot=n_gamma,
    )

    ##################################
    # Test correctness
    ##################################
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

    # viz_from_grid(grid, rerun_session_name=f"GRID_{ntr}_{nrot}", ycb_obj_id=13)

    # ##################################
    # # Test time
    # ##################################
    import time

    start = time.time()
    timed_grid = pose_grid_jit(
        pose0,
        min_x=min_x,
        min_y=min_y,
        min_z=min_z,
        max_x=max_x,
        max_y=max_y,
        max_z=max_z,
        nx=nx,
        ny=ny,
        nz=ny,
        min_euler_angle=min_euler,
        max_euler_angle=max_euler,
        n_xrot=n_alpha,
        n_yrot=n_beta,
        n_zrot=n_gamma,
    )
    end = time.time()
    print(f"Time taken: {(end-start)*1000} milliseconds for {ntr*nrot} poses")

    ### Visualize ###
    viz_grid = pose_grid_jit(
        pose0,
        min_x=min_x,
        min_y=min_y,
        min_z=min_z,
        max_x=max_x,
        max_y=max_y,
        max_z=max_z,
        nx=nx,
        ny=ny,
        nz=ny,
        min_euler_angle=min_euler,
        max_euler_angle=max_euler,
        n_xrot=1,
        n_yrot=1,
        n_zrot=1,
    )
    b3d.rr_init("pose_grid_test")
    rr_log_pose_arrows_grid(viz_grid)
    b3d.rr_log_pose(pose0, channel="original_pose")
