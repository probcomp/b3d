import os

import jax
import jax.numpy as jnp

# for debug+test only
import rerun as rr
from jax.scipy.spatial.transform import Rotation

import b3d
from b3d import Mesh

from .core import Pose


def viz_rotation(pose, vertices, t=0, channel="mesh/xfm_cloud"):
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
    b3d.rr_log_pose(pose, "axes/xfm_pose_axes")


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

    # log the default pose
    b3d.rr_set_time(0)
    b3d.rr_log_pose(Pose.identity(), "axes/default_pose_axes")
    viz_rotation(Pose.identity(), mesh_vertices, 0, "mesh/default_pose_cloud")

    # visualize
    for t, pose_viz in enumerate(pose_grid):
        viz_rotation(pose_viz, mesh_vertices, t + 1, "mesh/xfm_cloud")


def sorted_linspace(delta, half_num):
    if half_num == 0:  # only the zero-transform sample is returned
        return jnp.array([0.0])

    num_samples = half_num * 2 + 1
    linspace = jnp.linspace(-delta, delta, num_samples)
    ordered_linspace = linspace[jnp.argsort(jnp.abs(linspace))]
    return ordered_linspace


def rr_log_pose_arrows_grid(pose_grid, channel="pose_grid", scale=0.02):
    origins = jnp.tile(pose_grid.pos, (3, 1))
    colors = jnp.tile(jnp.eye(3), (len(origins), 1))
    vectors = jax.vmap(lambda pose: pose.as_matrix()[:3, :3].T)(pose_grid) * scale
    vectors = vectors.reshape(-1, 3)
    rr.log(channel, rr.Arrows3D(origins=origins, vectors=vectors, colors=colors))


def make_rotation_grid_enumeration(
    half_d_angle,
    half_n_alpha,
    half_n_beta,
    half_n_gamma,
) -> Pose:
    """
    Enumerate rotations via euler angles uniformly gridded in the range [min_angle, max_angle]
    for each axis (angle of axes: X = alpha, Y = beta, Z = gamma)
    """
    alphas = sorted_linspace(half_d_angle, half_n_alpha)
    betas = sorted_linspace(half_d_angle, half_n_beta)
    gammas = sorted_linspace(half_d_angle, half_n_gamma)

    # nest vmap over all axes
    def _inner_proposal(alpha, beta, gamma):
        return Rotation.from_euler("ZYX", jnp.array([gamma, beta, alpha]))

    _proposal_z = lambda gamma, beta: jax.vmap(  # noqa:E731
        _inner_proposal, in_axes=(None, None, 0)
    )(gamma, beta, alphas)
    _proposal_zy = lambda gamma: jax.vmap(_proposal_z, in_axes=(None, 0))(gamma, betas)  # noqa:E731
    proposal_zyx = jax.vmap(_proposal_zy, in_axes=(0,))(gammas)

    n_alpha, n_beta, n_gamma = (
        2 * half_n_alpha + 1,
        2 * half_n_beta + 1,
        2 * half_n_gamma + 1,
    )
    return Pose(
        jnp.zeros((n_alpha * n_beta * n_gamma, 3)),
        proposal_zyx.as_quat().reshape(-1, 4),
    )


def make_translation_grid_enumeration(
    half_dx, half_dy, dz, half_num_x, half_num_y, half_num_z
) -> Pose:
    """
    Generate uniformly spaced translation proposals in a 3D box
    Args:
        half_dx, half_dy, dz: half-dimension of each of the x, y, z directions
        half_num_x, half_num_y, half_num_z: samples in each of the dimensions, EXCLUDING the zero sample.
    """
    x_space = sorted_linspace(half_dx, half_num_x)
    y_space = sorted_linspace(half_dy, half_num_y)
    z_space = sorted_linspace(dz, half_num_z)
    deltas = jnp.stack(
        jnp.meshgrid(
            x_space,
            y_space,
            z_space,
        ),
        axis=-1,
    )
    deltas = deltas.reshape((-1, 3), order="F")

    num_x, num_y, num_z = 2 * half_num_x + 1, 2 * half_num_y + 1, 2 * half_num_z + 1
    return Pose(deltas, jnp.tile(Pose.identity_quaternion, (num_x * num_y * num_z, 1)))


def pose_grid(
    pose_center: Pose,
    half_dx=float,
    half_dy=float,
    half_dz=float,
    half_nx=int,
    half_ny=int,
    half_nz=int,
    half_dangle=float,
    half_n_xrot=int,
    half_n_yrot=int,
    half_n_zrot=int,
) -> Pose:
    tr_grid = make_translation_grid_enumeration(
        half_dx, half_dy, half_dz, half_nx, half_ny, half_nz
    )
    rot_grid = make_rotation_grid_enumeration(
        half_dangle, half_n_xrot, half_n_yrot, half_n_zrot
    )

    compose_pose = lambda tr, rot: pose_center.compose(tr).compose(rot)  # noqa:E731
    inner_vmap = lambda tr: jax.vmap(compose_pose, in_axes=(None, 0))(  # noqa:E731
        tr, rot_grid
    )
    _total_grid = jax.vmap(inner_vmap, in_axes=(0,))(tr_grid)  # vmap over tr

    total_grid = _total_grid.reshape(-1)

    return total_grid


if __name__ == "__main__":
    #################################
    # Setup test
    #################################
    VIZ_TEST = False  # toggle to visualize all grid on rerun

    # center pose
    xyz0 = jnp.array([0.0, 0.0, 0.0])
    rot0 = jnp.array([0.0, 0.0, 0.0, 1.0])
    pose0 = Pose(xyz0, rot0)

    # translation grid
    half_nx = half_ny = half_nz = 2
    ntr = (2 * half_nx + 1) * (2 * half_ny + 1) * (2 * half_nz + 1)
    print(f"Generating {ntr} translations")
    half_dx, half_dy, half_dz = 1, 1, 0.001

    # rotation grid
    half_n_alpha = half_n_beta = half_n_gamma = 2
    half_d_euler = jnp.pi / 3
    nrot = (2 * half_n_alpha + 1) * (2 * half_n_beta + 1) * (2 * half_n_gamma + 1)
    print(f"Generating {nrot} rotations")

    ##################################
    # Generate pose grid
    ##################################

    pose_grid_jit = jax.jit(
        pose_grid,
        static_argnames=(
            "half_nx",
            "half_ny",
            "half_nz",
            "half_n_xrot",
            "half_n_yrot",
            "half_n_zrot",
        ),
    )
    grid = pose_grid_jit(
        pose0,
        half_dx=half_dx,
        half_dy=half_dy,
        half_dz=half_dz,
        half_nx=half_nx,
        half_ny=half_ny,
        half_nz=half_nz,
        half_dangle=half_d_euler,
        half_n_xrot=half_n_alpha,
        half_n_yrot=half_n_beta,
        half_n_zrot=half_n_gamma,
    )

    ##################################
    # Test correctness
    ##################################
    # 1a. sanity check sizes
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

    # 1b. check that original pose is in grid
    assert (
        grid.pos.tolist().index(pose0.pos.tolist())
        == grid.quat.tolist().index(pose0.quat.tolist())
        != -1
    ), "Center pose not in grid"
    print("Size checks and center-pose checks passed")

    # 2. visualize grid
    if VIZ_TEST:
        print(f"Visualizing {ntr*nrot} poses in rerun...")
        viz_from_grid(grid, rerun_session_name=f"GRID_{ntr}_{nrot}", ycb_obj_id=13)
    else:
        print("Skipping visualization...")

    # ##################################
    # # Test time
    # ##################################
    import time

    print("Testing jitted runtime...")
    start = time.time()
    timed_grid = pose_grid_jit(
        pose0,
        half_dx=half_dx,
        half_dy=half_dy,
        half_dz=half_dz,
        half_nx=half_nx,
        half_ny=half_ny,
        half_nz=half_nz,
        half_dangle=half_d_euler,
        half_n_xrot=half_n_alpha,
        half_n_yrot=half_n_beta,
        half_n_zrot=half_n_gamma,
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
