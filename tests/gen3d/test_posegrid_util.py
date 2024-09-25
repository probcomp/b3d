import jax
import jax.numpy as jnp
from b3d import Pose
from b3d.pose.grid import pose_grid, viz_from_grid


def test_pose_gridding():
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

    # 3. Test that a 1-pose grid is just the starting point
    pose_from_grid = pose_grid_jit(
        pose0,
        half_dx=half_dx,
        half_dy=half_dy,
        half_dz=half_dz,
        half_nx=0,
        half_ny=0,
        half_nz=0,
        half_dangle=half_d_euler,
        half_n_xrot=0,
        half_n_yrot=0,
        half_n_zrot=0,
    )[0]
    assert jnp.allclose(pose_from_grid.position, pose0.position) and jnp.allclose(
        pose_from_grid.quaternion, pose0.quaternion
    ), "Single-pose grid not equal to original pose"

    # ##################################
    # # Test time
    # ##################################
    import time

    print("Testing jitted runtime...")
    start = time.time()
    _ = pose_grid_jit(
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
