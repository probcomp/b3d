import argparse

import b3d
import jax
import jax.numpy as jnp
from b3d import Pose
from tqdm import tqdm

b3d.rr_init("acquire_object_model")

# python scripts/acquire_object_model.py assets/shared_data_bucket/input_data/lysol_static.r3d


def acquire(input_path, output_path=None):
    if output_path is None:
        output_path = input_path + ".graphics_edits.mp4"

    data = b3d.io.load_r3d(input_path)

    _, _, fx, fy, cx, cy, near, far = data["camera_intrinsics_depth"]
    image_height, image_width = data["depth"].shape[1:3]
    num_scenes = data["depth"].shape[0]

    indices = jnp.arange(0, num_scenes, 10)

    camera_poses_full = data["camera_pose"]
    camera_poses = camera_poses_full[indices]

    xyz = b3d.xyz_from_depth_vectorized(data["depth"][indices], fx, fy, cx, cy)
    xyz_world_frame = camera_poses[:, None, None].apply(xyz)

    # for i in range(len(xyz_world_frame)):
    #     b3d.rr_set_time(i)
    #     b3d.utils.rr_log_cloud("xyz", xyz_world_frame[i])

    # Resize rgbs to be same size as depth.
    rgbs = data["rgb"]
    rgbs_resized = jnp.clip(
        jax.vmap(jax.image.resize, in_axes=(0, None, None))(
            rgbs[indices] / 255.0,
            (image_height, image_width, 3),
            "linear",
        ),
        0.0,
        1.0,
    )

    masks = [b3d.carvekit_get_foreground_mask(r) for r in rgbs_resized]
    masks_concat = jnp.stack(masks, axis=0)

    grid_center = jnp.median(camera_poses[0].apply(xyz[0][masks[0]]), axis=0)
    W = 0.3
    D = 100
    grid = jnp.stack(
        jnp.meshgrid(
            jnp.linspace(grid_center[0] - W / 2, grid_center[0] + W / 2, D),
            jnp.linspace(grid_center[1] - W / 2, grid_center[1] + W / 2, D),
            jnp.linspace(grid_center[2] - W / 2, grid_center[2] + W / 2, D),
        ),
        axis=-1,
    ).reshape(-1, 3)

    occ_free_occl_, colors_per_voxel_ = (
        b3d.voxel_occupied_occluded_free_parallel_camera_depth(
            camera_poses,
            rgbs_resized,
            xyz[..., 2] * masks_concat + (1.0 - masks_concat) * 5.0,
            grid,
            fx,
            fy,
            cx,
            cy,
            6.0,
            0.005,
        )
    )
    i = len(occ_free_occl_)
    occ_free_occl, colors_per_voxel = occ_free_occl_[:i], colors_per_voxel_[:i]
    total_occ = (occ_free_occl == 1.0).sum(0)
    total_free = (occ_free_occl == -1.0).sum(0)
    ratio = total_occ / (total_occ + total_free) * ((total_occ + total_free) > 1)

    grid_colors = colors_per_voxel.sum(0) / (total_occ[..., None])
    model_mask = ratio > 0.2

    resolution = 0.0015

    grid_points = grid[model_mask]
    colors = grid_colors[model_mask]

    meshes = b3d.mesh.transform_mesh(
        jax.vmap(b3d.mesh.Mesh.cube_mesh)(
            jnp.ones((grid_points.shape[0], 3)) * resolution * 2.0, colors
        ),
        b3d.Pose.from_translation(grid_points)[:, None],
    )
    _object_mesh = b3d.mesh.Mesh.squeeze_mesh(meshes)

    object_pose = Pose.from_translation(jnp.median(_object_mesh.vertices, axis=0))
    object_mesh = _object_mesh.transform(object_pose.inv())
    object_mesh.rr_visualize("mesh")

    mesh_filename = input_path + ".mesh.obj"
    # Save the mesh
    print(f"Saving obj file to {mesh_filename}")
    object_mesh.save(mesh_filename)

    renderer = b3d.RendererOriginal(
        image_width, image_height, fx, fy, cx, cy, near, far
    )
    rgbds = renderer.render_rgbd_many(
        (camera_poses[:, None].inv() @ object_pose).apply(object_mesh.vertices),
        object_mesh.faces,
        jnp.tile(object_mesh.vertex_attributes, (len(camera_poses), 1, 1)),
    )

    sub_indices = jnp.array([0, 5, len(camera_poses) - 15, len(camera_poses) - 5])
    mask = rgbds[sub_indices, ..., 3] == 0.0

    background_xyzs = xyz_world_frame[sub_indices][mask]
    colors = rgbs_resized[sub_indices][mask, :]
    distances_from_camera = xyz[sub_indices][..., 2][mask][..., None] / fx

    # subset = jax.random.choice(jax.random.PRNGKey(0), jnp.arange(background_xyzs.shape[0]), shape=(background_xyzs.shape[0]//3,), replace=False)

    # background_xyzs = background_xyzs[subset]
    # colors = colors[subset]
    # distances_from_camera = distances_from_camera[subset]

    meshes = b3d.mesh.transform_mesh(
        jax.vmap(b3d.mesh.Mesh.cube_mesh)(
            jnp.ones((background_xyzs.shape[0], 3)) * distances_from_camera, colors
        ),
        b3d.Pose.from_translation(background_xyzs)[:, None],
    )
    background_mesh = b3d.mesh.Mesh.squeeze_mesh(meshes)
    background_mesh.rr_visualize("background_mesh")

    object_poses = [
        object_pose,
        Pose.identity(),
        object_pose @ Pose.from_translation(jnp.array([-0.1, 0.0, 0.1])),
        object_pose @ Pose.from_translation(jnp.array([-0.1, 0.0, -0.1])),
    ]

    scene_mesh = b3d.mesh.transform_and_merge_meshes(
        [object_mesh, background_mesh, object_mesh, object_mesh],
        object_poses,
    )

    viz_images = []
    for t in tqdm(range(len(camera_poses_full))):
        b3d.utils.rr_set_time(t)
        rgbd = renderer.render_rgbd_from_mesh(
            scene_mesh.transform(camera_poses_full[t].inv())
        )
        viz_images.append(b3d.viz_rgb(rgbd))

    output_path = input_path + ".graphics_edits.mp4"
    b3d.make_video_from_pil_images(viz_images, output_path, fps=30.0)
    print(f"Saved video to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser("acquire_object_mode")
    parser.add_argument("input", help="r3d file", type=str)
    args = parser.parse_args()
    filename = args.input
    return acquire(filename)


if __name__ == "__main__":
    main()
