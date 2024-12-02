import jax.numpy as jnp

import b3d

b3d.rr_init("high_quality")

# python scripts/acquire_object_model.py assets/shared_data_bucket/input_data/lysol_static.r3d


# ssh sam-b3d-l4.us-west1-a.probcomp-caliban -L 5000:localhost:5000


input_path = (
    "/home/nishadgothoskar/b3d/assets/shared_data_bucket/input_data/409_bottle.r3d"
)


data = b3d.io.load_r3d(input_path)
image_height, image_width = data["depth"].shape[1:3]
num_scenes = data["depth"].shape[0]
indices = jnp.arange(0, num_scenes, 10)
camera_poses = data["camera_pose"][indices]

image_width, image_height, fx, fy, cx, cy, near, far = data["camera_intrinsics_rgb"]
image_height, image_width = int(image_height), int(image_width)

depths = b3d.utils.resize_image_linear_vmap(
    data["depth"][indices], image_height, image_width
)
rgbs = data["rgb"][indices]

xyz = b3d.xyz_from_depth_vectorized(depths, fx, fy, cx, cy)
xyz_world_frame = camera_poses[:, None, None].apply(xyz)


# Take point cloud at frame 0
point_cloud = xyz[0][::3, ::3, :].reshape(-1, 3)
point_cloud_colors = (rgbs[0] / 255.0)[::3, ::3, :].reshape(-1, 3)
assert point_cloud.shape == point_cloud_colors.shape

background_mesh = b3d.mesh.Mesh.mesh_from_xyz_colors_dimensions(
    point_cloud[:], point_cloud_colors[:], point_cloud[:, 2] / fx * 6.0
)


renderer = b3d.RendererOriginal(image_width, image_height, fx, fy, cx, cy, near, far)
rgbd = renderer.render_rgbd_from_mesh(background_mesh)
b3d.rr_log_rgbd("rgbd", rgbd)

background_mesh.save("high_resolution_background_mesh.obj")


start_t, end_t = 17, 50
acquisition_indices = jnp.array(range(start_t, end_t, 3))
print(len(acquisition_indices))


masks_concat = jnp.stack(
    [
        b3d.carvekit_get_foreground_mask(rgbs[index] / 255.0)
        for index in acquisition_indices
    ]
)


mask_0 = masks_concat[0]

b3d.rr_log_rgb("rgb", rgbs[acquisition_indices[0]])
b3d.rr_log_depth("mask", mask_0 * 1.0)

cloud_0 = xyz_world_frame[acquisition_indices[0]][mask_0]
b3d.rr_set_time(0)
b3d.rr_log_cloud("c", cloud_0)


grid_center = jnp.median(cloud_0, axis=0)

W = 0.25
D = 150
grid = (
    jnp.stack(
        jnp.meshgrid(
            jnp.linspace(-W / 2, +W / 2, D) * 0.8,
            jnp.linspace(-W / 2, +W / 2, D) * 1.2,
            jnp.linspace(-W / 2, +W / 2, D) * 0.8,
        ),
        axis=-1,
    ).reshape(-1, 3)
    + grid_center
)

occ_free_occl_, colors_per_voxel_ = (
    b3d.voxel_occupied_occluded_free_parallel_camera_depth(
        camera_poses[acquisition_indices],
        rgbs[acquisition_indices] / 255.0,
        xyz[acquisition_indices][..., 2] * masks_concat + (1.0 - masks_concat) * 5.0,
        grid,
        fx,
        fy,
        cx,
        cy,
        6.0,
        0.004,
    )
)

i = len(occ_free_occl_)
occ_free_occl, colors_per_voxel = occ_free_occl_[:i], colors_per_voxel_[:i]
total_occ = (occ_free_occl == 1.0).sum(0)
total_free = (occ_free_occl == -1.0).sum(0)
ratio = total_occ / (total_occ + total_free) * ((total_occ + total_free) > 1)

grid_colors = colors_per_voxel.sum(0) / (total_occ[..., None])
model_mask = ratio > 0.6
resolution = 0.0015

object_mesh = b3d.mesh.Mesh.mesh_from_xyz_colors_dimensions(
    grid[model_mask],
    grid_colors[model_mask],
    jnp.ones_like(grid)[model_mask] * resolution * 2.0,
)
object_mesh.rr_visualize("object_mesh")
object_mesh.save("high_resolution_object_mesh.obj")
