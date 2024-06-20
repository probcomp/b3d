import jax.numpy as jnp
import os
import b3d
import rerun as rr
import jax

rr.init("demo")
rr.connect("127.0.0.1:8812")


path = os.path.join(
    b3d.get_root_path(),
    "assets/shared_data_bucket/input_data/lysol_static.r3d.video_input.npz",
)
video_input = b3d.io.VideoInput.load(path)

import numpy as np
image_width, image_height, fx, fy, cx, cy, near, far = np.array(
    video_input.camera_intrinsics_depth
)
image_width, image_height = int(image_width), int(image_height)
fx, fy, cx, cy, near, far = (
    float(fx),
    float(fy),
    float(cx),
    float(cy),
    float(near),
    float(far),
)


camera_poses = b3d.Pose(
    video_input.camera_positions,
    video_input.camera_quaternions
)
xyz = video_input.xyz
xyz_world_frame = camera_poses[:,None, None].apply(xyz)
xyz_world_frame_flat = xyz_world_frame.reshape(xyz_world_frame.shape[0], -1, 3)

# Resize rgbs to be same size as depth.
rgbs_resized = jnp.clip(
    jax.vmap(jax.image.resize, in_axes=(0, None, None))(
        video_input.rgb / 255.0, (video_input.xyz.shape[1], video_input.xyz.shape[2], 3), "linear"
    ),
    0.0,
    1.0,
)


for t in range(0, xyz_world_frame_flat.shape[0], 10):
    rr.set_time_sequence("frame", t)
    rr.log(
        "lysol",
        rr.Points3D(xyz_world_frame_flat[t], colors=rgbs_resized[t].reshape(-1,3)),
    )


HIINTERFACE = None
def carvekit_get_foreground_mask(image):
    global HIINTERFACE
    if HIINTERFACE is None:
        import torch
        from carvekit.api.high import HiInterface

        HIINTERFACE = HiInterface(
            object_type="object",  # Can be "object" or "hairs-like".
            batch_size_seg=5,
            batch_size_matting=1,
            device="cuda" if torch.cuda.is_available() else "cpu",
            seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
            matting_mask_size=2048,
            trimap_prob_threshold=220,  # 231,
            trimap_dilation=15,
            trimap_erosion_iters=20,
            fp16=False,
        )
    imgs = HIINTERFACE([b3d.get_rgb_pil_image(image)])
    mask = jnp.array(imgs[0])[..., -1] > 0.5
    return mask

def discretize(data, resolution):
    """
    Discretizes a point cloud.
    """
    return jnp.round(data / resolution) * resolution


def voxelize(data, resolution):
    """
        Voxelize a point cloud.
    Args:
        data: (N,3) point cloud
        resolution: (float) resolution of the voxel grid
    Returns:
        data: (M,3) voxelized point cloud
    """
    data = discretize(data, resolution)
    data, indices, occurences = jnp.unique(data, axis=0, return_index=True, return_counts=True)
    return data, indices, occurences


def voxel_occupied_occluded_free(camera_pose, rgb_image, depth_image, grid, fx,fy,cx,cy, far,tolerance):
    grid_in_cam_frame = camera_pose.inv().apply(grid)
    height,width = depth_image.shape[:2]
    pixels = b3d.xyz_to_pixel_coordinates(grid_in_cam_frame, fx,fy,cx,cy).astype(jnp.int32)
    valid_pixels = (
        (0 <= pixels[:, 0])
        * (0 <= pixels[:, 1])
        * (pixels[:, 0] < height)
        * (pixels[:, 1] < width)
    )
    real_depth_vals = depth_image[pixels[:, 0], pixels[:, 1]] * valid_pixels + (
        1 - valid_pixels
    ) * (far + 1.0)


    projected_depth_vals = grid_in_cam_frame[:, 2]
    occupied = jnp.abs(real_depth_vals - projected_depth_vals) < tolerance
    real_rgb_values = rgb_image[pixels[:, 0], pixels[:, 1]] * occupied[...,None]
    occluded = real_depth_vals < projected_depth_vals
    occluded = occluded * (1.0 - occupied)
    _free = (1.0 - occluded) * (1.0 - occupied)
    return 1.0 * occupied  -  1.0 * _free, real_rgb_values

voxel_occupied_occluded_free_jit = jax.jit(voxel_occupied_occluded_free)
voxel_occupied_occluded_free_parallel_camera = jax.jit(
    jax.vmap(voxel_occupied_occluded_free, in_axes=(0, None, None, None, None, None, None, None, None, None))
)
voxel_occupied_occluded_free_parallel_camera_depth = jax.jit(
    jax.vmap(voxel_occupied_occluded_free, in_axes=(0, 0, 0, None, None, None, None, None, None, None))
)

masks = [carvekit_get_foreground_mask(r) for r in rgbs_resized]
masks_concat = jnp.stack(masks, axis=0)

grid_center = jnp.median(camera_poses[0].apply(video_input.xyz[0][masks[0]]),axis=0)
W = 0.3
D = 100
grid = jnp.stack(
    jnp.meshgrid(
        jnp.linspace(grid_center[0]-W/2, grid_center[0]+W/2, D),
        jnp.linspace(grid_center[1]-W/2, grid_center[1]+W/2, D),
        jnp.linspace(grid_center[2]-W/2, grid_center[2]+W/2, D),
    ),
    axis=-1,
).reshape(-1, 3)

rr.log(
    "grid",
    rr.Points3D(grid)
)

occ_free_occl, colors_per_voxel = voxel_occupied_occluded_free_parallel_camera_depth(
    camera_poses, rgbs_resized, video_input.xyz[...,2] * masks_concat + (1.0 - masks_concat) * 5.0, grid, fx,fy,cx,cy, 6.0, 0.005
)
total_occ = (occ_free_occl == 1.0).sum(0)
total_free = (occ_free_occl == -1.0).sum(0)
ratio = total_occ / (total_occ + total_free) * ((total_occ + total_free) > 1)

grid_colors = colors_per_voxel.sum(0)/ (total_occ[...,None])
model_mask = ratio > 0.2
rr.log(
    "grid",
    rr.Points3D(grid[model_mask], colors=grid_colors[model_mask])
)

resolution = 0.0015
vertices, faces, vertex_colors, face_colors = b3d.make_mesh_from_point_cloud_and_resolution(
   grid[model_mask], grid_colors[model_mask], resolution * jnp.ones_like(model_mask) * 2.0
)
vertices_centered = vertices - vertices.mean(0)
rr.log("mesh", rr.Mesh3D(
    vertex_positions=vertices_centered,
    indices=faces,
    vertex_colors=vertex_colors,
))


# Save the mesh
jnp.savez(
    path + ".mesh.npz",
    vertices=vertices_centered,
    faces=faces,
    vertex_colors=vertex_colors,
)

from b3d.renderer_original import RendererOriginal
renderer = RendererOriginal(image_width, image_height, fx, fy, cx, cy, near, far)

rasterization_output = renderer.rasterize(b3d.Pose.from_translation(jnp.array([0.0, 0.0, 0.5])).apply(vertices_centered)[None,...], faces)
rr.log("rerender", rr.Image(vertex_colors[faces[:,0][rasterization_output[0,...,-1].astype(jnp.int32)]]))