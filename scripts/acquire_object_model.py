import jax.numpy as jnp
import os
import b3d
import rerun as rr
import jax
import argparse

rr.init("acquire_object_model")
rr.connect("127.0.0.1:8812")


parser = argparse.ArgumentParser("acquire_object_mode")
parser.add_argument("input", help="Video Input File", type=str)
args = parser.parse_args()

filename = args.input
# filename = "assets/shared_data_bucket/input_data/lysol_static.r3d.video_input.npz"
video_input = b3d.io.VideoInput.load(filename)

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

indices = jnp.arange(0, video_input.xyz.shape[0], 10)




camera_poses = b3d.Pose(
    video_input.camera_positions,
    video_input.camera_quaternions
)[indices]

xyz = video_input.xyz[indices]
xyz_world_frame = camera_poses[:,None, None].apply(xyz)

# Resize rgbs to be same size as depth.
rgbs_resized = jnp.clip(
    jax.vmap(jax.image.resize, in_axes=(0, None, None))(
        video_input.rgb[indices] / 255.0, (video_input.xyz.shape[1], video_input.xyz.shape[2], 3), "linear"
    ),
    0.0,
    1.0,
)


masks = [b3d.carvekit_get_foreground_mask(r) for r in rgbs_resized]
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

occ_free_occl_, colors_per_voxel_ = b3d.voxel_occupied_occluded_free_parallel_camera_depth(
    camera_poses, rgbs_resized, xyz[...,2] * masks_concat + (1.0 - masks_concat) * 5.0, grid, fx,fy,cx,cy, 6.0, 0.005
)
i = len(occ_free_occl_)
occ_free_occl, colors_per_voxel = occ_free_occl_[:i], colors_per_voxel_[:i]
total_occ = (occ_free_occl == 1.0).sum(0)
total_free = (occ_free_occl == -1.0).sum(0)
ratio = total_occ / (total_occ + total_free) * ((total_occ + total_free) > 1)

grid_colors = colors_per_voxel.sum(0)/ (total_occ[...,None])
model_mask = ratio > 0.2

resolution = 0.0015

grid_points = grid[model_mask]
colors = grid_colors[model_mask]


meshes = b3d.mesh.transform_mesh(
    jax.vmap(b3d.mesh.Mesh.cube_mesh)(jnp.ones((grid_points.shape[0],3)) * resolution * 2.0, colors),
    b3d.Pose.from_translation(grid_points)[:,None]
)
full_mesh = b3d.mesh.Mesh.squeeze_mesh(meshes)
full_mesh.rr_visualize("mesh")


# Save the mesh
jnp.savez(
    filename + ".mesh.npz",
    full_mesh.vertices,
    full_mesh.faces,
    full_mesh.vertex_attributes
)
# full_mesh = b3d.Mesh(*jnp.load(filename + ".mesh.npz").values())
