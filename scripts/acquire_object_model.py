import jax.numpy as jnp
import os
import b3d
import rerun as rr
import jax
import argparse

rr.init("acquire_object_model")
rr.connect("127.0.0.1:8812")


parser = argparse.ArgumentParser("r3d_to_video_input")
parser.add_argument("input", help=".r3d File", type=str)
args = parser.parse_args()

filename = args.input
video_input = b3d.io.VideoInput.load(filename)

parser = argparse.ArgumentParser("r3d_to_video_input")
parser.add_argument("input", help=".r3d File", type=str)
args = parser.parse_args()


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
    rr.log("rgb", rr.Image(rgbs_resized[t]))
    rr.log(
        "lysol",
        rr.Points3D(xyz_world_frame_flat[t], colors=rgbs_resized[t].reshape(-1,3)),
    )

voxel_occupied_occluded_free_jit = jax.jit(b3d.voxel_occupied_occluded_free)
voxel_occupied_occluded_free_parallel_camera = jax.jit(
    jax.vmap(b3d.voxel_occupied_occluded_free, in_axes=(0, None, None, None, None, None, None, None, None, None))
)
voxel_occupied_occluded_free_parallel_camera_depth = jax.jit(
    jax.vmap(b3d.voxel_occupied_occluded_free, in_axes=(0, 0, 0, None, None, None, None, None, None, None))
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

rr.log(
    "grid",
    rr.Points3D(grid)
)

occ_free_occl_, colors_per_voxel_ = voxel_occupied_occluded_free_parallel_camera_depth(
    camera_poses, rgbs_resized, video_input.xyz[...,2] * masks_concat + (1.0 - masks_concat) * 5.0, grid, fx,fy,cx,cy, 6.0, 0.005
)
i = len(occ_free_occl_)
occ_free_occl, colors_per_voxel = occ_free_occl_[:i], colors_per_voxel_[:i]
total_occ = (occ_free_occl == 1.0).sum(0)
total_free = (occ_free_occl == -1.0).sum(0)
ratio = total_occ / (total_occ + total_free) * ((total_occ + total_free) > 1)

grid_colors = colors_per_voxel.sum(0)/ (total_occ[...,None])
model_mask = ratio > 0.2
rr.set_time_sequence("frame", i)

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

# meshes = []
# for i in range(10, len(rgbs_resized), 10):
#     occ_free_occl, colors_per_voxel = occ_free_occl_[:i], colors_per_voxel_[:i]
#     total_occ = (occ_free_occl == 1.0).sum(0)
#     total_free = (occ_free_occl == -1.0).sum(0)
#     ratio = total_occ / (total_occ + total_free) * ((total_occ + total_free) > 1)

#     grid_colors = colors_per_voxel.sum(0)/ (total_occ[...,None])
#     model_mask = ratio > 0.2
#     rr.set_time_sequence("frame", i)

#     resolution = 0.0015
#     vertices, faces, vertex_colors, face_colors = b3d.make_mesh_from_point_cloud_and_resolution(
#     grid[model_mask], grid_colors[model_mask], resolution * jnp.ones_like(model_mask) * 2.0
#     )
#     vertices_centered = vertices - vertices.mean(0)
#     rr.log("mesh", rr.Mesh3D(
#         vertex_positions=vertices_centered,
#         indices=faces,
#         vertex_colors=vertex_colors,
#     ))
#     meshes.append((i, vertices_centered, faces, vertex_colors))

# base_pose = b3d.Pose.from_translation(jnp.array([0.0, 0.0, 0.3]))
# poses = base_pose @ b3d.Pose.stack_poses([
#     b3d.Pose.from_quat(b3d.Rot.from_rotvec(jnp.array([0.0, angle + 0.2, 0.0])).as_quat())
#     for angle in jnp.linspace(0.0, 2*jnp.pi, 40)
# ]
# ) 


# from b3d.renderer_original import RendererOriginal
# renderer = RendererOriginal(image_width, image_height, fx, fy, cx, cy, near, far)



# rgbds = renderer.render_rgbd_many(poses[:,None].apply(jnp.tile(vertices_centered, (len(poses), 1, 1))), faces, jnp.tile(vertex_colors, (len(poses), 1, 1)))
# for (i,rgbd) in enumerate(rgbds):
#     rr.log(f"rerender/{i}", rr.Image(rgbd[...,:3]))





# # Save the mesh
# jnp.savez(
#     path + ".mesh.npz",
#     vertices=vertices_centered,
#     faces=faces,
#     vertex_colors=vertex_colors,
# )

