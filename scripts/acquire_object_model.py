import argparse
import time

import b3d
import jax
import jax.numpy as jnp
from b3d import Mesh, Pose
from tqdm import tqdm

b3d.rr_init("acquire_object_model")

# python scripts/acquire_object_model.py assets/shared_data_bucket/input_data/lysol_static.r3d


# ssh sam-b3d-l4.us-west1-a.probcomp-caliban -L 5000:localhost:5000

state = {}


def acquire(input_path, output_path=None):
    global state

    start_time = time.time()

    if output_path is None:
        output_path = input_path + ".graphics_edits.mp4"

    data = b3d.io.load_r3d(input_path)
    _, _, fx, fy, cx, cy, near, far = data["camera_intrinsics_depth"]
    image_height, image_width = data["depth"].shape[1:3]
    num_scenes = data["depth"].shape[0]
    indices = jnp.arange(0, num_scenes, 10)
    camera_poses = data["camera_pose"][indices]
    depths = data["depth"][indices]
    rgbs = data["rgb"][indices]

    xyz = b3d.xyz_from_depth_vectorized(depths, fx, fy, cx, cy)
    xyz_world_frame = camera_poses[:, None, None].apply(xyz)

    b3d.reload(b3d.utils)
    # Resize rgbs to be same size as depth.
    rgbs_resized = b3d.utils.clip_rgb(
        b3d.utils.resize_image_linear_vmap(rgbs / 255.0, image_height, image_width)
    )

    # Foreground segmentation
    masks_concat = jnp.stack(
        [b3d.carvekit_get_foreground_mask(r) for r in rgbs_resized], axis=0
    )

    grid_center = jnp.median(camera_poses[0].apply(xyz[0][masks_concat[0]]), axis=0)
    W = 0.4
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
            0.004,
        )
    )
    i = len(occ_free_occl_)
    occ_free_occl, colors_per_voxel = occ_free_occl_[:i], colors_per_voxel_[:i]
    total_occ = (occ_free_occl == 1.0).sum(0)
    total_free = (occ_free_occl == -1.0).sum(0)
    ratio = total_occ / (total_occ + total_free) * ((total_occ + total_free) > 1)

    grid_colors = colors_per_voxel.sum(0) / (total_occ[..., None])
    model_mask = ratio > 0.1

    grid_points = grid[model_mask]
    colors = grid_colors[model_mask]

    _object_mesh = Mesh.voxel_mesh_from_xyz_colors_dimensions(
        grid_points,
        jnp.ones((grid_points.shape[0], 3)) * 0.002 * 2.0,
        colors,
    )
    object_pose = Pose.from_translation(jnp.median(_object_mesh.vertices, axis=0))
    object_mesh = _object_mesh.transform(object_pose.inv())
    object_mesh.rr_visualize("object_mesh")

    # Save the mesh
    # mesh_filename = input_path + ".mesh.obj"
    # print(f"Saving obj file to {mesh_filename}")
    # object_mesh.save(mesh_filename)

    # Load the mesh

    renderer = state.get("renderer")
    render_images = state.get("render_images")
    if renderer is None or renderer.width != image_width or renderer.fx != fx:
        print("Creating new renderer and render functions")
        renderer = b3d.RendererOriginal(
            image_width, image_height, fx, fy, cx, cy, near, far
        )

        def render_image(t):
            return renderer.render_rgbd_from_mesh(
                scene_mesh.transform(data["camera_pose"][t].inv())
            )

        render_images = jax.jit(jax.vmap(render_image))
        state["renderer"] = renderer
        state["render_images"] = render_images
    else:
        print("Reusing renderer and render functions")

    rerendered_rgbds = renderer.render_rgbd_many(
        (camera_poses[:, None].inv() @ object_pose).apply(object_mesh.vertices),
        object_mesh.faces,
        jnp.tile(object_mesh.vertex_attributes, (len(camera_poses), 1, 1)),
    )

    # t = 0
    # mask_now = (rerendered_rgbds[t,:,:,3] == 0)
    # xyz_world_frame_t = xyz_world_frame[t,:,:]
    # b3d.rr_log_cloud("cloud", xyz_world_frame_t[mask_now,:])
    # b3d.rr_log_cloud("cloud2", xyz_world_frame_t[~mask_now])
    # b3d.rr_log_depth("mask", mask_now * 1.0)
    # b3d.rr_log_depth("mask/depth", xyz[t,...,2])

    # t = 50
    # camera_pose = camera_poses[t]
    # mask_now = (rerendered_rgbds[t,::2,::2,3] == 0)
    # xyz_1 = xyz_world_frame[t,::2,::2]
    # b3d.rr_log_cloud("cloud3", xyz_1)
    # b3d.rr_log_cloud("cloud2", xyz_1)

    # pixels = b3d.xyz_to_pixel_coordinates(velo[:, :3], fx, fy, cx, cy).astype(jnp.int32)
    # depth_image = jnp.zeros((height, width))

    # xyz_world_frame

    # print(xyz_0.shape)

    # for t in range(len(rerendered_rgbds)):
    #     b3d.rr_set_time(t)
    #     mask_now = (rerendered_rgbds[t,...,3] == 0)
    #     xyz_0 = xyz_world_frame[t][mask_now]
    #     b3d.rr_log_cloud("cloud", xyz_0)
    #     b3d.rr_log_depth("mask", mask_now * 1.0)
    #     b3d.rr_log_depth("mask/depth", xyz_world_frame[t][..., 2])

    sub_indices = jnp.array([0, 5, len(camera_poses) - 15, len(camera_poses) - 5])
    mask = rerendered_rgbds[sub_indices, ..., 3] == 0.0

    background_xyzs = xyz_world_frame[sub_indices][mask]
    colors = rgbs_resized[sub_indices][mask, :]
    distances_from_camera = xyz[sub_indices][..., 2][mask][..., None] / fx

    clustering = b3d.segment_point_cloud(
        background_xyzs, threshold=0.03, min_points_in_cluster=10
    )
    m = clustering != -1
    background_xyzs_ = background_xyzs[m]
    distances_from_camera_ = distances_from_camera[m]
    colors_ = colors[m]

    background_mesh = Mesh.voxel_mesh_from_xyz_colors_dimensions(
        background_xyzs_,
        jnp.ones((background_xyzs_.shape[0], 3)) * distances_from_camera_,
        colors_,
    )
    # background_mesh.rr_visualize("background_mesh")

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

    ss = jnp.concatenate(
        [
            jnp.arange(0, len(data["camera_pose"]), 30),
            jnp.array([len(data["camera_pose"]) - 1]),
        ]
    )
    ss = jnp.vstack([ss[:-1], ss[1:]]).T
    images = jnp.concatenate([render_images(jnp.arange(s[0], s[1])) for s in ss])

    viz_images = []
    for t in tqdm(range(len(images))):
        viz_images.append(b3d.viz_rgb(images[t]))

    b3d.make_video_from_pil_images(viz_images, output_path, fps=30.0)
    print(f"Saved video to {output_path}")

    # Code to be timed

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
    return output_path


def main():
    parser = argparse.ArgumentParser("acquire_object_mode")
    parser.add_argument("input", help="r3d file", type=str)
    args = parser.parse_args()
    filename = args.input
    _ = acquire(filename)
    run2 = acquire(filename)
    return run2


if __name__ == "__main__":
    main()