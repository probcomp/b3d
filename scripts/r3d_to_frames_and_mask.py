from pathlib import Path
from typing import Optional

import cv2
import fire
import imageio
import jax
import jax.numpy as jnp
import numpy as np
import trimesh
from r3d_to_video_input import load_r3d_video_input

import b3d


def get_masks(rgb_imgs: jax.Array) -> jax.Array:
    masks = [b3d.carvekit_get_foreground_mask(img) for img in rgb_imgs]
    return jnp.stack(masks, axis=0)


def fit_voxel_mesh_model(
    camera_poses: b3d.Pose,
    rgb_imgs: jax.Array,
    xyzs: jax.Array,
    masks: jax.Array,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> trimesh.Trimesh:
    # adapted from acquire_object_model.py
    grid_center = jnp.median(camera_poses[0].apply(xyzs[0][masks[0]]), axis=0)
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

    occ_free_occl_, colors_per_voxel_ = jax.jit(
        jax.vmap(
            b3d.voxel_occupied_occluded_free,
            in_axes=(0, 0, 0, None, None, None, None, None, None, None),
        )
    )(
        camera_poses,
        rgb_imgs,
        xyzs[..., 2] * masks + (1.0 - masks) * 5.0,
        grid,
        fx,
        fy,
        cx,
        cy,
        6.0,
        0.005,
    )
    i = len(occ_free_occl_)
    occ_free_occl, colors_per_voxel = occ_free_occl_[:i], colors_per_voxel_[:i]
    total_occ = (occ_free_occl == 1.0).sum(0)
    total_free = (occ_free_occl == -1.0).sum(0)
    ratio = total_occ / (total_occ + total_free) * ((total_occ + total_free) > 1)

    grid_colors = colors_per_voxel.sum(0) / (total_occ[..., None])
    model_mask = ratio > 0.2

    resolution = 0.0015
    vertices, faces, vertex_colors, face_colors = (
        b3d.make_mesh_from_point_cloud_and_resolution(
            grid[model_mask],
            grid_colors[model_mask],
            resolution * jnp.ones_like(model_mask) * 2.0,
        )
    )
    vertices_centered = vertices - vertices.mean(0)

    return trimesh.Trimesh(
        vertices=vertices_centered,
        faces=faces,
        vertex_colors=np.array(vertex_colors * 255).astype(np.uint8),
        face_colors=np.array(face_colors * 255).astype(np.uint8),
    )


def main(r3d_path: str, out_dir: Optional[str] = None, create_obj: bool = True) -> None:
    """A utility script to decode R3D video into frames and extract the object
    mask for the first frame.

    Args:
        r3d_path (str): The path to the R3D video file.
        out_dir (str, optional): The output directory. By default, the frames
        will be save to a directory named f"{r3d_path}_frames" at the same
        directory as the input video.
        create_obj (bool, optional): Whether to fit a voxel-based mesh model for
        the object in the scene. Defaults to True. Note that this requires that
        the object to be visible from the first frame
    """
    if out_dir is None:
        out_dir = f"{r3d_path}_frames"
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=False)

    if r3d_path.lower().endswith(".r3d"):
        video_input = load_r3d_video_input(r3d_path)
    elif r3d_path.lower().endswith(".npz"):
        video_input = b3d.io.VideoInput.load(r3d_path)

    # resize RGB images to match the shape of the depth images
    rgbs_resized = jnp.clip(
        jax.vmap(jax.image.resize, in_axes=(0, None, None))(
            video_input.rgb / 255.0,
            (video_input.xyz.shape[1], video_input.xyz.shape[2], 3),
            "linear",
        ),
        0.0,
        1.0,
    )
    # object masks
    masks = get_masks(rgbs_resized)

    # Camera intrinsics
    K = np.eye(3)
    fx, fy, cx, cy = np.array(video_input.camera_intrinsics_depth)[2:6]
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy

    if create_obj:
        camera_poses = b3d.Pose(
            video_input.camera_positions, video_input.camera_quaternions
        )
        mesh = fit_voxel_mesh_model(
            camera_poses, rgbs_resized, video_input.xyz, masks, fx, fy, cx, cy
        )

    #####################################
    # Start dumping files
    #####################################

    # compute the number of leading zeros needed to name the frames
    num_frames = video_input.rgb.shape[0]
    num_digits = int(np.ceil(np.log10(num_frames)))

    np.savetxt(out_dir / "cam_K.txt", K, delimiter=" ")

    # RGB frames
    rgb_dir = out_dir / "rgb"
    rgb_dir.mkdir()

    # Convert RGB to OpenCV's BGR format
    rgbs_resized = np.array(rgbs_resized * 255).astype(np.uint8)[..., ::-1]
    for i, frame in enumerate(rgbs_resized):
        cv2.imwrite(
            str(rgb_dir / f"{i:0{num_digits}}.png"),
            frame,
        )

    # Depth frames
    depth_dir = out_dir / "depth"
    depth_dir.mkdir()
    # Convert depth values to millimeters and keep one channel only
    depths = np.array(video_input.xyz[..., 2] * 1000).clip(0).astype(np.uint16)
    for i, frame in enumerate(depths):
        cv2.imwrite(
            str(depth_dir / f"{i:0{num_digits}}.png"),
            frame,
        )

    # Masks
    # Note: FoundationPose only needs the mask for first frame, but here we
    # are saving more than the necessary frames for logging purpose.
    mask_dir = out_dir / "masks"
    mask_dir.mkdir()
    for i, mask in enumerate(masks):
        cv2.imwrite(
            str(mask_dir / f"{i:0{num_digits}}.png"),
            np.array(mask * 255).astype(np.uint8),
        )

    if create_obj:
        # Save the mesh
        mesh_dir = out_dir / "mesh"
        mesh_dir.mkdir()
        mesh.export(str(mesh_dir / "mesh.obj"))


if __name__ == "__main__":
    fire.Fire(main)
