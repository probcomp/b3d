import jax.numpy as jnp
import numpy as np
import jax
import os
import shutil
from unity_data import UnityData
from b3d.io.segmented_video_input import SegmentedVideoInput
from b3d.utils import downsize_images
from generate_visualization import create_segmentation_data_gif
from path_utils import get_assets_path
from unity_to_python import (
    convert_unity_to_cv2_camera_pos_and_quat, 
    convert_unity_to_cv2_world_pos_and_quat, 
    convert_rgb_float_to_uint, 
    downsize_images_1D
)
from b3d.camera import unproject_depth

def get_camera_position_and_quaternion(unity_camera_position: jnp.ndarray, unity_camera_quaternion: jnp.ndarray):
    """Convert camera positions and quaternions from Unity coordinates system to OpenCV convention."""
    vmap_frames_to_cv2_pos_quat = jax.vmap(convert_unity_to_cv2_camera_pos_and_quat, in_axes=(0, 0, None))
    camera_position, camera_quaternion = vmap_frames_to_cv2_pos_quat(unity_camera_position, unity_camera_quaternion, False)
    return camera_position, camera_quaternion

def get_object_positions_and_quaternions(unity_object_positions: jnp.ndarray, unity_object_quaternions: jnp.ndarray):
    """Convert Unity world object positions to OpenCV convention."""
    vmap_object_unity_to_cv2_pos_quat = jax.vmap(
        jax.vmap(convert_unity_to_cv2_world_pos_and_quat, in_axes=(0, 0, None)), 
        in_axes=(0, 0, None)
        )

    object_positions, object_quaternions = vmap_object_unity_to_cv2_pos_quat(unity_object_positions,
                                                                    unity_object_quaternions, 
                                                                    True)
    
    return object_positions, object_quaternions

def get_xyz(unity_depth: jnp.ndarray, camera_intrinsics: jnp.ndarray):
    "Extract xyz values from camera intrinsics and depth"
    vmap_unproject_depth = jax.vmap(unproject_depth, in_axes=(0,None))
    xyz = vmap_unproject_depth(unity_depth, camera_intrinsics)
    return xyz

def get_rgb_float_to_uint(float_rgb: jnp.ndarray):
    vmap_convert_rgb_float_to_uint = jax.vmap(convert_rgb_float_to_uint, in_axes=(0,))
    rgb = vmap_convert_rgb_float_to_uint(float_rgb)
    return rgb

def convert_unity_to_segmented_video_input(unity_data: UnityData) -> SegmentedVideoInput:
    """Convert Unity data to Segmented Video Input."""
    rgb = get_rgb_float_to_uint(unity_data.rgb)
    xyz = get_xyz(unity_data.depth, unity_data.camera_intrinsics)
    camera_position, camera_quaternion = get_camera_position_and_quaternion(unity_data.camera_position, unity_data.camera_quaternion)
    object_positions, object_quaternions = get_object_positions_and_quaternions(unity_data.object_positions, unity_data.object_quaternions)

    return SegmentedVideoInput(
        rgb=rgb,
        xyz=xyz,
        segmentation=unity_data.segmentation,
        camera_positions=camera_position,
        camera_quaternions=camera_quaternion,
        camera_intrinsics_rgb=unity_data.camera_intrinsics,
        camera_intrinsics_depth=unity_data.camera_intrinsics,
        object_positions=object_positions,
        object_quaternions=object_quaternions,
        object_catalog_ids=unity_data.object_catalog_ids,
        # fps=unity_data.fps
    )

def convert_from_zip(zip_path: str) -> SegmentedVideoInput:
    unity_data = UnityData.segmented_video_input_data_from_zip(zip_path)
    segmented_video_input_data = convert_unity_to_segmented_video_input(unity_data)
    return segmented_video_input_data

def save_segmented_video_input_data(data: SegmentedVideoInput, file_info: dict, create_gif: bool=False) -> None:
    folder_path = get_assets_path('s', file_info['scene_folder'], file_info['data_name'])
    file_name = f"{file_info['light_setting']}_{file_info['background_setting']}_{file_info['resolution']}.input.npz"
    filepath = str(folder_path / file_name)

    data.save(filepath)
    print(f"Saved {filepath}")

    if (create_gif):
        gifname = f"{file_info['data_name']}_{file_info['light_setting']}_{file_info['background_setting']}_segmented.gif"
        create_segmentation_data_gif(data, str(folder_path / gifname))

def downsize_video_input(data: SegmentedVideoInput, k: float) -> SegmentedVideoInput:
    # camera_intrinsics = data.camera_intrinsics_rgb.at[0].mul(1/k).at[1].mul(1/k)
    camera_intrinsics = np.array(data.camera_intrinsics_rgb).copy()
    camera_intrinsics[0] /= k
    camera_intrinsics[1] /= k

    depth_from_xyz = data.xyz[..., 2]

    rgb = downsize_images(data.rgb, k).astype(np.uint8)
    depth = downsize_images_1D(depth_from_xyz, k)
    segmentation = downsize_images_1D(data.segmentation, k)
    
    xyz = get_xyz(depth, data.camera_intrinsics_rgb)

    return SegmentedVideoInput(
        rgb=rgb,
        xyz=xyz,
        segmentation=segmentation,
        camera_positions=data.camera_positions,
        camera_quaternions=data.camera_quaternions,
        camera_intrinsics_rgb=camera_intrinsics,
        camera_intrinsics_depth=camera_intrinsics,
        object_positions=data.object_positions,
        object_quaternions=data.object_quaternions,
        object_catalog_ids=data.object_catalog_ids,
        # fps=data.fps
    )

def save_downscaled_video_input(data: SegmentedVideoInput, target_res: int, file_info: dict, create_gif: bool=False) -> None:
    k = int(data.camera_intrinsics_rgb[0] / target_res)
    small_version = downsize_video_input(data, k)

    file_info['resolution'] = f"{target_res}p"
    save_segmented_video_input_data(small_version, file_info, create_gif)

def process(zip_path: str, moveFile: bool=True) -> None:
    """Process a ZIP file and save the segmented video input data."""
    unity_data = UnityData.segmented_video_input_data_from_zip(zip_path)
    segmented_video_data = convert_unity_to_segmented_video_input(unity_data)

    # Save segmented_video_data
    save_segmented_video_input_data(segmented_video_data, unity_data.file_info, create_gif=True)

    # Save a 200p version
    save_downscaled_video_input(segmented_video_data, 200, unity_data.file_info, create_gif=False)

    # move zip_path file into FBData/processed folder
    if (moveFile):
        processed_folder = 'Processed'
        os.makedirs(processed_folder, exist_ok=True)
        shutil.move(zip_path, processed_folder)

    print(f"{zip_path} has been processed.")

