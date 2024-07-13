import jax.numpy as jnp
import jax
import os
import shutil
from unity_data import UnityData
from b3d.io.feature_track_data import FeatureTrackData
from b3d.utils import downsize_images
from generate_visualization import create_keypoints_gif
from path_utils import get_assets_path
from dataclasses import replace
from unity_to_python import (
    convert_unity_to_cv2_camera_pos_and_quat, 
    project_to_screen, 
    convert_unity_to_cv2_object_pos, 
    clamp_and_replace_with_nan, 
    is_point_in_front,
    downsize_2d_coordinates
)

def convert_unity_to_feature_track(unity_data: UnityData) -> FeatureTrackData:
    """Convert Unity data to FeatureTrackData."""
    vmap_frames_to_cv2_pos_quat = jax.vmap(convert_unity_to_cv2_camera_pos_and_quat, in_axes=(0, 0, None))
    vmap_project_to_screen = jax.vmap(
        jax.vmap(project_to_screen, in_axes=(0, None, None, None)),
        in_axes=(0, 0, 0, None)
    )
    vmap_keypoints_to_cv2_pos = jax.vmap(
        jax.vmap(convert_unity_to_cv2_object_pos, in_axes=(0,)), 
        in_axes=(0,)
    )
    vmap_is_point_in_frustum = jax.vmap(
        jax.vmap(is_point_in_front, in_axes=(0, None, None)),  
        in_axes=(0, 0, 0)
    )

    camera_position, camera_quaternion = vmap_frames_to_cv2_pos_quat(
        unity_data.camera_position, unity_data.camera_quaternion, False
    )

    latent_keypoint_positions = vmap_keypoints_to_cv2_pos(unity_data.latent_keypoint_positions)

    observed_keypoints_positions = vmap_project_to_screen(
        latent_keypoint_positions, camera_position, camera_quaternion, unity_data.camera_intrinsics
    )

    width, height = unity_data.camera_intrinsics[:2]
    observed_keypoints_positions = clamp_and_replace_with_nan(observed_keypoints_positions, width, height)

    in_frustum_mask = vmap_is_point_in_frustum(
        unity_data.latent_keypoint_positions, unity_data.camera_position, unity_data.camera_quaternion
    )
    
    observed_keypoints_positions = jnp.where(in_frustum_mask[..., None], observed_keypoints_positions, jnp.nan)

    depth_expanded = jnp.expand_dims(unity_data.depth, axis=-1)
    rgbd = jnp.concatenate((unity_data.rgb, depth_expanded), axis=-1)

    return FeatureTrackData(
        observed_keypoints_positions=observed_keypoints_positions,
        keypoint_visibility=unity_data.keypoint_visibility,
        camera_intrinsics=unity_data.camera_intrinsics,
        rgbd_images=rgbd, 
        fps=unity_data.fps,
        latent_keypoint_positions=latent_keypoint_positions,
        object_assignments=unity_data.object_assignments,
        camera_position=camera_position,
        camera_quaternion=camera_quaternion
    )

def convert_from_zip(zip_path: str) -> FeatureTrackData:
    unity_data = UnityData.from_zip(zip_path)
    feature_track_data = convert_unity_to_feature_track(unity_data)
    return feature_track_data

def downsize_feature_track(data: FeatureTrackData, k: float) -> FeatureTrackData:
    camera_intrinsics = data.camera_intrinsics.at[0].mul(1/k).at[1].mul(1/k)

    rgbd = downsize_images(data.rgbd, 4)
    observed_keypoints_positions = downsize_2d_coordinates(data.observed_keypoints_positions, k)

    return replace(
        data,
        camera_intrinsics=camera_intrinsics,
        rgbd_images=rgbd,
        observed_keypoints_positions=observed_keypoints_positions
    )

def save_downscaled_feature_track(data: FeatureTrackData, target_res: int, file_info: dict) -> None:
    folder_path = get_assets_path('f', file_info['scene_folder'], file_info['data_name'])
    file_name = f"{file_info['light_setting']}_{file_info['background_setting']}_{target_res}p.input.npz"
    filepath = str(folder_path / file_name)

    k = int(data.camera_intrinsics[0] / target_res)
    small_version = downsize_feature_track(data, k)
    small_version.save(filepath)

def process(zip_path: str, moveFile: bool=True) -> None:
    """Process a ZIP file and save the feature track data."""
    unity_data = UnityData.from_zip(zip_path)
    feature_track_data = convert_unity_to_feature_track(unity_data)
    
    file_info = unity_data.file_info
    folder_path = get_assets_path('f', file_info['scene_folder'], file_info['data_name'])
    file_name = f"{file_info['light_setting']}_{file_info['background_setting']}_{file_info['resolution']}.input.npz"
    filepath = str(folder_path / file_name)
    
    # Save feature_track_data and create a GIF
    feature_track_data.save(filepath)

    # Save a 200p version
    save_downscaled_feature_track(feature_track_data, 200, file_info)

    # Create a gif
    create_keypoints_gif(feature_track_data, filepath.replace('.npz', '.gif'))

    # move zip_path file into FBData/processed folder
    if (moveFile):
        processed_folder = 'Processed'
        os.makedirs(processed_folder, exist_ok=True)
        shutil.move(zip_path, processed_folder)

    print(f"Saved to {filepath}")
