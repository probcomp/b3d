import jax.numpy as jnp
import numpy as np
import jax
from jax.scipy.spatial.transform import Rotation as Rot
from pose import Pose
from b3d.camera import screen_from_camera, Intrinsics, unproject_depth
from path_utils import get_assets_path
from unity_data import UnityData
from b3d.io.feature_track_data import FeatureTrackData

# Rotation of 90 degrees along the x-axis
QUAT_90_DEG_X = Rot.from_rotvec(jnp.array([1., 0., 0.]) * jnp.pi / 2).as_quat()
QUAT_90_DEG_X_Pose = Pose(np.array([0., 0., 0.]), QUAT_90_DEG_X)

def convert_unity_to_cv2_object_pos(original_pos: jnp.ndarray) -> jnp.ndarray:
    """Convert Unity position to OpenCV position."""
    return jnp.array([original_pos[0], original_pos[2], original_pos[1]])

def convert_unity_to_cv2_camera_pos_and_quat(
    original_pos: jnp.ndarray, original_rot: jnp.ndarray, z_up: bool = False
) -> Pose:
    """Convert Unity position and quaternion to OpenCV position and quaternion."""
    pose_unity = Pose(original_pos, original_rot)
    pose_matrix = pose_unity.as_matrix()

    x_axis_of_cam = pose_matrix[:3, 0] * jnp.array([1, 1, -1])
    y_axis_of_cam = pose_matrix[:3, 1] * jnp.array([1, 1, -1])
    z_axis_of_cam = pose_matrix[:3, 2] * jnp.array([1, 1, -1])

    new_rotation_matrix = jnp.hstack([
        x_axis_of_cam.reshape(3, 1),
        -y_axis_of_cam.reshape(3, 1),
        z_axis_of_cam.reshape(3, 1)
    ])

    camera_pos = jnp.array([original_pos[0], original_pos[1], -original_pos[2]])
    new_pose = Pose(camera_pos, Rot.from_matrix(new_rotation_matrix).quat)

    new_pose = QUAT_90_DEG_X_Pose @ new_pose

    if z_up:
        new_pose = new_pose @ QUAT_90_DEG_X_Pose

    return new_pose.pos, new_pose._quaternion

def project_to_screen(
    x: jnp.ndarray, camera_pos: jnp.ndarray, camera_quat: jnp.ndarray, intr: jnp.ndarray
) -> jnp.ndarray:
    """Project 3D points into 2D camera view."""
    intr = Intrinsics(*intr)
    cam = Pose(camera_pos, camera_quat)
    return screen_from_camera(cam.inv().apply(x), intr)

def clamp_and_replace_with_nan(
    obs_array: np.ndarray, width: int, height: int
) -> np.ndarray:
    """Clamp values within specified ranges and replace out-of-bounds values with NaN."""
    mask_x = (obs_array[..., 0] < 0) | (obs_array[..., 0] >= width)
    mask_y = (obs_array[..., 1] < 0) | (obs_array[..., 1] >= height)

    array_clamped = np.copy(obs_array)
    array_clamped[..., 0] = np.clip(obs_array[..., 0], 0, width)
    array_clamped[..., 1] = np.clip(obs_array[..., 1], 0, height)

    array_clamped[..., 0][mask_x] = np.nan
    array_clamped[..., 1][mask_y] = np.nan

    return array_clamped

def convert_rgb_float_to_uint(rgb: jnp.ndarray) -> jnp.ndarray:
    """Convert RGB floats to uint8."""
    return (rgb * 255).astype(jnp.uint8)

def convert_depth_float_to_xyz(
    depth: jnp.ndarray, intrinsics: jnp.ndarray
) -> jnp.ndarray:
    """Unproject depth floats into xyz."""
    return unproject_depth(depth, intrinsics)

def is_point_in_front(
    point_position: jnp.ndarray, camera_position: jnp.ndarray, camera_quaternion: jnp.ndarray
) -> bool:
    """Check if a point is in front of the camera."""
    camera_pose = Pose(camera_position, camera_quaternion)
    point_camera = camera_pose.inv().apply(point_position)
    return jnp.where(point_camera[2] >= 0, True, False)

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

    depth_expanded = np.expand_dims(unity_data.depth, axis=-1)
    rgbd = np.concatenate((unity_data.rgb, depth_expanded), axis=-1)

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

def convert(zip_path: str) -> FeatureTrackData:
    unity_data = UnityData.from_zip(zip_path)
    feature_track_data = convert_unity_to_feature_track(unity_data)
    return feature_track_data

def process(zip_path: str) -> None:
    """Process a ZIP file and save the feature track data."""
    unity_data = UnityData.from_zip(zip_path)
    feature_track_data = convert_unity_to_feature_track(unity_data)
    
    file_info = unity_data.file_info
    filepath = get_assets_path('f', file_info['scene_folder'], file_info['base_name']) + f"/{file_info['light_setting']}_{file_info['background_setting']}_{file_info['resolution']}.input.npz"
    
    feature_track_data.save(filepath)
    print(f"Saved to {filepath}")
