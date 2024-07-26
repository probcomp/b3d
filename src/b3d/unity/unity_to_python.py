import jax.numpy as jnp
import jax
from jax.scipy.spatial.transform import Rotation as Rot
from functools import partial
from pose import Pose
from b3d.camera import screen_from_camera, Intrinsics, unproject_depth

# Rotation of 90 degrees along the x-axis
QUAT_90_DEG_X = Rot.from_rotvec(jnp.array([1.0, 0.0, 0.0]) * jnp.pi / 2).as_quat()
QUAT_90_DEG_X_Pose = Pose(jnp.array([0.0, 0.0, 0.0]), QUAT_90_DEG_X)


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

    new_rotation_matrix = jnp.hstack(
        [
            x_axis_of_cam.reshape(3, 1),
            -y_axis_of_cam.reshape(3, 1),
            z_axis_of_cam.reshape(3, 1),
        ]
    )

    camera_pos = jnp.array([original_pos[0], original_pos[1], -original_pos[2]])
    new_pose = Pose(camera_pos, Rot.from_matrix(new_rotation_matrix).quat)

    new_pose = QUAT_90_DEG_X_Pose @ new_pose

    if z_up:
        new_pose = new_pose @ QUAT_90_DEG_X_Pose

    return new_pose.pos, new_pose._quaternion


def convert_unity_to_cv2_world_pos_and_quat(
    original_pos: jnp.ndarray, original_rot: jnp.ndarray, z_up: bool = True
) -> Pose:
    """Convert Unity position and quaternion to OpenCV position and quaternion."""
    pose_unity = Pose(original_pos, original_rot)
    pose_matrix = pose_unity.as_matrix()

    x_axis_of_cam = pose_matrix[:3, 0] * jnp.array([1, 1, -1])
    y_axis_of_cam = pose_matrix[:3, 1] * jnp.array([1, 1, -1])
    z_axis_of_cam = pose_matrix[:3, 2] * jnp.array([1, 1, -1])

    new_rotation_matrix = jnp.hstack(
        [
            x_axis_of_cam.reshape(3, 1),
            -y_axis_of_cam.reshape(3, 1),
            z_axis_of_cam.reshape(3, 1),
        ]
    )

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
    obs_array: jnp.ndarray, width: int, height: int
) -> jnp.ndarray:
    """Clamp values within specified ranges and replace out-of-bounds values with NaN."""
    mask_x = (obs_array[..., 0] < 0) | (obs_array[..., 0] >= width)
    mask_y = (obs_array[..., 1] < 0) | (obs_array[..., 1] >= height)

    array_clamped = jnp.copy(obs_array)
    array_clamped = array_clamped.at[..., 0].set(jnp.clip(obs_array[..., 0], 0, width))
    array_clamped = array_clamped.at[..., 1].set(jnp.clip(obs_array[..., 1], 0, height))

    array_clamped = array_clamped.at[..., 0].set(
        jnp.where(mask_x, jnp.nan, array_clamped[..., 0])
    )
    array_clamped = array_clamped.at[..., 1].set(
        jnp.where(mask_y, jnp.nan, array_clamped[..., 1])
    )

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
    point_position: jnp.ndarray,
    camera_position: jnp.ndarray,
    camera_quaternion: jnp.ndarray,
) -> bool:
    """Check if a point is in front of the camera."""
    camera_pose = Pose(camera_position, camera_quaternion)
    point_camera = camera_pose.inv().apply(point_position)
    return jnp.where(point_camera[2] >= 0, True, False)


def downsize_2d_coordinates(points, k):
    resized_points = points.at[..., 0].mul(1 / k).at[..., 1].mul(1 / k)
    return resized_points


@partial(jax.jit, static_argnums=1)
def downsize_single_channel_image(ims, k):
    """Downsize an array of images by a given factor."""
    shape = (ims.shape[1] // k, ims.shape[2] // k)
    return jax.vmap(jax.image.resize, (0, None, None))(ims, shape, "linear")


def downsize_intrinsics(intrinsics, k):
    """Adjust camera intrinsics for the downscaled images."""
    adjusted_intrinsics = jnp.array(intrinsics.copy())
    adjusted_intrinsics = adjusted_intrinsics.at[0].mul(1/k)  # width
    adjusted_intrinsics = adjusted_intrinsics.at[1].mul(1/k)  # height
    adjusted_intrinsics = adjusted_intrinsics.at[2].mul(1/k)  # fx
    adjusted_intrinsics = adjusted_intrinsics.at[3].mul(1/k)  # fy
    adjusted_intrinsics = adjusted_intrinsics.at[4].mul(1/k)  # cx
    adjusted_intrinsics = adjusted_intrinsics.at[5].mul(1/k)  # cy
    return adjusted_intrinsics
