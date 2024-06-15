from dataclasses import dataclass
from b3d.types import Array
import jax.numpy as jnp
from b3d.camera import Intrinsics
from typing import Optional


@dataclass
class FeatureTrackData:
    """
    Feature track data class. Note: Spatial units are measured in meters.

    Args:
            observed_keypoints_positions: (T, N, 2) Float Array
            keypoint_visibility: (T, N) Boolean Array
            camera_intrinsics: (8,) Float Array of camera intrinsics, see `camera.py`.
            rgbd_imgs: (T, H, W, 4) Float Array
            observed_features (optional): (T, N, F) Float Array OR None
            latent_keypoint_positions (optional): (T, N, 3) Float Array OR None
            latent_keypoint_quaternions (optional): (T, N, 4) Float Array OR None
            object_assignments (optional): (N,) Int Array OR None
            camera_position (optional): (T, 3) Float Array OR None
            camera_quaternion (optional): (T, 4) Float Array OR None
    """
    # Required fields: Observed Data
    observed_keypoints_positions: Array
    keypoint_visibility: Array
    camera_intrinsics: Array
    rgbd_imgs: Array
    # Optional fields: Ground truth data
    observed_features: Optional[Array]
    latent_keypoint_positions: Optional[Array]
    latent_keypoint_quaternions: Optional[Array]
    object_assignments: Optional[Array]
    camera_position: Optional[Array]
    camera_quaternion: Optional[Array]

    def __init__(self,
                observed_keypoints_positions: Array,
                keypoint_visibility: Array,
                rgb_imgs: Array,
                camera_intrinsics: Array,
                observed_features: Optional[Array] = None,
                latent_keypoint_positions: Optional[Array] = None,
                latent_keypoint_quaternions: Optional[Array] = None,
                object_assignments: Optional[Array] = None,
                camera_position: Optional[Array] = None,
                camera_quaternion: Optional[Array] = None):
        self.observed_keypoints_positions = observed_keypoints_positions
        self.observed_features = observed_features
        self.keypoint_visibility = keypoint_visibility
        self.camera_intrinsics = camera_intrinsics
        self.rgb_imgs = rgb_imgs
        self.latent_keypoint_positions = latent_keypoint_positions
        self.latent_keypoint_quaternions = latent_keypoint_quaternions
        self.object_assignments = object_assignments
        self.camera_position = camera_position
        self.camera_quaternion = camera_quaternion

    @property
    def uv(self): return self.observed_keypoints_positions

    @property
    def visibility(self): return self.keypoint_visibility

    @property
    def rgb(self): return self.latent_keypoint_positions

    @property
    def camera_poses(self): return Pose(self.camera_position, self.camera_quaternion)

    @property
    def intrinsics(self):
        """Returns camera intrinsics as an Intrinsics object"""
        return Intrinsics.from_array(self.camera_intrinsics)

    def save(self, filepath: str):
        """Saves input to file"""
        to_save = dict(
            latent_keypoint_positions=self.latent_keypoint_positions,
            latent_keypoint_quaternions=self.latent_keypoint_quaternions,
            observed_keypoints_positions=self.observed_keypoints_positions,
            observed_features=self.observed_features,
            rgb_imgs=self.rgb_imgs,
            keypoint_visibility=self.keypoint_visibility,
            object_assignments=self.object_assignments,
            camera_position=self.camera_position,
            camera_quaternion=self.camera_quaternion,
            camera_intrinsics=self.camera_intrinsics,
        )
        to_save = {k: v for k, v in to_save.items() if v is not None}
        jnp.savez(filepath, **to_save)

    @classmethod
    def load(cls, filepath: str):
        """Loads input from file"""
        with open(filepath, "rb") as f:
            data = jnp.load(f, allow_pickle=True)

            def get_or_none(data, key):
                if key in data:
                    return jnp.array(data[key])
                else:
                    return None

            return cls(
                latent_keypoint_positions=get_or_none(
                    data, "latent_keypoint_positions"
                ),
                latent_keypoint_quaternions=get_or_none(
                    data, "latent_keypoint_quaternions"
                ),
                observed_keypoints_positions=get_or_none(
                    data, "observed_keypoints_positions"
                ),
                observed_features=get_or_none(data, "observed_features"),
                rgb_imgs=get_or_none(data, "rgb_imgs"),
                keypoint_visibility=get_or_none(
                    data, "keypoint_visibility"
                ),
                object_assignments=get_or_none(data, "object_assignments"),
                camera_position=get_or_none(data, "camera_position"),
                camera_quaternion=get_or_none(data, "camera_quaternion"),
                camera_intrinsics=get_or_none(data, "camera_intrinsics"),
            )

